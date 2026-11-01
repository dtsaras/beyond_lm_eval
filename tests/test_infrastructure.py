"""
Tests for the BLME model loader, results module, and CLI infrastructure.
"""

import json
import os
import pytest
import torch
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# wrapper.py — argument parsing (no model download needed)
# ---------------------------------------------------------------------------

from blme.models.wrapper import parse_model_args, _parse_bool, _resolve_dtype, _parse_max_memory


class TestParseModelArgs:
    def test_basic(self):
        args = parse_model_args("pretrained=gpt2,dtype=float16")
        assert args == {"pretrained": "gpt2", "dtype": "float16"}

    def test_empty_string(self):
        assert parse_model_args("") == {}

    def test_none_string(self):
        assert parse_model_args(None) == {}

    def test_spaces(self):
        args = parse_model_args("pretrained = gpt2 , dtype = bfloat16")
        assert args["pretrained"] == "gpt2"
        assert args["dtype"] == "bfloat16"

    def test_complex_value_with_equals(self):
        # values like paths can have = in them
        args = parse_model_args("pretrained=/path/to/model")
        assert args["pretrained"] == "/path/to/model"

    def test_all_supported_keys(self):
        s = "pretrained=gpt2,dtype=bfloat16,device_map=auto,trust_remote_code=true,attn_implementation=sdpa,revision=main,load_in_8bit=false,load_in_4bit=false"
        args = parse_model_args(s)
        assert len(args) == 8
        assert args["pretrained"] == "gpt2"
        assert args["device_map"] == "auto"
        assert args["attn_implementation"] == "sdpa"


class TestParseBool:
    def test_true_variants(self):
        for val in ["true", "True", "TRUE", "1", "yes"]:
            assert _parse_bool(val) is True

    def test_false_variants(self):
        for val in ["false", "False", "0", "no", "anything"]:
            assert _parse_bool(val) is False


class TestResolveDtype:
    def test_float16(self):
        assert _resolve_dtype("float16") == torch.float16
        assert _resolve_dtype("fp16") == torch.float16

    def test_bfloat16(self):
        assert _resolve_dtype("bfloat16") == torch.bfloat16
        assert _resolve_dtype("bf16") == torch.bfloat16

    def test_float32(self):
        assert _resolve_dtype("float32") == torch.float32
        assert _resolve_dtype("fp32") == torch.float32

    def test_auto(self):
        assert _resolve_dtype("auto") == "auto"

    def test_case_insensitive(self):
        assert _resolve_dtype("Float16") == torch.float16
        assert _resolve_dtype("BFLOAT16") == torch.bfloat16

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            _resolve_dtype("int8")


class TestParseMaxMemory:
    def test_single_gpu(self):
        result = _parse_max_memory("0:20GiB")
        assert result == {0: "20GiB"}

    def test_multi_device(self):
        result = _parse_max_memory("0:20GiB,1:20GiB,cpu:40GiB")
        assert result == {0: "20GiB", 1: "20GiB", "cpu": "40GiB"}


# ---------------------------------------------------------------------------
# wrapper.py — model loading (mocked)
# ---------------------------------------------------------------------------

class TestLoadModelAndTokenizer:
    @patch("blme.models.wrapper.AutoTokenizer")
    @patch("blme.models.wrapper._load_model")
    def test_basic_load(self, mock_load, mock_tok_cls):
        # Setup mocks
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.dtype = torch.float32
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.to.return_value = mock_model  # .to() returns self
        mock_load.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 2
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from blme.models.wrapper import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer("pretrained=gpt2", device="cpu")

        assert model is mock_model
        mock_model.eval.assert_called_once()
        assert tokenizer.pad_token == "<eos>"


class TestModelKwargsCompat:
    """Campaign-3 release blockers in the model loader (transformers v5)."""

    def _model_kwargs_for(self, model_args):
        with patch("blme.models.wrapper.AutoTokenizer") as mock_tok_cls, \
             patch("blme.models.wrapper._load_model") as mock_load:
            mock_model = MagicMock()
            mock_param = MagicMock()
            mock_param.dtype = torch.float32
            mock_param.device = torch.device("cpu")
            mock_model.parameters.return_value = iter([mock_param])
            mock_model.to.return_value = mock_model
            mock_load.return_value = mock_model
            tok = MagicMock()
            tok.pad_token = None
            tok.eos_token = "<eos>"
            tok.eos_token_id = 2
            mock_tok_cls.from_pretrained.return_value = tok
            from blme.models.wrapper import load_model_and_tokenizer
            load_model_and_tokenizer(model_args, device="cpu")
            return mock_load.call_args[0][1]  # the model_kwargs dict

    def test_dtype_kwarg_replaces_deprecated_torch_dtype(self):
        """transformers v5 deprecates torch_dtype= in favour of dtype=; passing
        torch_dtype prints a deprecation warning on every load."""
        kw = self._model_kwargs_for("pretrained=gpt2,dtype=bfloat16")
        assert kw.get("dtype") == torch.bfloat16
        assert "torch_dtype" not in kw

    def test_dtype_auto_uses_dtype_kwarg(self):
        kw = self._model_kwargs_for("pretrained=gpt2,dtype=auto")
        assert kw.get("dtype") == "auto"
        assert "torch_dtype" not in kw

    def test_max_memory_semicolon_survives_as_multi_device(self):
        """Comma-delimited max_memory is mangled by parse_model_args (which
        splits on ','); the documented semicolon form must survive intact."""
        kw = self._model_kwargs_for("pretrained=gpt2,max_memory=0:20GiB;cpu:40GiB")
        assert kw["max_memory"] == {0: "20GiB", "cpu": "40GiB"}

    def test_load_in_8bit_uses_quantization_config(self):
        """Passing load_in_8bit= directly to from_pretrained raises TypeError on
        transformers v5; it must be wrapped in a BitsAndBytesConfig."""
        from transformers import BitsAndBytesConfig
        kw = self._model_kwargs_for("pretrained=gpt2,load_in_8bit=true")
        assert "load_in_8bit" not in kw
        assert isinstance(kw.get("quantization_config"), BitsAndBytesConfig)
        assert kw["quantization_config"].load_in_8bit is True

    def test_load_in_4bit_uses_quantization_config(self):
        from transformers import BitsAndBytesConfig
        kw = self._model_kwargs_for("pretrained=gpt2,load_in_4bit=true")
        assert "load_in_4bit" not in kw
        assert isinstance(kw.get("quantization_config"), BitsAndBytesConfig)
        assert kw["quantization_config"].load_in_4bit is True


def test_force_eager_covers_all_attention_tasks():
    """Campaign-3 release blocker: the force-eager list was incomplete
    (missing geometry_positional_decay and interpretability_attention_rollout),
    so those tasks got SDPA's all-None attentions and silently failed on a
    default model load. Every task that reads outputs.attentions /
    cache.get_attentions() must be in _ATTENTION_TASKS."""
    import re
    from pathlib import Path
    from blme.core import _ATTENTION_TASKS

    tasks_root = Path(__file__).resolve().parents[1] / "src" / "blme" / "tasks"
    consuming = set()
    for py in tasks_root.rglob("*.py"):
        src = py.read_text()
        if "output_attentions=True" in src or ".attentions" in src or "get_attentions" in src:
            for name in re.findall(r'register_task\(\s*"([a-z_]+)"', src):
                consuming.add(name)

    missing = consuming - set(_ATTENTION_TASKS)
    assert not missing, (
        f"attention-consuming tasks not forced to eager attention: {sorted(missing)}"
    )
    # the two the finding flagged must be present
    assert "geometry_positional_decay" in _ATTENTION_TASKS
    assert "interpretability_attention_rollout" in _ATTENTION_TASKS


def test_task_timeout_survives_broad_except():
    """Campaign-3 release blocker: the SIGALRM handler raised TimeoutError, which
    a task's own `except Exception` swallowed, silently defeating the per-task
    timeout (the timed-out task was then reported OK). TaskTimeout must derive
    from BaseException so `except Exception` cannot catch it."""
    from blme.core import TaskTimeout

    assert issubclass(TaskTimeout, BaseException)
    assert not issubclass(TaskTimeout, Exception)

    swallowed = False
    try:
        try:
            raise TaskTimeout("timeout")  # as the SIGALRM handler now raises
        except Exception:                 # a task's broad handler
            swallowed = True
    except TaskTimeout:
        pass  # the core loop catches it here
    assert not swallowed, "TaskTimeout must not be catchable by `except Exception`"


def test_load_model_rejects_bare_name_instead_of_loading_gpt2():
    """Campaign-3 release blocker: a bare model name (no 'pretrained=') was
    dropped by the parser and silently fell back to gpt2 — evaluating the wrong
    model. It must raise instead."""
    import pytest
    from blme.models.wrapper import load_model_and_tokenizer

    with pytest.raises(ValueError, match="pretrained"):
        load_model_and_tokenizer("meta-llama/Llama-3-8B", device="cpu")

    @patch("blme.models.wrapper.AutoTokenizer")
    @patch("blme.models.wrapper._load_model")
    def test_dtype_passed_through(self, mock_load, mock_tok_cls):
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.dtype = torch.float16
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_load.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from blme.models.wrapper import load_model_and_tokenizer
        load_model_and_tokenizer("pretrained=gpt2,dtype=float16", device="cpu")

        # Verify _load_model was called with torch_dtype
        call_kwargs = mock_load.call_args[0][1]
        assert call_kwargs["torch_dtype"] == torch.float16

    @patch("blme.models.wrapper.AutoTokenizer")
    @patch("blme.models.wrapper._load_model")
    def test_device_map_skips_to(self, mock_load, mock_tok_cls):
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.dtype = torch.float32
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_load.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from blme.models.wrapper import load_model_and_tokenizer
        load_model_and_tokenizer("pretrained=gpt2,device_map=auto", device="cpu")

        # model.to() should NOT be called when device_map is used
        mock_model.to.assert_not_called()

    @patch("blme.models.wrapper.AutoTokenizer")
    @patch("blme.models.wrapper._load_model")
    def test_trust_remote_code(self, mock_load, mock_tok_cls):
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.dtype = torch.float32
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_load.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from blme.models.wrapper import load_model_and_tokenizer
        load_model_and_tokenizer(
            "pretrained=gpt2,trust_remote_code=true", device="cpu"
        )

        # Check tokenizer was called with trust_remote_code
        tok_kwargs = mock_tok_cls.from_pretrained.call_args
        assert tok_kwargs[1].get("trust_remote_code") is True

        # Check model was called with trust_remote_code
        model_kwargs = mock_load.call_args[0][1]
        assert model_kwargs.get("trust_remote_code") is True


# ---------------------------------------------------------------------------
# results.py
# ---------------------------------------------------------------------------

from blme.results import build_results_envelope, print_results_table, save_results


class TestBuildResultsEnvelope:
    def test_structure(self):
        env = build_results_envelope(
            model_args="pretrained=gpt2",
            tasks_requested=["geometry_svd", "geometry_cka"],
            task_results={"geometry_svd": {"isotropy": 0.5}},
            task_errors={"geometry_cka": "division by zero"},
            device="cpu",
        )
        assert env["blme_version"]
        assert env["timestamp"]
        assert env["config"]["model_args"] == "pretrained=gpt2"
        assert env["summary"]["total_tasks"] == 2
        assert env["summary"]["completed_tasks"] == 1
        assert env["summary"]["failed_tasks"] == 1
        assert "geometry_svd" in env["results"]
        assert "geometry_cka" in env["errors"]

    def test_no_errors(self):
        env = build_results_envelope(
            model_args="pretrained=gpt2",
            tasks_requested=["geometry_svd"],
            task_results={"geometry_svd": {"isotropy": 0.5}},
            task_errors={},
            device="cpu",
        )
        assert env["errors"] is None

    def test_error_dict_result_counted_as_failure(self):
        """Campaign-3 release blocker: a task returning {"error": ...} was
        stored in task_results and counted as completed, so failed_tasks=0 and
        errors=None despite the failure. It must count as a failure and surface
        in errors."""
        env = build_results_envelope(
            model_args="pretrained=gpt2",
            tasks_requested=["good", "bad", "raised"],
            task_results={"good": {"metric": 1.0}, "bad": {"error": "boom"}},
            task_errors={"raised": "exception text"},
            device="cpu",
        )
        assert env["summary"]["completed_tasks"] == 1, "only 'good' truly completed"
        assert env["summary"]["failed_tasks"] == 2, "'bad' (error dict) + 'raised'"
        assert env["errors"]["bad"] == "boom"
        assert env["errors"]["raised"] == "exception text"


class TestPrintResultsTable:
    def test_no_crash(self, capsys):
        print_results_table(
            {"geometry_svd": {"isotropy": 0.42}, "bad_task": {"error": "oops"}},
            {"crashed_task": "exception text"},
        )
        captured = capsys.readouterr()
        assert "geometry_svd" in captured.out
        assert "bad_task" in captured.out
        assert "crashed_task" in captured.out


class TestSaveResults:
    def test_save_json(self, tmp_path):
        env = build_results_envelope(
            model_args="pretrained=gpt2",
            tasks_requested=["geometry_svd"],
            task_results={"geometry_svd": {"val": 1.0}},
            task_errors={},
            device="cpu",
        )
        path = save_results(env, str(tmp_path), "json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["config"]["model_args"] == "pretrained=gpt2"

    def test_save_csv(self, tmp_path):
        env = build_results_envelope(
            model_args="pretrained=gpt2",
            tasks_requested=["geometry_svd"],
            task_results={"geometry_svd": {"val": 1.0}},
            task_errors={},
            device="cpu",
        )
        path = save_results(env, str(tmp_path), "csv")
        assert os.path.exists(path)
        assert path.endswith(".csv")

    def test_unknown_format_raises(self, tmp_path):
        env = build_results_envelope(
            model_args="", tasks_requested=[], task_results={},
            task_errors={}, device="cpu",
        )
        with pytest.raises(ValueError, match="Unknown output format"):
            save_results(env, str(tmp_path), "xml")


# ---------------------------------------------------------------------------
# registry.py
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_tasks_registered(self):
        from blme.core import _register_all_tasks
        from blme.registry import list_tasks

        _register_all_tasks()
        tasks = list_tasks()
        # Original library: 50 tasks.
        # Phase B added: IsoScore, Contextualization, AttentionRankCollapse (+3)
        # Phase C1-C5 added: PositionSensitivity, FormatRobustness,
        #   NeuralCollapse, SelfConsistency, RefusalDirection (+5)
        # Future additions should bump this number (tests/test_completeness.py
        # lists every task).
        assert len(tasks) == 98
        # Spot check some representative tasks
        assert "geometry_svd" in tasks
        assert "geometry_isoscore" in tasks
        assert "geometry_contextualization" in tasks
        assert "geometry_neural_collapse" in tasks
        assert "interpretability_attention_entropy" in tasks
        assert "interpretability_attention_rank" in tasks
        assert "causality_tracing" in tasks
        assert "topology_homology" in tasks
        assert "consistency_calibration" in tasks
        assert "consistency_position_sensitivity" in tasks
        assert "consistency_format_robustness" in tasks
        assert "consistency_self_consistency" in tasks
        assert "dynamics_stability" in tasks
        assert "repe_task_vectors" in tasks
        assert "repe_refusal_direction" in tasks
        assert "causality_knowledge_neurons" in tasks
        assert "causality_edge_attribution" in tasks
        assert "dynamics_sharpness" in tasks
        assert "interpretability_head_roles" in tasks
        # Phase D
        assert "topology_persistence_landscape" in tasks
        assert "consistency_bias_weat" in tasks
        assert "dynamics_generation_diversity" in tasks
        assert "consistency_membership_inference" in tasks
        # Phase E
        assert "dynamics_gradient_flow" in tasks
        assert "geometry_weight_norms" in tasks
        assert "geometry_tokenizer_efficiency" in tasks
        assert "consistency_icl_slope" in tasks


# ---------------------------------------------------------------------------
# cli.py — argument parsing (no model download)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_list_tasks_runs(self, capsys):
        """Test that list-tasks subcommand works."""
        import sys
        from blme.cli import main

        with patch.object(sys, "argv", ["blme", "list-tasks"]):
            main()
        captured = capsys.readouterr()
        assert "geometry" in captured.out.lower()
        assert "Available BLME tasks" in captured.out

    def test_list_tasks_group_filter(self, capsys):
        import sys
        from blme.cli import main

        with patch.object(sys, "argv", ["blme", "list-tasks", "--group", "topology"]):
            main()
        captured = capsys.readouterr()
        assert "topology_homology" in captured.out
        # Should NOT contain other groups
        assert "geometry_svd" not in captured.out

    def test_list_tasks_json_runs(self, capsys):
        import json
        import sys
        from blme.cli import main

        with patch.object(sys, "argv", ["blme", "list-tasks", "--group", "geometry", "--json"]):
            main()
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 43
        first = payload["tasks"][0]
        assert {"name", "group", "status", "papers"} <= set(first)
        assert all(task["group"] == "geometry" for task in payload["tasks"])

    def test_evaluate_dry_run_validates_before_model_load(self, capsys):
        import json
        import sys
        from blme.cli import main

        with patch.object(sys, "argv", [
            "blme", "evaluate", "--model-args", "pretrained=test",
            "--tasks", "geometry_svd", "--dry-run", "--strict",
        ]):
            main()
        payload = json.loads(capsys.readouterr().out)
        assert payload["diagnostic_tasks"] == ["geometry_svd"]
        assert payload["unknown_tasks"] == []
        assert payload["task_metadata"]["geometry_svd"]["status"]

    def test_evaluate_strict_rejects_unknown_before_model_load(self):
        import sys
        from blme.cli import main

        with patch.object(sys, "argv", [
            "blme", "evaluate", "--model-args", "pretrained=test",
            "--tasks", "not_a_task", "--dry-run", "--strict",
        ]):
            with pytest.raises(SystemExit):
                main()

    def test_evaluate_no_args_exits(self):
        import sys
        from blme.cli import main

        with patch.object(sys, "argv", ["blme", "evaluate"]):
            with pytest.raises(SystemExit):
                main()

    def test_no_subcommand_exits(self):
        import sys
        from blme.cli import main

        with patch.object(sys, "argv", ["blme"]):
            with pytest.raises(SystemExit):
                main()


# ---------------------------------------------------------------------------
# NumPy 2.x compatibility — the eager `getattr(np, "trapezoid", np.trapz)`
# pattern raises AttributeError on NumPy>=2.0 (np.trapz removed; the default
# arg is evaluated before getattr). Verified bug: it crashed the metric path
# of geometry_svd, causality_ablation, and topology_persistence_landscape.
# ---------------------------------------------------------------------------


class TestNumpyTrapezoidCompat:
    def test_no_eager_trapz_getattr_in_source(self):
        """Guard against reintroducing the eager-eval np.trapz pattern."""
        import pathlib
        import re

        src = pathlib.Path(__file__).resolve().parents[1] / "src" / "blme"
        bad = re.compile(r"getattr\(\s*np\s*,\s*[\"']trapezoid[\"']\s*,\s*np\.trapz\s*\)")
        offenders = []
        for py in src.rglob("*.py"):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                # skip comment lines that merely document the anti-pattern
                if line.lstrip().startswith("#"):
                    continue
                if bad.search(line):
                    offenders.append(f"{py}:{i}")
        assert not offenders, f"eager np.trapz pattern present (crashes NumPy 2.x): {offenders}"

    def test_landscape_stats_runs_under_installed_numpy(self):
        import numpy as np
        from blme.tasks.topology.persistence_landscape import (
            _compute_landscape, _landscape_stats,
        )
        L = _compute_landscape(np.array([[0.0, 2.0], [0.5, 3.0]]), 5, 200)
        stats = _landscape_stats(L, dt=0.01)
        assert np.isfinite(stats["mean_landscape_integral"])

    def test_svd_metrics_runs_under_installed_numpy(self):
        import numpy as np
        from blme.tasks.geometry.isotropy import _svd_metrics_for_layer
        m = _svd_metrics_for_layer(torch.randn(50, 8))
        assert m is not None and np.isfinite(m["_meta_svd_auc"])

    def test_no_bare_np_trapezoid_calls_in_source(self):
        """Campaign-4 B5: magnitude.py called np.trapezoid(...) directly,
        which is NumPy>=2.0-only — REPRODUCE.md pins numpy<2.0, so
        evaluate() crashed with AttributeError there. Every call site must
        use the lazy `np.trapezoid if hasattr(np, "trapezoid") else
        np.trapz` selection instead of a bare call."""
        import pathlib
        import re

        src = pathlib.Path(__file__).resolve().parents[1] / "src" / "blme"
        bare_call = re.compile(r"np\.trapezoid\s*\(")
        offenders = []
        for py in src.rglob("*.py"):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if bare_call.search(line):
                    offenders.append(f"{py}:{i}")
        assert not offenders, (
            f"bare np.trapezoid( calls (crash on NumPy 1.x): {offenders}"
        )

    def test_magnitude_summary_runs_under_installed_numpy(self):
        import numpy as np
        from blme.tasks.geometry.magnitude import _magnitude_summary
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4))
        sq = np.sum(X * X, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
        np.fill_diagonal(D, 0.0)
        out = _magnitude_summary(D, 6, 10.0)
        assert np.isfinite(out["magnitude_auc"])


def test_force_eager_covers_double_backward_tasks():
    """Regression (campaign-4 B3): dynamics_sharpness needs SECOND-order
    autograd (HVPs), which SDPA/Flash kernels don't implement — on a default
    model load the task always returned its SDPA error and the study got no
    sharpness features. The diagnostic loop must force eager for it."""
    from blme.core import _ATTENTION_TASKS, _DOUBLE_BACKWARD_TASKS

    assert "dynamics_sharpness" in _DOUBLE_BACKWARD_TASKS
    # And the union actually reaches the force-eager check.
    import inspect
    from blme import core
    src = inspect.getsource(core)
    assert "_ATTENTION_TASKS | _DOUBLE_BACKWARD_TASKS" in src
