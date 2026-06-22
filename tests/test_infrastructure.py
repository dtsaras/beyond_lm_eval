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
        # Future additions should bump this number.
        # Current registry: 74 diagnostic tasks across 7 categories.
        assert len(tasks) == 74
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
        assert payload["count"] == 27
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
        assert m is not None and np.isfinite(m["svd_auc"])
