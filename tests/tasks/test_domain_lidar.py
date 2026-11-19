import json
import math

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from blme.core import _register_all_tasks
from blme.registry import get_task
from blme.tasks.geometry.corpus_metrics import _document_nll, paired_lidar


@pytest.mark.parametrize(("task", "field"), [("geometry_domain_bpb", "documents"),
                                           ("geometry_paraphrase_lidar", "triples")])
def test_structured_panel_reports_missing_fields(task, field):
    _register_all_tasks()
    with pytest.raises(ValueError, match=f"Invalid {field} row 0"):
        get_task(task)({}).evaluate(None, None, [{"text": "unlabeled passage"}])


class ByteTokenizer:
    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = None

    def __call__(self, texts, **kw):
        return {"input_ids": [list(t.encode("utf-8")) for t in texts]}

    def batch_decode(self, seqs, **kw):
        return [bytes(seq).decode("utf-8") for seq in seqs]


@pytest.fixture
def lm():
    torch.manual_seed(34)
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=258, n_positions=96)).eval()
    _register_all_tasks()
    return model


def dense_rank(a, b, rel=.01):
    n, d = a.shape
    means = (a+b)/2
    between = np.cov(means.T)
    within = (a-b).T @ (a-b) / (2*n)
    ridge = rel*(np.trace(within)+np.trace(between))/(n-1)
    vals, vec = np.linalg.eigh(within+ridge*np.eye(d))
    invsqrt = (vec*(1/np.sqrt(vals)))@vec.T
    eig = np.linalg.eigvalsh(invsqrt @ between @ invsqrt)
    p = np.maximum(eig,0)/eig.sum()
    p = p[p>0]
    return np.exp(-(p*np.log(p)).sum())


@pytest.mark.parametrize("n,d", [(20,7),(9,28)])
def test_paired_lidar_matches_dense_whitening_and_invariances(n,d):
    rng=np.random.default_rng(9)
    a,b=rng.normal(size=(2,n,d))
    expected=dense_rank(a,b)
    assert paired_lidar(a,b)==pytest.approx(expected, rel=1e-10)
    q=np.linalg.qr(rng.normal(size=(d,d)))[0]
    assert paired_lidar(7*a@q+3,7*b@q+3)==pytest.approx(expected, rel=1e-10)
    assert paired_lidar(np.pad(a,((0,0),(0,5))),np.pad(b,((0,0),(0,5))))==pytest.approx(expected)
    assert paired_lidar(b,a)==pytest.approx(expected)


def test_planted_dimension_and_undefined():
    a=np.eye(9)
    assert paired_lidar(a,a)==pytest.approx(8)
    with pytest.raises(ValueError):
        paired_lidar(a,-a)
    with pytest.raises(ValueError):
        paired_lidar(a*0,a*0)


def test_bpb_counts_all_content_bytes_and_all_first_tokens(lm):
    tok=ByteTokenizer()
    rows=[{"id":"x","text":"éclair","domain":"prose","block":0},{"id":"y","text":"x","domain":"prose","block":0},
          {"id":"z","text":"µ+y","domain":"math","block":0}]
    result=get_task("geometry_domain_bpb")({"batch_size":3}).evaluate(lm,tok,rows)
    total=0
    for row,got in zip(rows,result["documents"]):
        ids=torch.tensor([[tok.bos_token_id,*row["text"].encode()]])
        with torch.no_grad():
            logp=lm(ids).logits[0,:-1].double().log_softmax(-1)
        nll=-float(logp[torch.arange(ids.shape[1]-1),ids[0,1:]].sum())
        assert got["nll_nats"]==pytest.approx(nll,rel=1e-6)
        assert got["bytes"]==got["tokens"]==len(row["text"].encode())
        if row["domain"]=="prose": total+=nll
    assert result["by_domain"]["prose"]==pytest.approx(total/(math.log(2)*8),rel=1e-6)
    with pytest.raises(ValueError,match="overlong"):
        get_task("geometry_domain_bpb")({"max_tokens":3}).evaluate(lm,tok,rows)


def test_lidar_scores_the_final_content_states_and_ignores_padding(lm):
    tok=ByteTokenizer()
    rows=[{"id":str(i),"block":0,"anchor":f"The number is {i}.","positive":f"It is {i}.",
           "negative":f"Value: {i+3}."} for i in range(6)]
    task=get_task("geometry_paraphrase_lidar")({"batch_size":5})
    got=task.evaluate(lm,tok,rows)
    features=[]
    with torch.no_grad():
        for row in rows:
            pair=[]
            for k in ("anchor","positive"):
                ids=torch.tensor([[256,*row[k].encode()]])
                pair.append(lm(ids,output_hidden_states=True).hidden_states[-1][0,1:].mean(0).numpy())
            features.append(pair)
    a,b=np.asarray(features,dtype=float).transpose(1,0,2)
    assert got["pooled"]["positive_rank"]==pytest.approx(dense_rank(a,b),rel=1e-4)
    assert got["blocks"]["0"]["n"]==6

def test_public_evaluate_reads_panel_without_loading_wikitext(lm, tmp_path, monkeypatch):
    from blme import core, cache
    tok = ByteTokenizer()
    path = tmp_path / "panel.json"
    path.write_text(json.dumps({"documents": [{"id":"1","text":"word","domain":"news","block":0}]}))
    monkeypatch.setattr(core, "load_model_and_tokenizer", lambda *args, **kwargs: (lm, tok))
    def unexpected(*args, **kwargs):
        pytest.fail("An explicit structured panel must be sufficient")
    monkeypatch.setattr(cache, "load_default_corpus", unexpected)
    task = "geometry_domain_bpb"
    env = core.evaluate(model_args="pretrained=tiny", tasks=[task], device="cpu",
                        task_configs={task: {"panel_path": str(path)}})
    assert not env["errors"]
    assert env["results"][task]["documents"][0]["bytes"] == 4


def test_large_vocabulary_cpu_likelihood_uses_accurate_class_reduction():
    # A few large values and many small ones stress summation in the softmax.
    logits = torch.zeros(2, 6, 50257)
    logits[:, :, 0] = 12
    targets = torch.tensor([[0, 1, 2345, 12, -100], [0, 9, 50000, -100, -100]])
    exact = logits[:, :-1].double().log_softmax(-1)
    terms = -exact.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    expected = terms.masked_fill(targets == -100, 0).sum(1)
    torch.testing.assert_close(_document_nll(logits, targets), expected, rtol=1e-10, atol=0)
