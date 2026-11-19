"""Populate the hosted runner's dedicated Hugging Face and NLTK caches before offline pytest."""
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
import nltk


WEIGHTS = {
    "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "distilgpt2": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    "hf-internal-testing/tiny-random-gpt2": "71034c5d8bde858ff824298bdedc65515b97d2b9",
    "EleutherAI/pythia-160m": "50f5173d932e8e61f858120bcb800b97af589f46",
    "EleutherAI/pythia-70m-deduped": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
}
TOKENIZERS = {
    "unsloth/Llama-3.2-1B": "9535bd9b1d1dea6acafbdc4813b728796aeb28da",
    "unsloth/Llama-3.2-1B-Instruct": "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "microsoft/phi-4": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
}
DATASETS = [
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3"),
    ("EleutherAI/wikitext_document_level", "wikitext-103-raw-v1", "647234772b9554e208af6c826f23b99e3cac88c8"),
    ("NeelNanda/counterfact-tracing", "default", "c945b082ca08d0a8f3ba227fb78404a09614c36e"),
    ("EleutherAI/lambada_openai", "en", "900124bf3b8235c6daf21033af9948b3f07346c4"),
]


def main():
    home = Path(os.environ["HF_HOME"])
    patterns = ["config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
                "merges.txt", "vocab.*", "*.model", "*.jinja", "added_tokens.json"]
    for repo, revision in {**WEIGHTS, **TOKENIZERS}.items():
        wanted = [*patterns, "model.safetensors"] if repo in WEIGHTS else patterns
        snapshot = Path(snapshot_download(repo, revision=revision, cache_dir=home / "hub", allow_patterns=wanted))
        if snapshot.name != revision or (repo in WEIGHTS and not (snapshot / "model.safetensors").is_file()):
            raise RuntimeError(f"Incomplete pinned fixture for {repo}")
        # Public tests use repo names without revisions. Bind the offline ref
        # in the job's private cache to the exact snapshot downloaded above.
        ref = snapshot.parent.parent / "refs/main"
        ref.parent.mkdir(exist_ok=True)
        ref.write_text(revision)
    for repo, config, revision in DATASETS:
        data = load_dataset(repo, config, revision=revision)
        if not data or any(len(split) == 0 for split in data.values()):
            raise RuntimeError(f"Empty dataset fixture for {repo}/{config}")
    nltk.download("punkt_tab", download_dir=os.environ["NLTK_DATA"], raise_on_error=True)


if __name__ == "__main__":
    main()
