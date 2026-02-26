import pytest
import torch
import numpy as np
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast,
    BertConfig, BertLMHeadModel, BertTokenizerFast
)

@pytest.fixture(params=["gpt2", "llama", "bert"])
def mock_model_tokenizer(request):
    """
    Returns a tuple of (model, tokenizer) parameterized over different architectures.
    These are real Hugging Face models instantiated with tiny configurations so they load instantly
    and require no downloads, while preserving the exact module hierarchy (e.g., .transformer.h vs .model.layers)
    """
    arch = request.param
    device = "cpu"
    
    if arch == "gpt2":
        config = GPT2Config(
            vocab_size=1000,
            n_positions=128,
            n_embd=32,
            n_layer=2,
            n_head=2,
        )
        model = GPT2LMHeadModel(config)
        # Trivial mock tokenizer
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.BPE()))
        tokenizer.pad_token = "[PAD]"
        tokenizer.eos_token = "[EOS]"
        
    elif arch == "llama":
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        model = LlamaForCausalLM(config)
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.BPE()))
        tokenizer.pad_token = "[PAD]"
        tokenizer.eos_token = "[EOS]"
        
    elif arch == "bert":
        # Note: BERT is technically an encoder, but we use BertLMHeadModel for causal-like generation tests in some frameworks,
        # or just as a representative of the .encoder.layer structure.
        config = BertConfig(
            vocab_size=1000,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=128,
            is_decoder=True # needed for lm head
        )
        model = BertLMHeadModel(config)
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.BPE()))
        tokenizer.pad_token = "[PAD]"
        tokenizer.eos_token = "[EOS]"

    model.eval()
    
    # We patch the tokenizer output so it actually produces valid tensor shapes when called
    class DummyTokenizer:
        def __init__(self, base_tokenizer, vocab_size):
            self.base = base_tokenizer
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.eos_token_id = 1
            
        def __call__(self, text, return_tensors="pt", **kwargs):
            # Return realistic tensor shapes
            # If text is a list, process as batch
            is_list = isinstance(text, (list, tuple))
            batch_size = len(text) if is_list else 1
            length = 8
            
            ids = torch.randint(0, self.vocab_size, (batch_size, length))
            mask = torch.ones_like(ids)
            
            if return_tensors == "pt":
                class BatchDict(dict):
                    def to(self, dev): return self
                return BatchDict({"input_ids": ids, "attention_mask": mask})
            return {"input_ids": ids.tolist(), "attention_mask": mask.tolist()}
            
        def encode(self, text, return_tensors=None, **kwargs):
            ids = torch.randint(0, self.vocab_size, (1, 8))
            if return_tensors == "pt":
                return ids
            return ids[0].tolist()
            
        def decode(self, *args, **kwargs):
            return "dummy text"
            
    return model, DummyTokenizer(tokenizer, config.vocab_size)

@pytest.fixture
def mock_model(mock_model_tokenizer):
    return mock_model_tokenizer[0]

@pytest.fixture
def mock_tokenizer(mock_model_tokenizer):
    return mock_model_tokenizer[1]
