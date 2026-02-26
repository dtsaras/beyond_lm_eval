import sys
import os

# Ensure we can import blme without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from blme import evaluate
from blme.registry import list_tasks

print("Available tasks:", list_tasks())

# Mock evaluation
print("\nRunning evaluation...")
# using mocked model loading for speed/test if possible, but load_model_and_tokenizer will try to load real HF model
# We can just check if registry works for now
evaluate(model_args="pretrained=gpt2", tasks=["geometry_svd"], limit=0.1)
