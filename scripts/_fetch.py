import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]; out = sys.argv[2]
p = snapshot_download(repo, cache_dir=out,
    allow_patterns=["*.json","*.txt","*.safetensors","tokenizer*","preprocessor*"])
print("OK", p)
