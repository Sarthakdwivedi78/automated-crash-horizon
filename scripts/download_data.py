"""Download a named split after accepting dataset conditions; uses HF_TOKEN."""
import argparse
from huggingface_hub import snapshot_download
p = argparse.ArgumentParser()
p.add_argument('--split',choices=['train','validation','test'],default='validation')
p.add_argument('--output',default='data/vzcrash')
a = p.parse_args()
snapshot_download('vzc-research-chapter/VZCrash',repo_type='dataset',allow_patterns=[f'data/{a.split}-*','README.md'],local_dir=a.output)
