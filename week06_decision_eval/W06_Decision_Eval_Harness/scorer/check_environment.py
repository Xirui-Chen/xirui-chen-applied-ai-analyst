#!/usr/bin/env python3
"""Check dependencies, contracts, and optional Hugging Face model availability."""
from __future__ import annotations
import argparse,importlib.metadata,platform,sys
from common import DEFAULT_CONFIG,load_config
from validate_scenarios import validate_all
def ver(x):
    try:return importlib.metadata.version(x)
    except importlib.metadata.PackageNotFoundError:return 'not-installed'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(DEFAULT_CONFIG));ap.add_argument('--check-model',action='store_true');ap.add_argument('--local-files-only',action='store_true');a=ap.parse_args();print('Python:',sys.version.split()[0]);print('Platform:',platform.platform());print('Machine:',platform.machine())
    for p in ['PyYAML','jsonschema','numpy','pandas','scikit-learn','torch','transformers','sentencepiece','tqdm']:print(f'{p}: {ver(p)}')
    print(f'PASS: {len(validate_all(a.config))} scenarios validated.')
    if a.check_model:
        from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
        name=load_config(a.config)['judge']['model_name'];print('Checking model:',name);AutoTokenizer.from_pretrained(name,local_files_only=a.local_files_only);AutoModelForSeq2SeqLM.from_pretrained(name,local_files_only=a.local_files_only);print('PASS: model available.')
    return 0
if __name__=='__main__':raise SystemExit(main())
