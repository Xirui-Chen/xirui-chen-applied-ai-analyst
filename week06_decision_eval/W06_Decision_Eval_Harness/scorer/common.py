"""Shared loading, path, and serialization helpers."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import yaml
PROJECT_ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=PROJECT_ROOT/'config.yaml'
def load_config(path: str|Path=DEFAULT_CONFIG)->dict[str,Any]:
    p=Path(path); p=p if p.is_absolute() else (PROJECT_ROOT/p).resolve()
    return yaml.safe_load(p.read_text(encoding='utf-8'))
def load_scenarios(scenario_dir: str|Path|None=None)->list[dict[str,Any]]:
    root=Path(scenario_dir) if scenario_dir else PROJECT_ROOT/'scenarios'
    excluded={'scenario_schema.yaml','action_taxonomy.yaml','index.yaml'}
    out=[]
    for p in sorted(root.glob('*.yaml')):
        if p.name in excluded: continue
        x=yaml.safe_load(p.read_text(encoding='utf-8')); x['_source_file']=str(p.relative_to(PROJECT_ROOT)); out.append(x)
    return out
def write_json(path:Path,payload:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False,default=str),encoding='utf-8')
def ensure_results_dir()->Path:
    p=PROJECT_ROOT/'results';p.mkdir(parents=True,exist_ok=True);return p
