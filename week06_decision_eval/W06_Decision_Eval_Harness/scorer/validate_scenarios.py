#!/usr/bin/env python3
"""Validate the 50 YAML scenarios against the public contract."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator
from common import DEFAULT_CONFIG,PROJECT_ROOT,load_config,load_scenarios
def validate_all(config_path: str|Path=DEFAULT_CONFIG)->list[dict]:
    cfg=load_config(config_path); schema=yaml.safe_load((PROJECT_ROOT/'scenarios/scenario_schema.yaml').read_text(encoding='utf-8')); v=Draft202012Validator(schema)
    records=load_scenarios(); errors=[]; ids=[]
    for rec in records:
        src=rec.pop('_source_file','<unknown>'); ids.append(rec.get('id','<missing>'))
        for e in sorted(v.iter_errors(rec),key=lambda z:list(z.path)):
            loc='.'.join(map(str,e.path)) or '<root>'; errors.append(f'{src}: {loc}: {e.message}')
        if rec.get('target_action') not in rec.get('allowed_actions',[]): errors.append(f'{src}: target_action must be included in allowed_actions')
    dup=[x for x,n in Counter(ids).items() if n>1]
    if dup: errors.append(f'Duplicate scenario IDs: {dup}')
    expected=int(cfg['project']['expected_scenario_count'])
    if len(records)!=expected: errors.append(f'Expected {expected} scenarios, found {len(records)}')
    actual=Counter(r['domain'] for r in records); target=Counter(cfg['project']['expected_domain_counts'])
    if actual!=target: errors.append(f'Domain counts differ. Expected {dict(target)}, found {dict(actual)}')
    if errors: raise ValueError('Scenario validation failed:\n- '+'\n- '.join(errors))
    return records
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(DEFAULT_CONFIG));args=ap.parse_args(); records=validate_all(args.config); counts=Counter(x['domain'] for x in records)
    print(f'PASS: {len(records)} scenarios validated.')
    for d,n in sorted(counts.items()): print(f'  {d}: {n}')
    return 0
if __name__=='__main__': raise SystemExit(main())
