#!/usr/bin/env python3
"""Build scenario, domain, and error-analysis outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from common import PROJECT_ROOT
def mode_or_missing(s):
    v=s.dropna();return np.nan if v.empty else v.mode().iloc[0]
def summarize(rule_path,llm_path=None):
    rule=pd.read_csv(rule_path)
    if llm_path and Path(llm_path).exists():
        llm=pd.read_csv(llm_path);g=llm.groupby('scenario_id',as_index=False).agg(judge_majority_verdict=('verdict',mode_or_missing),judge_mean_score=('score','mean'),judge_valid_runs=('parse_ok','sum'),judge_verdict_count=('verdict','count'));sc=rule.merge(g,on='scenario_id',how='left')
    else:
        sc=rule.copy();sc['judge_majority_verdict']=np.nan;sc['judge_mean_score']=np.nan;sc['judge_valid_runs']=0;sc['judge_verdict_count']=0
    dom=sc.groupby('domain',as_index=False).agg(scenario_count=('scenario_id','count'),rule_exact_accuracy=('exact_match','mean'),fallback_rate=('fallback_used','mean'),judge_mean_score=('judge_mean_score','mean'))
    err=sc[~sc.exact_match.astype(bool)].copy();return sc,dom,err
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rule-input',default=str(PROJECT_ROOT/'results/rule_controller_predictions.csv'));ap.add_argument('--llm-input',default=str(PROJECT_ROOT/'results/llm_judge_all_runs.csv'));a=ap.parse_args();llm=Path(a.llm_input);sc,dom,err=summarize(a.rule_input,llm if llm.exists() else None);res=PROJECT_ROOT/'results';sc.to_csv(res/'scenario_level_results.csv',index=False);dom.to_csv(res/'domain_summary.csv',index=False);err.to_csv(res/'error_analysis.csv',index=False)
    md=['# W06 Evaluation Summary','','## Domain-level evaluation summary','','`judge_mean_score` is the mean ordinal rubric score (FAIL=0, PARTIAL=1, PASS=2), not accuracy.','',dom.to_markdown(index=False),'']
    rp=res/'reliability_summary.csv'
    if rp.exists():md+=['## LLM judge reliability','',pd.read_csv(rp).to_markdown(index=False),'']
    else:md+=['## LLM judge reliability','','LLM judge outputs have not been generated yet.','']
    md+=['## Interpretation boundary','','Exact match against `target_action` is authoritative. The small-model LLM judge is an experimental secondary layer. Agreement does not prove correctness, and low agreement limits automated use.','']
    (res/'evaluation_summary.md').write_text('\n'.join(md),encoding='utf-8');print(dom.to_string(index=False));print('Rule-controller mismatches:',len(err));return 0
if __name__=='__main__':raise SystemExit(main())
