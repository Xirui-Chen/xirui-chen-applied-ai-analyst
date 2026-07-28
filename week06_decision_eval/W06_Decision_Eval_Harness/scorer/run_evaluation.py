#!/usr/bin/env python3
"""Run validation, rule baseline, three seeded LLM judge runs, and reliability."""
from __future__ import annotations
import argparse,platform,sys
from pathlib import Path
import pandas as pd,torch,transformers
from tqdm import tqdm
from common import DEFAULT_CONFIG,ensure_results_dir,load_config,write_json
from compute_reliability import compute
from llm_judge import SmallHFJudge
from rule_based_controller import run_rule_controller
from summarize_results import summarize
from validate_scenarios import validate_all
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(DEFAULT_CONFIG));ap.add_argument('--model');ap.add_argument('--judge-seeds',nargs='*',type=int);ap.add_argument('--device');ap.add_argument('--max-scenarios',type=int);ap.add_argument('--skip-llm',action='store_true');ap.add_argument('--local-files-only',action='store_true');ap.add_argument('--overwrite',action='store_true');a=ap.parse_args();cfg=load_config(a.config);res=ensure_results_dir();validate_all(a.config)
    rule=run_rule_controller(a.config,a.max_scenarios);rp=res/'rule_controller_predictions.csv';rule.to_csv(rp,index=False);rs=rule.groupby('domain',as_index=False).agg(scenario_count=('scenario_id','count'),exact_match_accuracy=('exact_match','mean'),fallback_rate=('fallback_used','mean'));rs.to_csv(res/'rule_controller_summary.csv',index=False);print('\nRule baseline:\n',rs.to_string(index=False))
    if a.skip_llm:
        sc,dom,err=summarize(rp,None);sc.to_csv(res/'scenario_level_results.csv',index=False);dom.to_csv(res/'domain_summary.csv',index=False);err.to_csv(res/'error_analysis.csv',index=False);print('\nSkipped LLM judge by request.');return 0
    model=a.model or cfg['judge']['model_name'];seeds=a.judge_seeds or list(cfg['judge']['seeds']);device=a.device or cfg['judge']['device'];scores=cfg['scoring']['verdict_scores'];j=SmallHFJudge(model,device,a.local_files_only,int(cfg['judge']['max_input_tokens']),int(cfg['judge']['max_new_tokens']),bool(cfg['judge']['do_sample']),float(cfg['judge']['temperature']),float(cfg['judge']['top_p']));print(f'Loaded judge model: {model} on {j.device}')
    allrows=[];records=rule.to_dict('records')
    for seed in seeds:
        op=res/f'llm_judgments_seed_{seed}.csv'
        if op.exists() and not a.overwrite:
            old=pd.read_csv(op)
            if len(old)==len(rule):print('SKIP completed judge seed',seed);allrows+=old.to_dict('records');continue
        rows=[]
        for row in tqdm(records,desc=f'Judge seed {seed}'):
            z=j.judge(row,seed,scores);rows.append({'scenario_id':row['scenario_id'],'domain':row['domain'],'task_type':row['task_type'],'difficulty':row['difficulty'],'risk_level':row['risk_level'],'judge_seed':seed,'scenario_seed':z.scenario_seed,'model_name':model,'candidate_action':row['candidate_action'],'target_action':row['target_action'],'exact_match':row['exact_match'],'verdict':z.verdict,'score':z.score,'reason':z.reason,'parse_ok':z.parse_ok,'used_retry':z.used_retry,'latency_sec':z.latency_sec,'raw_text':z.raw_text})
        pd.DataFrame(rows).to_csv(op,index=False);allrows+=rows
    allf=pd.DataFrame(allrows).sort_values(['scenario_id','judge_seed']);apath=res/'llm_judge_all_runs.csv';allf.to_csv(apath,index=False);rel,pair=compute(allf);rel.to_csv(res/'reliability_summary.csv',index=False);pair.to_csv(res/'pairwise_kappa.csv',index=False);sc,dom,err=summarize(rp,apath);sc.to_csv(res/'scenario_level_results.csv',index=False);dom.to_csv(res/'domain_summary.csv',index=False);err.to_csv(res/'error_analysis.csv',index=False);write_json(res/'run_metadata.json',{'model_name':model,'judge_seeds':seeds,'device':j.device,'scenario_count':len(rule),'python':sys.version.split()[0],'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,'transformers':transformers.__version__,'authoritative_score':'exact match against target_action','secondary_score':'seeded small-model LLM rubric'});print('\nReliability:\n',rel.to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
