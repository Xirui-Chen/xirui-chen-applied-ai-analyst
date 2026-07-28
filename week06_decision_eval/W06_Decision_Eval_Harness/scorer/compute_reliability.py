#!/usr/bin/env python3
"""Nominal Krippendorff alpha and pairwise Cohen kappa."""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import combinations
import numpy as np,pandas as pd
from sklearn.metrics import cohen_kappa_score
from common import PROJECT_ROOT
def alpha_nominal(matrix:pd.DataFrame)->float:
    disagree=pairs=0;allvals=[]
    for _,row in matrix.iterrows():
        vals=[str(v) for v in row.tolist() if pd.notna(v)];allvals+=vals
        for a,b in combinations(vals,2):pairs+=1;disagree+=int(a!=b)
    if pairs==0:return float('nan')
    do=disagree/pairs;c=Counter(allvals);n=sum(c.values())
    if n<2:return float('nan')
    de=(n*n-sum(x*x for x in c.values()))/(n*(n-1))
    return float('nan') if de==0 else float(1-do/de)
def interpretation(a:float)->str:
    if not np.isfinite(a):return 'undefined: insufficient valid variation'
    if a>=.80:return 'strong agreement for this test bank'
    if a>=.667:return 'tentative agreement; useful with review'
    if a>=.40:return 'limited agreement; unsuitable as a sole scorer'
    return 'poor agreement; unreliable for automated use'
def compute(frame:pd.DataFrame):
    valid=frame[frame.parse_ok.astype(bool)].copy();p=valid.pivot(index='scenario_id',columns='judge_seed',values='verdict');complete=p.dropna();a=alpha_nominal(p)
    rows=[{'scope':'overall','domain':'all','n_scenarios':frame.scenario_id.nunique(),'n_valid_judgments':len(valid),'parse_rate':frame.parse_ok.astype(bool).mean(),'complete_scenario_count':len(complete),'exact_three_run_agreement':(complete.nunique(axis=1)==1).mean() if len(complete) else np.nan,'krippendorff_alpha_nominal':a,'interpretation':interpretation(a)}]
    for d,g in frame.groupby('domain'):
        gv=g[g.parse_ok.astype(bool)];q=gv.pivot(index='scenario_id',columns='judge_seed',values='verdict');qc=q.dropna();aa=alpha_nominal(q);rows.append({'scope':'domain','domain':d,'n_scenarios':g.scenario_id.nunique(),'n_valid_judgments':len(gv),'parse_rate':g.parse_ok.astype(bool).mean(),'complete_scenario_count':len(qc),'exact_three_run_agreement':(qc.nunique(axis=1)==1).mean() if len(qc) else np.nan,'krippendorff_alpha_nominal':aa,'interpretation':interpretation(aa)})
    pairs=[]
    for a1,a2 in combinations(sorted(p.columns),2):
        z=p[[a1,a2]].dropna();pairs.append({'judge_seed_a':a1,'judge_seed_b':a2,'n_common_scenarios':len(z),'cohen_kappa':cohen_kappa_score(z[a1],z[a2]) if len(z) else np.nan})
    return pd.DataFrame(rows),pd.DataFrame(pairs)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default=str(PROJECT_ROOT/'results/llm_judge_all_runs.csv'));a=ap.parse_args();f=pd.read_csv(a.input);s,p=compute(f);s.to_csv(PROJECT_ROOT/'results/reliability_summary.csv',index=False);p.to_csv(PROJECT_ROOT/'results/pairwise_kappa.csv',index=False);print(s.to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
