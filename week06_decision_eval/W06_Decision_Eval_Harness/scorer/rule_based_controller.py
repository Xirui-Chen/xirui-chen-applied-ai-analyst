#!/usr/bin/env python3
"""Transparent baseline rules for all four synthetic decision domains."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import pandas as pd
from common import DEFAULT_CONFIG,PROJECT_ROOT,load_config
from validate_scenarios import validate_all

def choose_allowed(proposed:str,allowed:list[str],fallback:list[str])->tuple[str,bool]:
    if proposed in allowed:return proposed,False
    for x in fallback:
        if x in allowed:return x,True
    return allowed[0],True

def rover(i:dict[str,Any],cfg:dict)->tuple[str,str]:
    reserve=float(cfg['rule_controller']['return_reserve_pct']); b=float(i['battery_soc_pct']); eb=float(i['energy_to_base_pct'])
    if i.get('actuator_fault'):return 'safe_stop','Actuator fault takes precedence over route completion.'
    if float(i.get('localization_confidence',1))<float(cfg['rule_controller']['low_localization_threshold']):
        return ('hold_and_request_operator','Low localization with a live operator link.') if i.get('comms_status')=='good' else ('reroute_to_safe_checkpoint','Low localization without remote guidance.')
    if b<=eb+reserve:return 'return_to_base','Battery is at or below the configured return reserve.'
    if i.get('weather_severity')=='severe' and b<=eb+reserve+8:return 'return_to_base','Severe weather narrows the usable energy margin.'
    if i.get('route_blocked'):return 'reroute_to_safe_checkpoint','The current route is blocked.'
    if i.get('immediate_human_safety_event'):
        need=float(i.get('energy_to_priority_task_pct',100))
        if b>eb+need+reserve:return 'switch_to_priority_route','The immediate safety event is energy-feasible.'
        return 'return_to_base','The baseline cannot safely service the event within its energy rule.'
    p={'none':0,'low':1,'normal':2,'medium':3,'high':4,'critical':5}
    if p.get(str(i.get('new_task_priority')),0)>p.get(str(i.get('current_route_priority')),0) and b>eb+float(i.get('energy_to_priority_task_pct',0))+reserve:
        return 'switch_to_priority_route','A higher-priority task is energy-feasible.'
    return 'continue_current_route','No higher-priority safety or feasibility rule fired.'

def sentinel(i:dict[str,Any],cfg:dict)->tuple[str,str]:
    sev=str(i['alert_severity']); c=float(i['confidence']); n=int(i['corroborating_sensor_count']); bw=int(i['bandwidth_kbps']); net=str(i['network_status'])
    if i.get('immediate_danger') or sev=='critical':return 'trigger_local_emergency','Critical or immediate danger is handled locally.'
    if c<float(cfg['rule_controller']['sentinel_low_confidence']) and n<2:return 'suppress_and_log','Low confidence and weak corroboration.'
    if c<float(cfg['rule_controller']['sentinel_medium_confidence']):return 'request_sensor_recheck','Intermediate confidence requires another observation.'
    if i.get('privacy_zone'):return 'suppress_and_log','Configured privacy-zone restriction.'
    if net=='offline':return 'cache_evidence_and_monitor','No external link; retain evidence locally.'
    if bw<int(cfg['rule_controller']['very_low_bandwidth_kbps']):return 'send_metadata_only','Only essential metadata fits the link.'
    if sev=='high' and i.get('operator_channel_available',False):return 'escalate_to_operator','High severity with an available operator channel.'
    if bw<int(cfg['rule_controller']['compressed_evidence_bandwidth_kbps']):return 'send_compressed_evidence','A compact evidence package fits the degraded link.'
    return 'continue_local_monitoring','No escalation or constrained-link rule fired.'

def senpai(i:dict[str,Any])->tuple[str,str]:
    if i.get('safeguarding_flag'):return 'pause_and_safeguard','Safeguarding supersedes the teaching objective.'
    if i.get('teacher_override'):return 'request_teacher_support','Authorized teacher override.'
    if float(i.get('intent_ambiguity',0))>=.75:return 'ask_clarifying_question','The learner request is underspecified.'
    if float(i.get('frustration_score',0))>=.80 or float(i.get('fatigue_score',0))>=.80:return 'offer_short_break','High frustration or fatigue.'
    if i.get('send_profile')!='none' and i.get('accessibility_barrier'):return 'activate_send_accommodation','An approved SEND support is relevant.'
    if float(i.get('mastery_score',0))>=.85 and int(i.get('consecutive_correct',0))>=3:return 'increase_difficulty','Mastery and repeated success.'
    if int(i.get('consecutive_errors',0))>=3:return 'provide_step_by_step_support','Repeated errors require structured scaffolding.'
    if i.get('modality_mismatch'):return 'switch_learning_modality','The current presentation mode is not working.'
    if int(i.get('consecutive_errors',0))>=1:return 'provide_light_hint','A small hint is the least intrusive support.'
    return 'continue_current_activity','No adaptation trigger fired.'

def fari(i:dict[str,Any])->tuple[str,str]:
    if i.get('fall_detected') and not i.get('resident_responsive'):return 'initiate_emergency_escalation','Confirmed fall with no response.'
    if i.get('self_harm_language') or i.get('vital_severity')=='critical':return 'initiate_emergency_escalation','Immediate safety or critical vital signal.'
    if i.get('fall_detected'):return 'notify_caregiver','Responsive fall still needs caregiver follow-up.'
    if i.get('privacy_request'):return 'pause_sensitive_data_collection','Resident has withdrawn consent.'
    if i.get('suspected_duplicate_dose'):return 'request_clinical_review','Possible duplicate dose requires authorized review.'
    if i.get('medication_due') and not i.get('identity_verified'):return 'request_identity_verification','Medication interaction requires identity verification.'
    if int(i.get('medication_refusal_count',0))>=2:return 'notify_caregiver','Baseline escalates repeated refusal to the caregiver.'
    if i.get('medication_due'):return 'provide_medication_reminder','Routine verified medication reminder.'
    if i.get('vital_severity')=='moderate':return 'request_clinical_review','Baseline escalates a moderate vital anomaly.'
    if i.get('confusion_level') in {'moderate','high'}:return 'offer_reorientation','Known disorientation without immediate danger.'
    return 'continue_supportive_conversation','No clinical or safety escalation trigger.'

def predict_one(rec:dict[str,Any],cfg:dict)->dict[str,Any]:
    d=rec['domain'];i=rec['inputs']
    if d=='aido_rover': proposed,why=rover(i,cfg); fb=['hold_and_request_operator','reroute_to_safe_checkpoint','return_to_base','continue_current_route','safe_stop']
    elif d=='sentinel_prime': proposed,why=sentinel(i,cfg); fb=['request_sensor_recheck','suppress_and_log','continue_local_monitoring','send_metadata_only','send_compressed_evidence','cache_evidence_and_monitor','escalate_to_operator','trigger_local_emergency']
    elif d=='senpai': proposed,why=senpai(i); fb=['ask_clarifying_question','provide_light_hint','provide_step_by_step_support','switch_learning_modality','activate_send_accommodation','offer_short_break','request_teacher_support','pause_and_safeguard','continue_current_activity','increase_difficulty','simplify_content']
    elif d=='fari': proposed,why=fari(i); fb=['continue_supportive_conversation','offer_reorientation','provide_medication_reminder','request_identity_verification','suggest_rest_and_recheck','respect_refusal_and_monitor','notify_caregiver','request_clinical_review','initiate_emergency_escalation','pause_sensitive_data_collection']
    else:raise ValueError(d)
    action,fallback=choose_allowed(proposed,rec['allowed_actions'],fb)
    if fallback:why+=' The proposed action was unavailable, so the first permitted fallback was used.'
    return {'scenario_id':rec['id'],'domain':d,'task_type':rec['task_type'],'difficulty':rec['metadata']['difficulty'],'risk_level':rec['metadata']['risk_level'],'candidate_action':action,'candidate_rationale':why,'target_action':rec['target_action'],'target_rationale':rec['target_rationale'],'exact_match':action==rec['target_action'],'fallback_used':fallback,'allowed_actions':'|'.join(rec['allowed_actions'])}

def run_rule_controller(config_path: str|Path=DEFAULT_CONFIG,max_scenarios:int|None=None)->pd.DataFrame:
    cfg=load_config(config_path); records=validate_all(config_path);records=records[:max_scenarios] if max_scenarios else records
    return pd.DataFrame(predict_one(x,cfg) for x in records)
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(DEFAULT_CONFIG));ap.add_argument('--max-scenarios',type=int);ap.add_argument('--output',default=str(PROJECT_ROOT/'results/rule_controller_predictions.csv'));a=ap.parse_args()
    f=run_rule_controller(a.config,a.max_scenarios);Path(a.output).parent.mkdir(parents=True,exist_ok=True);f.to_csv(a.output,index=False)
    print(f'Wrote {len(f)} predictions to {a.output}');print(f'Exact-match accuracy: {f.exact_match.mean():.3f}');print(f.groupby('domain').exact_match.agg(['count','mean']).to_string());return 0
if __name__=='__main__':raise SystemExit(main())
