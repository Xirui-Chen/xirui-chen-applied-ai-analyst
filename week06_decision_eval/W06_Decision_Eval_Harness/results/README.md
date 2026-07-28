# Results Directory

The full run creates:

- `rule_controller_predictions.csv`
- `rule_controller_summary.csv`
- `llm_judgments_seed_7.csv`
- `llm_judgments_seed_17.csv`
- `llm_judgments_seed_27.csv`
- `llm_judge_all_runs.csv`
- `reliability_summary.csv`
- `pairwise_kappa.csv`
- `scenario_level_results.csv`
- `domain_summary.csv`
- `error_analysis.csv`
- `run_metadata.json`
- `evaluation_summary.md`

Exact match against `target_action` is authoritative. The LLM judge is a secondary experimental scoring layer whose usefulness depends on parse rate and seeded agreement.
