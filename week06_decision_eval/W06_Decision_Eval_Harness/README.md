# W06 Decision Evaluation Harness

## Purpose

This folder implements a synthetic, publication-safe decision-evaluation harness for four product contexts.

| Domain | Decision type | Scenarios |
|---|---|---:|
| Aido Rover | Patrol-route priority arbitration under energy and safety constraints | 15 |
| Sentinel Prime AI | Alert mode switching and escalation under bandwidth limits | 15 |
| Senpai | Tutoring-session conversational branching | 10 |
| Fari | Companion and care-support conversational branching | 10 |
| **Total** |  | **50** |

Every YAML scenario specifies inputs, allowed actions, an explicit correct action label, and a rationale. A transparent rule-based controller produces candidate actions. Exact match is the authoritative score. A small open Hugging Face model then assigns `PASS`, `PARTIAL`, or `FAIL` in three seeded judge runs.

## Scope and confidentiality

This is an analyst-built evaluation contract, not a production robot controller, clinical system, safeguarding system, or security dispatch system.

The public InGen website describes Origami AI as a shared physical-intelligence layer, but no public implementation-level AMDC decision schema was found. The phrase **AMDC-style decision evaluation** is therefore used here as an analyst abstraction: structured state enters a constrained action space, a target behavior defines the contract, and the decision is auditable.

All names, thresholds, health readings, student situations, energy values, bandwidth values, and incidents are synthetic and generalized. Confidential reference documents are not redistributed.

## Structure

```text
W06_Decision_Eval_Harness/
├── README.md
├── requirements_w06.txt
├── requirements_w06_lock.txt
├── artifact_manifest.json
├── config.yaml
├── scenarios/
│   ├── README.md
│   ├── scenario_schema.yaml
│   ├── action_taxonomy.yaml
│   ├── index.yaml
│   └── 50 scenario YAML files
├── scorer/
│   ├── common.py
│   ├── validate_scenarios.py
│   ├── rule_based_controller.py
│   ├── llm_judge.py
│   ├── compute_reliability.py
│   ├── summarize_results.py
│   ├── run_evaluation.py
│   ├── audit_judge_results.py
│   └── check_environment.py
└── results/
```

## Scoring contract

### Authoritative score

```text
candidate_action == target_action
```

### LLM rubric

- `PASS`: candidate matches the target or is clearly equivalent.
- `PARTIAL`: candidate differs but addresses the main constraint without a safety downgrade.
- `FAIL`: candidate misses the main constraint or creates unsafe escalation or de-escalation.

The configured judge is `google/flan-t5-base`. The final protocol uses a strict single-label classification prompt: `A = PASS`, `B = PARTIAL`, and `C = FAIL`. Free-form and multi-label outputs are rejected, and one deterministic retry is allowed after an invalid sampled response.

### Seeded reliability

Default seeds are `7`, `17`, and `27`. The harness reports nominal Krippendorff's alpha, exact three-run agreement, pairwise Cohen's kappa, and parse rate overall and by domain.

Descriptive interpretation:

- alpha >= 0.80: strong agreement for this bank
- 0.667 to 0.80: tentative, usable with review
- 0.40 to 0.667: limited, not suitable as a sole scorer
- below 0.40: poor reliability

High agreement does not prove correctness. A judge can be consistently wrong.

## 1. Repository location

```text
xirui-chen-applied-ai-analyst/
└── week06_decision_eval/
    ├── W06_Decision_Eval_Harness/
    └── W06_Eval_Methodology.md
```

## 2. Create a Python 3.11 environment

```bash
cd ~/Downloads/xirui-chen-applied-ai-analyst/week06_decision_eval/W06_Decision_Eval_Harness
python3.11 -m venv .venv_w06
source .venv_w06/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_w06.txt
```

## 3. Validate all scenarios

```bash
python scorer/validate_scenarios.py
```

Expected counts: 15 Rover, 15 Sentinel, 10 Senpai, and 10 Fari.

## 4. Rule-only smoke test

```bash
python scorer/run_evaluation.py --skip-llm --max-scenarios 8
```

Then run all 50 deterministic rule predictions:

```bash
python scorer/run_evaluation.py --skip-llm
```

## 5. Download and check the model

```bash
python scorer/check_environment.py --check-model
```

After the model is cached, verify offline availability with:

```bash
python scorer/check_environment.py --check-model --local-files-only
```

## 6. Full evaluation

```bash
python scorer/run_evaluation.py
```

Equivalent explicit command:

```bash
python scorer/run_evaluation.py \
  --model google/flan-t5-base \
  --judge-seeds 7 17 27 \
  --device auto
```

For a four-scenario end-to-end model smoke test:

```bash
python scorer/run_evaluation.py \
  --max-scenarios 4 \
  --judge-seeds 7 17 27 \
  --overwrite
```

Run the full command with `--overwrite` after a smoke test because the filenames are shared.

## 7. Recompute summaries without rerunning the model

```bash
python scorer/compute_reliability.py
python scorer/summarize_results.py
python scorer/audit_judge_results.py \
  --expected-scenarios 50 \
  --expected-seeds 7 17 27
```

## Expected outputs

```text
results/
├── rule_controller_predictions.csv
├── rule_controller_summary.csv
├── llm_judgments_seed_7.csv
├── llm_judgments_seed_17.csv
├── llm_judgments_seed_27.csv
├── llm_judge_all_runs.csv
├── reliability_summary.csv
├── pairwise_kappa.csv
├── scenario_level_results.csv
├── domain_summary.csv
├── error_analysis.csv
├── run_metadata.json
└── evaluation_summary.md
```

## Final measured run

The committed results were generated with `google/flan-t5-base` on CPU using
judge seeds 7, 17, and 27.

| Metric | Measured result |
|---|---:|
| Scenarios | 50 |
| Seeded judgments | 150 |
| Valid parse rate | 100% |
| Exact three-run agreement | 78.0% |
| Overall nominal Krippendorff's alpha | 0.725 |
| Aido Rover alpha | 0.905 |
| Sentinel Prime alpha | 0.748 |
| Fari alpha | 0.627 |
| Senpai alpha | 0.366 |

The overall result indicates tentative seeded agreement and supports using the
judge as an exploratory secondary scorer with human review. Reliability varied
substantially by domain.

Criterion alignment remained limited. The judge assigned PASS to 77 of 129
exact action matches, or 59.7%, and assigned PASS to none of the 21 non-exact
judgments. This indicates conservative behavior and substantial false-negative
risk. Exact match against `target_action` therefore remains the authoritative
score.

## Known limitations

- The scenario bank and action taxonomy are synthetic analyst contracts.
- Target actions are not validated field policies.
- The small model may be sensitive to prompt phrasing, sampling, and software versions.
- The judge sees the target action, so it evaluates rubric consistency rather than independently solving the task.
- Three seeds are a basic reliability check, not a complete robustness study.
- Krippendorff's alpha can be undefined when judgments lack category variation.
- Exact action labels simplify nuanced real-world decisions.
- No real patient, student, incident, customer, or robot-operation data is used.
- The outputs must not be used for clinical, safeguarding, security-dispatch, or autonomous-robot decisions.

## Public references

- InGen Dynamics: `https://ingendynamics.com/`
- Sentinel Prime AI: `https://ingendynamics.com/sentinel.html`
- FLAN-T5-base: `https://huggingface.co/google/flan-t5-base`
- Hugging Face T5 documentation: `https://huggingface.co/docs/transformers/model_doc/t5`

## AI assistance disclosure

AI assistance was used to draft scenario wording, code scaffolding, validation logic, and documentation. The intern remains responsible for reviewing the contracts, running the model, interpreting reliability, and revising unsupported claims.
