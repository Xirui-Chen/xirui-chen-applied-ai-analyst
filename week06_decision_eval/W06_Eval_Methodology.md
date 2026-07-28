# W06 Eval Methodology

**Project:** Week 6 Decision Evaluation Harness  
**Deliverable:** W06_Eval_Methodology.md  
**Author:** Xirui (Crissy) Chen, Applied AI Analyst Intern  
**Scope:** 50-scenario decision test bank, rule baseline, seeded LLM judge, and reliability analysis  

## 1. Purpose and Evaluation Framing

The Week 6 harness was designed as a structured decision-evaluation system for four product contexts: Aido Rover, Sentinel Prime AI, Senpai, and Fari. The goal was not to build a production controller or to claim that a small language model can replace human review. The goal was more practical: create a reproducible test bank where future engineers can add decision scenarios, run a transparent baseline controller, and assess whether a lightweight open-model judge is stable enough to be useful as a secondary scoring layer.

The harness covers three decision categories. Aido Rover scenarios test priority arbitration, especially patrol-route selection under battery and energy constraints. Sentinel Prime AI scenarios test alert mode switching and evidence escalation when bandwidth, uncertainty, and local safety constraints interact. Senpai and Fari scenarios test conversational decision branching, where the right action depends on tutoring context, safeguarding, care support, medication safety, privacy, or human handoff.

The final scenario bank contains 50 YAML scenarios:

| Domain | Scenario count | Decision category |
|---|---:|---|
| Aido Rover | 15 | Energy-constrained patrol arbitration |
| Sentinel Prime AI | 15 | Alert escalation and bandwidth-aware mode switching |
| Senpai | 10 | Tutoring-session branching |
| Fari | 10 | Companion-care branching |
| **Total** | **50** |  |

Each scenario is synthetic and publication-safe. The cases are informed by product-level themes, but they do not reproduce confidential thresholds, internal product rules, customer data, real patient or student information, or proprietary control policies. In that sense, the harness should be read as an analyst-defined evaluation contract, not an implementation of InGen's internal decision logic.

## 2. Harness Design

The repository is organized around a clear separation between scenario contracts, controller logic, judge logic, and generated results:

```text
W06_Decision_Eval_Harness/
├── README.md
├── config.yaml
├── requirements_w06.txt
├── requirements_w06_lock.txt
├── artifact_manifest.json
├── scenarios/
├── scorer/
└── results/
```

The `scenarios/` folder contains one YAML file per scenario, plus an action taxonomy, a schema file, and an index file. The `scorer/` folder contains loading, validation, rule-controller, LLM-judge, reliability, and summarization scripts. The `results/` folder contains the deterministic rule-controller outputs, seeded LLM judgments, reliability tables, pairwise kappa results, domain summaries, and error analysis.

The core design principle is that scenario correctness is explicit and auditable. Every scenario includes a finite action set and one target action. This avoids using the LLM to define the answer and then judge the answer. Instead, the LLM is only a secondary rubric layer, while the authoritative target remains deterministic and human-readable.

## 3. Scenario Schema and Extension Contract

Each YAML scenario follows the same contract:

```yaml
schema_version: "1.0"
id: rover_001
domain: aido_rover
task_type: priority_arbitration
title: ...
description: ...
inputs:
  ...
allowed_actions:
  - ...
target_action: ...
target_rationale: ...
metadata:
  difficulty: basic | boundary | safety_critical
  risk_level: low | medium | high | critical
  tags:
    - ...
  public_safe: true
  source_basis: ...
```

The key fields are:

| Field | Purpose |
|---|---|
| `id` | Unique scenario identifier |
| `domain` | Product context: Aido Rover, Sentinel Prime AI, Senpai, or Fari |
| `task_type` | Decision class: priority arbitration, mode switching, or conversational branching |
| `inputs` | Structured state observed by the decision system |
| `allowed_actions` | Valid actions for this scenario |
| `target_action` | Correct action label under the scenario contract |
| `target_rationale` | Human-readable reason for the target |
| `metadata` | Difficulty, risk level, tags, and public-safety note |

A future engineer can extend the bank without reading the implementation code by following four steps. First, copy an existing scenario in the same domain. Second, assign a new unique ID and use only action labels from `action_taxonomy.yaml`. Third, ensure `target_action` appears in `allowed_actions`. Fourth, run `python scorer/validate_scenarios.py`. The validator checks schema structure, uniqueness, domain counts, and target-action validity.

This design makes the harness easy to extend, but it also imposes a limitation. Nuanced real-world decisions are compressed into one label. That is useful for repeatable evaluation, but it is not a substitute for human judgment in safety, care, education, or security contexts.

## 4. Rule-Based Controller

The rule-based controller is the deterministic baseline. It is intentionally simple, transparent, and incomplete. It uses structured scenario inputs to choose a candidate action, records a rationale, and then compares the candidate action with the scenario's `target_action`.

The authoritative score is:

```text
candidate_action == target_action
```

This exact-match score is not the same as the LLM judge score. It is the primary correctness measure for the baseline controller.

The final rule baseline produced 50 predictions with an overall exact-match accuracy of 86 percent:

| Domain | Scenario count | Rule exact-match accuracy | Fallback rate |
|---|---:|---:|---:|
| Aido Rover | 15 | 80.0% | 0.0% |
| Fari | 10 | 80.0% | 0.0% |
| Senpai | 10 | 100.0% | 0.0% |
| Sentinel Prime AI | 15 | 86.7% | 6.7% |
| **Overall** | **50** | **86.0%** |  |

The seven rule-controller mismatches were:

```text
fari_006
fari_007
rover_008
rover_010
rover_014
sentinel_008
sentinel_015
```

These mismatches are analytically useful because they expose boundary cases rather than random failures. They include autonomy versus caregiver escalation, low-confidence health readings, energy infeasibility, safe checkpoint routing, operator handoff, repeated alerts, and bandwidth-aware evidence handling. The rule controller therefore functions as a reasonable baseline. It is not perfect, but the error pattern is interpretable.

## 5. LLM-Judged Scoring Rubric

The LLM judge is a secondary scoring layer. It evaluates the rule controller's candidate action against the scenario's target behavior using a three-level rubric:

| Verdict | Score | Meaning |
|---|---:|---|
| `PASS` | 2 | Candidate matches the target behavior or is clearly equivalent |
| `PARTIAL` | 1 | Candidate is different but still addresses the main constraint without a clear safety downgrade |
| `FAIL` | 0 | Candidate misses the main constraint, violates the target behavior, or creates an unsafe escalation or de-escalation |

The final model used was `google/flan-t5-base`, run locally through Hugging Face Transformers. The final protocol used a strict single-label classification format. The model was asked to return only one class label. Free-form prose and multi-label outputs were not accepted. One deterministic retry was available after an invalid sampled output.

The final seeded judge protocol used three seeds:

```text
7, 17, 27
```

For each seed, the model scored all 50 scenarios. This produced:

```text
50 scenarios × 3 judge seeds = 150 LLM judgments
```

The final run had a 100 percent parse rate, 0 percent retry rate, and no invalid outputs. This is an important improvement over earlier free-generation experiments, where permissive parsing could misclassify multi-label or rubric-repetition outputs as valid judgments.

## 6. Reliability Analysis

Reliability was measured using nominal Krippendorff's alpha across the three seeded judge runs. Pairwise Cohen's kappa was also computed across seed pairs. Krippendorff's alpha was chosen as the main reliability statistic because the judge produced nominal categories and there were three repeated judge runs rather than a single pair of raters.

The overall result was:

```text
Overall nominal Krippendorff's alpha = 0.725
Exact three-run agreement = 78.0%
Parse rate = 100.0%
```

This indicates tentative agreement. The judge is useful for exploratory secondary scoring with human review, but it should not be used as an autonomous scorer.

Domain-level reliability varied substantially:

| Scope | Scenario count | Valid judgments | Three-run agreement | Krippendorff's alpha | Interpretation |
|---|---:|---:|---:|---:|---|
| Overall | 50 | 150 | 78.0% | 0.725 | Tentative agreement, useful with review |
| Aido Rover | 15 | 45 | 93.3% | 0.905 | Strong agreement |
| Sentinel Prime AI | 15 | 45 | 80.0% | 0.748 | Tentative agreement |
| Fari | 10 | 30 | 70.0% | 0.627 | Limited agreement |
| Senpai | 10 | 30 | 60.0% | 0.366 | Poor agreement |

Pairwise Cohen's kappa was also positive across all seed pairs:

| Judge seeds | Common scenarios | Cohen's kappa |
|---|---:|---:|
| 7 vs 17 | 50 | 0.621 |
| 7 vs 27 | 50 | 0.730 |
| 17 vs 27 | 50 | 0.817 |

The pattern is consistent with the alpha results. The model was most stable for structured Rover scenarios. It was moderately stable for Sentinel mode-switching cases. It was less reliable for Fari and especially Senpai, where contextual judgment, conversational nuance, and safeguarding or support decisions made the rubric harder for the model to apply consistently.

## 7. Criterion Alignment and Error Behavior

Reliability is not the same as correctness. A judge can be consistent but consistently miscalibrated. For that reason, I also compared LLM verdicts with the authoritative exact-match criterion.

The final verdict distribution was:

| Verdict | Count |
|---|---:|
| `PASS` | 77 |
| `PARTIAL` | 5 |
| `FAIL` | 68 |
| **Total** | **150** |

By exact-match status:

| Exact match | FAIL | PARTIAL | PASS | Total |
|---|---:|---:|---:|---:|
| False | 21 | 0 | 0 | 21 |
| True | 47 | 5 | 77 | 129 |
| **Total** | **68** | **5** | **77** | **150** |

This shows two important things. First, the judge never assigned `PASS` to a non-exact rule-controller action. That is encouraging because it means the model was conservative and did not falsely accept any known non-exact action. Second, the judge assigned `PASS` to only 77 of 129 exact action matches, or 59.7 percent. This is a major false-negative issue. In many cases, the candidate action and target action were identical, but the model still returned `FAIL` or `PARTIAL`, likely because it compared rationale wording rather than prioritizing the action-label match.

This means the LLM judge is conservative, but not well calibrated. It is better at rejecting non-exact actions than recognizing exact matches. Therefore, the deterministic exact-match score must remain the authoritative correctness measure. The LLM score should be treated as a secondary interpretation layer.

The `PARTIAL` category was also underused. Only 5 of 150 judgments were `PARTIAL`, and all 21 non-exact judgments were labeled `FAIL`. This suggests that the model did not fully learn the distinction between "safe but suboptimal" and "incorrect or unsafe." For future iterations, the non-exact cases should be expanded and manually balanced to give the model more opportunities to distinguish partial credit from failure.

## 8. Seed Disagreement Analysis

The three seeded judge runs disagreed on 11 of 50 scenarios:

```text
fari_002
fari_008
fari_009
rover_007
senpai_001
senpai_006
senpai_007
senpai_008
sentinel_003
sentinel_009
sentinel_012
```

The disagreement distribution by domain was:

| Domain | Disagreement scenarios |
|---|---:|
| Aido Rover | 1 |
| Fari | 3 |
| Senpai | 4 |
| Sentinel Prime AI | 3 |

This aligns with the domain-level alpha scores. Aido Rover had the fewest disagreements and the highest alpha. Senpai had the most disagreement and the lowest alpha. This is consistent with the task structure. Rover scenarios use numeric and rule-like constraints such as energy, route feasibility, localization, and priority. Senpai scenarios depend more heavily on conversational context, emotion, ambiguity, learning support, and safeguarding boundaries.

The disagreement cases are not a failure of the harness. They are useful signals. They identify scenario types where a small open model is least stable and where a future engineer should either clarify the scenario contract, add more structured inputs, or route the decision to human review.

## 9. Usefulness of the Harness

The harness is useful in four ways.

First, it provides a reproducible scenario bank. Every case is explicit, versioned, and schema-validated. This makes the evaluation easier to inspect than an ad hoc list of prompts.

Second, it separates decision generation from decision scoring. The rule-based controller produces candidate actions. The deterministic exact-match scorer measures contract accuracy. The LLM judge provides a secondary rubric. This separation prevents the LLM from both defining and judging correctness.

Third, it quantifies judge reliability instead of assuming it. The alpha and kappa results show where the model is stable and where it is not. That is especially important because the model appears more reliable for structured robotics decisions than for conversational tutoring decisions.

Fourth, it gives a future engineer a clear path to extension. New scenarios can be added through YAML, validated with the schema, and scored with the existing scripts. No engineer needs to read the controller implementation to understand the scenario contract.

The overall result supports using the harness as an exploratory evaluation tool with human review. It does not support using the LLM judge as the sole scorer.

## 10. Known Limitations

The most important limitation is that all scenarios are synthetic. They are designed to be publication-safe and product-inspired, but they are not field logs or validated operational test cases.

The second limitation is that `target_action` is analyst-defined. The targets are reasonable for the scenario contracts, but they should not be interpreted as official InGen production policy.

The third limitation is action-label compression. Real decisions may need multi-step plans, uncertainty ranges, escalation chains, or human-in-the-loop states. A single label cannot represent all operational nuance.

The fourth limitation is judge calibration. Although `google/flan-t5-base` produced parseable outputs and moderate overall seeded agreement, it showed substantial false-negative behavior against exact action matches. It also rarely used the `PARTIAL` category.

The fifth limitation is that the rule controller is intentionally simple. It is useful as a transparent baseline, but it is not intended to optimize across all constraints.

The sixth limitation is domain imbalance in difficulty. Senpai and Fari contain more conversational and human-centered judgments, while Rover and Sentinel contain more structured operational constraints. Reliability comparisons should therefore be read as domain-specific evidence, not as proof that one product context is inherently easier.

The seventh limitation is that three judge seeds provide a basic reliability check, not a complete robustness study. More seeds, more balanced non-exact scenarios, additional models, and human annotations would be needed for a stronger evaluation.

## 11. Recommended Next Iterations

The next version should add more non-exact cases so that `PARTIAL` and `FAIL` can be meaningfully separated. At present, the LLM judge mostly treats all non-exact actions as `FAIL`.

A second improvement would be to add human review labels for a subset of scenarios. This would allow criterion validity to be measured against expert judgment, not only against action-label exact match.

A third improvement would be to add confidence calibration. The judge currently returns only a discrete verdict. A future version could ask for a confidence score and test whether low-confidence judgments correlate with seed disagreement.

A fourth improvement would be to evaluate additional small open models. The current result is specific to `google/flan-t5-base` under the selected prompt, seeds, and decoding settings.

A fifth improvement would be to expand scenario metadata. Adding fields such as `safety_priority`, `ambiguity_level`, `human_override_required`, and `partial_credit_expected` would make error analysis more precise.

## 12. Bottom Line

The Week 6 harness successfully creates a reproducible decision-evaluation workflow across 50 synthetic scenarios. The rule-based controller achieved 86 percent exact-match accuracy. The LLM judge produced 150 valid seeded judgments with a 100 percent parse rate and an overall Krippendorff's alpha of 0.725.

The reliability result is useful but limited. The LLM judge showed tentative overall agreement and strong performance on structured Rover decisions, but weaker reliability in conversational and care-oriented scenarios. It was also conservative, assigning no `PASS` labels to non-exact actions while failing to recognize many exact matches as `PASS`.

The harness is therefore useful as a structured evaluation and error-analysis tool. It is not reliable enough for fully automated scoring. The deterministic target-action score should remain authoritative, and the LLM judge should be treated as a secondary, review-assisted rubric layer.

## References

Krippendorff, K. (2011). Computing Krippendorff's Alpha Reliability. Departmental Papers, Annenberg School for Communication, University of Pennsylvania.

Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37-46.

Hugging Face. `google/flan-t5-base` model card. https://huggingface.co/google/flan-t5-base

Hugging Face Transformers documentation. https://huggingface.co/docs/transformers
