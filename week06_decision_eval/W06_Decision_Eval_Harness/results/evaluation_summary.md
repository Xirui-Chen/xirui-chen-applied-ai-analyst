# W06 Evaluation Summary

## Domain-level evaluation summary

`judge_mean_score` is the mean ordinal rubric score (FAIL=0, PARTIAL=1, PASS=2), not accuracy.

| domain         |   scenario_count |   rule_exact_accuracy |   fallback_rate |   judge_mean_score |
|:---------------|-----------------:|----------------------:|----------------:|-------------------:|
| aido_rover     |               15 |              0.8      |       0         |           0.711111 |
| fari           |               10 |              0.8      |       0         |           1.36667  |
| senpai         |               10 |              1        |       0         |           1.5      |
| sentinel_prime |               15 |              0.866667 |       0.0666667 |           0.911111 |

## LLM judge reliability

| scope   | domain         |   n_scenarios |   n_valid_judgments |   parse_rate |   complete_scenario_count |   exact_three_run_agreement |   krippendorff_alpha_nominal | interpretation                                 |
|:--------|:---------------|--------------:|--------------------:|-------------:|--------------------------:|----------------------------:|-----------------------------:|:-----------------------------------------------|
| overall | all            |            50 |                 150 |            1 |                        50 |                    0.78     |                     0.725046 | tentative agreement; useful with review        |
| domain  | aido_rover     |            15 |                  45 |            1 |                        15 |                    0.933333 |                     0.905172 | strong agreement for this test bank            |
| domain  | fari           |            10 |                  30 |            1 |                        10 |                    0.7      |                     0.626609 | limited agreement; unsuitable as a sole scorer |
| domain  | senpai         |            10 |                  30 |            1 |                        10 |                    0.6      |                     0.36612  | poor agreement; unreliable for automated use   |
| domain  | sentinel_prime |            15 |                  45 |            1 |                        15 |                    0.8      |                     0.748092 | tentative agreement; useful with review        |

## Interpretation boundary

Exact match against `target_action` is authoritative. The small-model LLM judge is an experimental secondary layer. Agreement does not prove correctness, and low agreement limits automated use.
