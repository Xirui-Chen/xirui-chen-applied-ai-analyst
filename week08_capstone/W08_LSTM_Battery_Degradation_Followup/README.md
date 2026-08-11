# W08 LSTM Battery Degradation Follow-Up

This folder addresses the supervisor request to analyze the LSTM Autoencoder on `accelerated_battery_degradation` before the capstone is finalized.

## Bottom line

The configured Week 4 LSTM Autoencoder is not suitable for this fault under the current benchmark. Its AUROC is **0.566 ± 0.128**, nine of ten splits have zero recall, and only **1 of 30** injected battery events are detected at least once at the original threshold.

The follow-up shows that:

- looser thresholds increase clean-window alerting faster than useful battery recall
- 5 and 10 training epochs do not materially improve the result
- battery-specific reconstruction channels contain more signal than the global 17-feature MSE
- a battery-focused reconstruction score improves AUROC to **0.670 ± 0.093**, but still trails LOF and One-Class SVM
- the Week 4 recommendation therefore remains unchanged: **LOF is the preferred accuracy-maximizing detector**

## Recommended repository location

Place the contents of this folder under:

```text
week08_capstone/
└── supporting_analysis/
```

## Files

```text
supporting_analysis/
├── W08_LSTM_Battery_Degradation_Analysis.md
├── W08_LSTM_Battery_Degradation_Analysis.ipynb
├── analysis_driver.py
├── run_followup.sh
├── requirements_w08_followup.txt
├── artifact_manifest.json
├── figures/
└── results/
```

The Markdown report is the supervisor-facing artifact. The notebook, result tables, and driver provide traceability and reproducibility.

## Reproduce the analysis

From `week08_capstone/supporting_analysis/`:

```bash
./run_followup.sh
```

The runner assumes the repository contains the final Week 4 notebook/results and the Week 3 five-unit telemetry sample. It reuses the exact Week 4 implementation cells for fault injection, feature engineering, and the LSTM Autoencoder.

No production or confidential data is included.
