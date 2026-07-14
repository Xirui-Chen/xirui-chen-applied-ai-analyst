# W04 Method Recommendation Memo

**Deliverable:** `W04_Method_Recommendation_Memo.md`  
**Prepared by:** Xirui (Crissy) Chen  
**Input artifacts:** `W04_Anomaly_Benchmark.ipynb`, `w04_metrics_summary_mean_std.csv`, `w04_wilcoxon_auroc_pairwise.csv`, `w04_fault_injection_ledger.csv`  
**Benchmark setting:** Week 3 synthetic Aido Rover telemetry, 5 units, 1 day, 1 Hz, with five controlled fault patterns injected only into held-out test blocks.

---

## Recommendation

For the current Week 4 benchmark, I would use **Local Outlier Factor (LOF)** as the preferred method when the operating goal is **accuracy-maximizing anomaly detection**, especially for offline review, fleet-health analytics, and analyst-facing triage. LOF achieved the strongest overall benchmark profile: **precision 0.492 ± 0.060, recall 0.970 ± 0.033, F1 0.651 ± 0.055, and AUROC 0.985 ± 0.008** across the 10 seeded splits. It also had the best AUROC for four of the five fault types: motor stall, gradual sensor drift, accelerated battery degradation, and GPS jitter. For intermittent software hang, One-Class SVM was essentially tied with LOF, with One-Class SVM at **0.996 ± 0.002 AUROC** and LOF at **0.996 ± 0.001 AUROC**.

For a **latency-constrained edge or near-real-time operating regime**, I would use **Isolation Forest** as the first-pass lightweight detector only if compute and memory budgets are strict. It is easier to deploy as a fast screening model because inference does not require the same neighbor-search behavior as LOF and does not require sequence reconstruction like the LSTM autoencoder. However, this recommendation has an important caveat: in the benchmark, Isolation Forest was clearly less accurate overall, with **0.844 ± 0.044 AUROC** and **0.365 ± 0.074 F1** across all faults. Its strongest performance was on motor stall, where it reached **0.965 ± 0.029 AUROC**, but it was materially weaker on accelerated battery degradation and GPS jitter. In a production-like Rover or Sentinel monitoring workflow, I would not rely on Isolation Forest alone. I would pair it with explicit rule-based guards for battery degradation, GPS consistency, stale telemetry, and task-success degradation.

---

## Evidence from the Benchmark

| Method | Overall Precision | Overall Recall | Overall F1 | Overall AUROC |
|---|---:|---:|---:|---:|
| Isolation Forest | 0.373 ± 0.129 | 0.394 ± 0.096 | 0.365 ± 0.074 | 0.844 ± 0.044 |
| One-Class SVM | 0.458 ± 0.080 | 0.966 ± 0.022 | 0.617 ± 0.075 | 0.981 ± 0.007 |
| LOF | **0.492 ± 0.060** | **0.970 ± 0.033** | **0.651 ± 0.055** | **0.985 ± 0.008** |
| LSTM Autoencoder | 0.382 ± 0.082 | 0.624 ± 0.069 | 0.466 ± 0.063 | 0.848 ± 0.037 |

The paired Wilcoxon signed-rank tests support the accuracy recommendation. On the overall AUROC comparison, LOF outperformed Isolation Forest by a mean AUROC difference of **+0.141** with **p = 0.002**, and LOF outperformed the LSTM autoencoder by **+0.137** with **p = 0.002**. LOF also outperformed One-Class SVM by a smaller mean AUROC difference of **+0.003**, with **p = 0.0195**. This means the LOF advantage over One-Class SVM is statistically detectable in this benchmark, but operationally small. For that reason, I would treat One-Class SVM as the strongest challenger rather than as a clearly inferior method.

Per fault type, the method ranking is more nuanced. LOF performed best on **motor stall** with **0.998 ± 0.001 AUROC**, **gradual sensor drift** with **0.992 ± 0.008**, **accelerated battery degradation** with **0.982 ± 0.028**, and **GPS jitter** with **0.956 ± 0.017**. One-Class SVM was marginally best on **intermittent software hang** with **0.996 ± 0.002 AUROC**, although LOF was effectively tied at **0.996 ± 0.001**. This supports a practical recommendation: LOF should be the default accuracy model, but software-hang detection should also use direct stale-value and missingness features rather than depending entirely on a generic anomaly score.

---

## Operating-Regime Recommendation

### 1. Latency-Constrained Monitoring

**Preferred method:** Isolation Forest, with explicit feature rules and conservative escalation.

This regime applies when the system needs fast scoring on edge hardware or near-real-time fleet monitoring. The goal is not to maximize every AUROC point. The goal is to catch obvious high-risk anomalies quickly, keep the model simple, and avoid high memory or sequence-processing overhead. Isolation Forest fits this role because it is simple, scalable, and easy to retrain. It is especially reasonable for abrupt propulsion issues such as motor stall.

**Caveat:** The benchmark does not support Isolation Forest as the best standalone detector. It had low recall overall and underperformed LOF and One-Class SVM on most faults. If used in this regime, it should be deployed as a first-pass screen and paired with deterministic checks, such as voltage-versus-SoC residual thresholds, GPS HDOP and fix-quality consistency checks, stale telemetry checks, and motor-current imbalance limits.

### 2. Accuracy-Maximizing Fleet Analytics

**Preferred method:** LOF.

This regime applies when the system can score one-minute windows in batch or nearline mode, such as nightly fleet-health review, analyst dashboarding, or benchmark-driven model comparison. LOF is preferred because it delivered the strongest overall AUROC, recall, and F1. It was also robust across several structurally different faults: propulsion, sensor drift, battery degradation, and GPS instability.

**Caveat:** LOF can be more expensive at inference time because novelty scoring depends on distances to the fitted training reference set. It may also require careful scaling and sampling when the full 50-unit, 30-day dataset is used. Before deployment, it should be stress-tested on larger data, route-specific partitions, and different contamination settings.

### 3. High-Stakes Escalation or Safety Review

**Preferred approach:** LOF plus rule-based safety checks, with One-Class SVM as a challenger model.

For safety-sensitive monitoring, a single unsupervised model is not enough. LOF should provide the primary anomaly score, while rule-based features should protect against failure modes that are operationally obvious but statistically rare. One-Class SVM should remain in the benchmark suite because its performance was close to LOF overall and essentially tied for software hang. This gives the team a useful challenger when LOF behavior becomes unstable under larger-scale or more heterogeneous data.

---

## Why the LSTM Autoencoder Is Not Preferred Yet

The LSTM autoencoder is conceptually attractive because several faults have temporal structure. However, in this benchmark it did not justify its complexity. Its overall AUROC was **0.848 ± 0.037**, similar to Isolation Forest and far below LOF and One-Class SVM. It also performed poorly on accelerated battery degradation, with **0.566 ± 0.128 AUROC**. The likely reason is that the current sequence length, sample size, and synthetic fault design are not enough for a sequence model to learn useful normal dynamics. The model also introduces training time, random initialization sensitivity, hyperparameter tuning, and deployment complexity. I would revisit the LSTM autoencoder only after longer-horizon data, richer sequence labels, and more realistic temporal fault patterns are available.

---

## Assumptions and Limitations

This recommendation is based on the Week 4 synthetic benchmark, not real Aido Rover field telemetry. The injected faults are controlled and useful for method comparison, but they may be cleaner and more separable than real-world failures. The benchmark uses one-minute windows, so it does not measure sub-second alert latency. It also reports model quality metrics, not actual wall-clock inference latency or memory footprint. Therefore, the latency recommendation is based on deployment complexity and algorithmic behavior, not direct runtime profiling.

The dataset used here covers **5 Rover units over 1 day**, while the Week 3 generator supports a larger **50-unit, 30-day** horizon. Results should be re-run on a larger sample before drawing final deployment conclusions. Finally, the thresholding strategy uses training-score quantiles under a fixed contamination assumption. In real operations, the alert threshold should be calibrated against operator workload, false-alert tolerance, safety severity, and escalation cost.

---

## Final Decision

Use **LOF as the primary accuracy-maximizing anomaly detector** for the Week 4 benchmark and recommendation path. Use **Isolation Forest only for latency-constrained first-pass screening**, and only with explicit rule-based safeguards. Keep **One-Class SVM as a close challenger**, especially for software-hang-like faults. Do **not** promote the LSTM autoencoder yet; treat it as a future research option once longer and more realistic time-series data is available.
