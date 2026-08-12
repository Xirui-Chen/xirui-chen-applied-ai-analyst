# Xirui Chen Applied AI Analyst Internship

**Eight-Week Applied AI / Physical AI Internship Portfolio**  
**InGen Dynamics | June–August 2026**  
**Final release target: `v1.0`**

## Project Overview

This repository contains the public, reproducible deliverables from my eight-week Applied AI Analyst internship with InGen Dynamics.

The project connects strategic research with applied technical work across physical AI, robotics analytics, anomaly detection, reinforcement learning, model evaluation, and analyst-facing decision support. The work progressed from market and platform research in Weeks 1–2, to telemetry and model benchmarking in Weeks 3–6, to dashboard integration and capstone synthesis in Weeks 7–8.

The main product anchors referenced throughout the project are:

- **Fari**: eldercare and companionship
- **Senpai**: education and tutoring
- **Sentinel Prime AI**: security intelligence and monitoring
- **Aido Rover**: patrol and inspection
- **Aido Humanoid**: general embodied robotics
- **Origami / PIC 2.0**: shared physical-intelligence platform

All public repository artifacts use public information, open-source software, synthetic telemetry, simulation environments, or analyst-defined synthetic evaluation scenarios. No production customer, patient, student, security, or robot-operation data is included.

---

## Capstone Conclusion

The central conclusion of the eight-week project is:

> **InGen's portfolio breadth becomes a strategic advantage only when the products share a common evidence system.**

From my analysis, the strongest cross-portfolio opportunity is not one universal robot or one universal model. It is a shared evidence and governance layer that standardizes telemetry contracts, evaluation contracts, uncertainty, safety rules, release metadata, and deployment metrics while allowing each product to retain product-specific models and operating policies.

The capstone therefore recommends:

1. a shared telemetry and provenance contract across products
2. a common evaluation registry for faults, policies, decision scenarios, seeds, thresholds, and release evidence
3. product-specific models and deterministic safeguards rather than a one-model-fits-all strategy
4. explicit human governance for uncertain or high-stakes decisions
5. a staged evidence path from synthetic testing to simulation, pilot validation, monitored deployment, and post-deployment recalibration

See `week08_capstone/` for the final report, executive deck, and the supervisor-requested LSTM battery-degradation follow-up.

---

## Eight-Week Workstream

| Week | Focus | Main output |
|---|---|---|
| 1 | Physical AI landscape and PIC 2.0 | Research synthesis and conceptual map |
| 2 | Competitive intelligence | 22-vendor matrix and strategic-gap analysis |
| 3 | Aido Rover telemetry | Synthetic telemetry generator, EDA, feature framework |
| 4 | Anomaly detection | Five-fault, four-method benchmark across 10 seeded splits |
| 5 | Reinforcement learning | PPO/SAC single-agent benchmarks and multi-agent diagnostics |
| 6 | Decision-model evaluation | 50-scenario harness, 150 seeded LLM judgments, reliability analysis |
| 7 | Analyst dashboard | Streamlit integration for Engineering, Product, and Customer Success |
| 8 | Capstone | Final synthesis, LSTM battery follow-up, executive recommendations, retrospective |

---

## Repository Structure

```text
xirui-chen-applied-ai-analyst/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── data/
│   ├── w03_rover_smoke/
│   └── w03_rover_5units_1day/
│
├── week01_landscape/
│   ├── W01_PhysicalAI_Landscape_Brief.md
│   ├── W01_PIC20_Conceptual_Map.md
│   └── W01_env_check.ipynb
│
├── week02_competitive/
│   ├── W02_Competitive_Matrix.xlsx
│   └── W02_Strategic_Gap_Memo.md
│
├── week03_telemetry/
│   ├── W03_Telemetry_Generator.py
│   ├── W03_Telemetry_EDA.ipynb
│   ├── W03_Feature_Dictionary.md
│   └── figures/
│
├── week04_anomaly/
│   ├── W04_Anomaly_Benchmark.ipynb
│   ├── W04_Method_Recommendation_Memo.md
│   ├── W04_MidPoint_Deck.pdf
│   ├── W04_MidPoint_Research_Summary_Report.pdf
│   └── results/
│
├── week05_rl/
│   ├── W05_RL_Analyst_Memo.md
│   └── W05_RL_Benchmarks/
│       ├── README.md
│       ├── configs/
│       ├── scripts/
│       ├── requirements_w05.txt
│       ├── requirements_w05_lock.txt
│       ├── results/
│       ├── plots/
│       └── tensorboard/
│
├── week06_decision_eval/
│   ├── W06_Eval_Methodology.md
│   └── W06_Decision_Eval_Harness/
│       ├── README.md
│       ├── config.yaml
│       ├── scenarios/
│       ├── scorer/
│       └── results/
│
├── week07_dashboard/
│   ├── W07_Dashboard/
│   │   ├── app.py
│   │   ├── dashboard/
│   │   ├── sample_data/
│   │   ├── scripts/
│   │   ├── launch_dashboard.sh
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── W07_Dashboard_Design_Doc.md
│   ├── W07_Dashboard_Walkthrough.mp4
│   └── annotated_screenshots/
│
├── week08_capstone/
│   ├── W08_Capstone_Report.pdf
│   ├── W08_Capstone_Deck.pdf
│   └── W08_LSTM_Battery_Degradation_Followup/
│       ├── W08_LSTM_Battery_Degradation_Analysis.md
│       ├── W08_LSTM_Battery_Degradation_Analysis.ipynb
│       ├── analysis_driver.py
│       ├── run_followup.sh
│       ├── requirements_w08_followup.txt
│       ├── figures/
│       └── results/
│
├── weekly/
│   ├── Wk-01-Recap.md
│   ├── Wk-02-Recap.md
│   ├── Wk-03-Recap.md
│   ├── Wk-04-Recap.md
│   ├── Wk-05-Recap.md
│   ├── Wk-06-Recap.md
│   ├── Wk-07-Recap.md
│   └── Wk-08-Retrospective.md
│
└── docs/
    └── Internship_Plan.pdf
```

The repository may contain additional generated CSVs, plots, manifests, and supporting result files inside the quantitative-week folders. The structure above highlights the primary reviewer-facing artifacts and reproducibility entry points.

---

## Environment

The project is designed around **Python 3.11**.

Major libraries include:

- pandas, NumPy, SciPy, statsmodels
- scikit-learn and PyOD
- PyTorch
- Gymnasium and Stable-Baselines3
- PettingZoo
- matplotlib, plotly, Streamlit
- transformers, sentencepiece, accelerate
- PyYAML and krippendorff
- JupyterLab

The root environment is defined in `requirements.txt`. Weeks 5, 6, 7, and 8 also include local requirement files where a narrower or more reproducible environment is useful.

### Git LFS

The Week 7 narrated walkthrough is stored with **Git LFS** because the MP4 exceeds GitHub's normal per-file Git limit.

Install Git LFS before cloning if you need the video file:

```bash
git lfs install
```

On macOS with Homebrew:

```bash
brew install git-lfs
git lfs install
```

---

# Clean-Clone Reproduction

The final release is intended to be reproducible from a clean clone using the instructions below.

## 1. Clone the repository

For the final tagged release:

```bash
git clone https://github.com/Xirui-Chen/xirui-chen-applied-ai-analyst.git
cd xirui-chen-applied-ai-analyst
git checkout v1.0
git lfs pull
```

If reproducing the latest development state before the release tag is created, use `main` instead of `v1.0`.

## 2. Create the root Python environment

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 3. Verify the base environment

From the repository root:

```bash
jupyter lab
```

Open and run:

```text
week01_landscape/W01_env_check.ipynb
```

---

# Week-by-Week Reproduction Guide

## Week 1: Physical AI Landscape and PIC 2.0

**Artifacts**

- `week01_landscape/W01_PhysicalAI_Landscape_Brief.md`
- `week01_landscape/W01_PIC20_Conceptual_Map.md`
- `week01_landscape/W01_env_check.ipynb`

Week 1 is primarily a research and synthesis deliverable. The Markdown artifacts are source-cited analytical documents rather than outputs of a computational pipeline.

To reproduce the environment check, launch JupyterLab from the repository root and run all cells in `week01_landscape/W01_env_check.ipynb`.

---

## Week 2: Competitive Landscape and Strategic Gaps

**Artifacts**

- `week02_competitive/W02_Competitive_Matrix.xlsx`
- `week02_competitive/W02_Strategic_Gap_Memo.md`

Week 2 is a public-source competitive-intelligence deliverable. The matrix records the reviewed vendors, comparison fields, source citations, and analysis date. The strategic memo is derived from that matrix.

No model training is required. Market information should be treated as an **as-of snapshot**, not a live market database.

---

## Week 3: Synthetic Aido Rover Telemetry

**Artifacts**

- `week03_telemetry/W03_Telemetry_Generator.py`
- `week03_telemetry/W03_Telemetry_EDA.ipynb`
- `week03_telemetry/W03_Feature_Dictionary.md`

The generator is deterministic under a fixed seed and writes robot-day partitions.

### Smoke test

From the repository root:

```bash
python week03_telemetry/W03_Telemetry_Generator.py \
  --seed 42 \
  --fleet-size 2 \
  --horizon-days 0.02 \
  --output-dir data/w03_rover_smoke_repro \
  --format csv.gz \
  --verify-reproducibility
```

### Five-unit, one-day sample

A committed downstream-analysis sample is available at:

```text
data/w03_rover_5units_1day/
```

To regenerate an equivalent analysis sample without overwriting the committed copy:

```bash
python week03_telemetry/W03_Telemetry_Generator.py \
  --seed 42 \
  --fleet-size 5 \
  --horizon-days 1 \
  --output-dir data/w03_rover_5units_1day_repro \
  --format csv.gz \
  --verify-reproducibility
```

### Full-scale generation

```bash
python week03_telemetry/W03_Telemetry_Generator.py \
  --seed 42 \
  --fleet-size 50 \
  --horizon-days 30 \
  --output-dir data/w03_rover_telemetry \
  --format csv.gz
```

The full 50-unit, 30-day output is large and is not required for downstream reproduction.

### EDA

Launch JupyterLab from the repository root and run:

```text
week03_telemetry/W03_Telemetry_EDA.ipynb
```

---

## Week 4: Anomaly Detection Benchmark

**Artifacts**

- `week04_anomaly/W04_Anomaly_Benchmark.ipynb`
- `week04_anomaly/W04_Method_Recommendation_Memo.md`
- `week04_anomaly/results/`

The benchmark uses the committed Week 3 sample:

```text
data/w03_rover_5units_1day/
```

From the repository root, launch JupyterLab and run all cells in:

```text
week04_anomaly/W04_Anomaly_Benchmark.ipynb
```

The benchmark injects five controlled fault classes only into held-out periods:

1. motor stall
2. gradual sensor drift
3. accelerated battery degradation
4. GPS jitter / spoofing-like behavior
5. intermittent software hang

It compares Isolation Forest, One-Class SVM, Local Outlier Factor, and an LSTM Autoencoder across 10 seeded train/test splits.

The committed result selected **LOF** as the preferred accuracy-maximizing detector, with One-Class SVM as a close challenger.

---

## Week 5: Reinforcement-Learning Benchmarks

The complete Week 5 guide is:

```text
week05_rl/W05_RL_Benchmarks/README.md
```

### Create the Week 5 environment

```bash
cd week05_rl/W05_RL_Benchmarks
python3.11 -m venv .venv_w05
source .venv_w05/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_w05_lock.txt
```

If the locked environment cannot be installed on the current platform, use `requirements_w05.txt` and record the resolved versions.

### Verify the environment

```bash
python scripts/check_environment.py
```

### Smoke test

```bash
python scripts/run_all.py \
  --profile smoke \
  --seeds 7 \
  --include single multi \
  --progress
```

### Full benchmark

```bash
python scripts/run_all.py \
  --profile standard \
  --include single multi \
  --progress
```

The standard grid uses seeds `7, 17, 27, 37, 47`.

### Rebuild tables and figures without retraining

```bash
python scripts/aggregate_results.py
python scripts/plot_learning_curves.py
```

Week 5 is a controlled learning benchmark, not evidence of physical-robot policy readiness.

---

## Week 6: Decision Evaluation Harness

The complete Week 6 guide is:

```text
week06_decision_eval/W06_Decision_Eval_Harness/README.md
```

### Create the Week 6 environment

```bash
cd week06_decision_eval/W06_Decision_Eval_Harness
python3.11 -m venv .venv_w06
source .venv_w06/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_w06_lock.txt
```

If needed, `requirements_w06.txt` is the less restrictive alternative.

### Validate the scenario bank

```bash
python scorer/validate_scenarios.py
```

Expected counts:

- 15 Aido Rover
- 15 Sentinel Prime AI
- 10 Fari
- 10 Senpai

### Rule-only smoke test

```bash
python scorer/run_evaluation.py --skip-llm --max-scenarios 8
```

### Full deterministic rule baseline

```bash
python scorer/run_evaluation.py --skip-llm
```

### Full seeded LLM evaluation

The configured judge is `google/flan-t5-base`.

```bash
python scorer/check_environment.py --check-model

python scorer/run_evaluation.py \
  --model google/flan-t5-base \
  --judge-seeds 7 17 27 \
  --device auto
```

### Recompute reliability and summaries

```bash
python scorer/compute_reliability.py
python scorer/summarize_results.py
python scorer/audit_judge_results.py \
  --expected-scenarios 50 \
  --expected-seeds 7 17 27
```

The committed final run contains 50 scenarios, 150 seeded judgments, 78% exact three-run agreement, and overall nominal Krippendorff's alpha of 0.725.

Exact action match remains authoritative. The LLM judge is a secondary semantic-review signal.

---

## Week 7: Streamlit Analyst Dashboard

**Artifacts**

- `week07_dashboard/W07_Dashboard/`
- `week07_dashboard/W07_Dashboard_Design_Doc.md`
- `week07_dashboard/annotated_screenshots/`
- `week07_dashboard/W07_Dashboard_Walkthrough.mp4`

The dashboard integrates final Weeks 3–6 outputs into three workspaces:

1. Fleet Health
2. Policy Benchmark
3. Decision Evaluation

It supports three personas:

- Engineering Manager
- Product Manager
- Customer Success

### One-command launch

```bash
cd week07_dashboard/W07_Dashboard
./launch_dashboard.sh
```

The script creates `.venv_w07`, installs the pinned dependencies, validates the local sample-data contracts, and starts Streamlit at:

```text
http://localhost:8501
```

Alternative port:

```bash
PORT=8502 ./launch_dashboard.sh
```

### Manual launch

```bash
python3.11 -m venv .venv_w07
source .venv_w07/bin/activate
python -m pip install -r requirements.txt
python scripts/validate_data.py
streamlit run app.py
```

The dashboard is self-contained and uses local, public-safe sample data.

---

## Week 8: Capstone and LSTM Battery Follow-Up

**Capstone artifacts**

```text
week08_capstone/W08_Capstone_Report.pdf
week08_capstone/W08_Capstone_Deck.pdf
weekly/Wk-08-Retrospective.md
```

These are synthesis and presentation artifacts and do not require model execution to review.

### Supervisor-requested LSTM follow-up

The reproducible supporting analysis is located at:

```text
week08_capstone/W08_LSTM_Battery_Degradation_Followup/
```

Key result:

- LSTM battery AUROC: **0.566 ± 0.128**
- zero-recall splits: **9 / 10**
- battery events detected at the original threshold: **1 / 30**
- battery-focused post-hoc AUROC: **0.670 ± 0.093**

The follow-up did not overturn the Week 4 recommendation. LOF remains the preferred accuracy-maximizing detector for the current benchmark.

### Reproduce the follow-up

```bash
cd week08_capstone/W08_LSTM_Battery_Degradation_Followup
./run_followup.sh
```

The runner uses the committed Week 3 five-unit telemetry and Week 4 benchmark implementation/results.

---

# Reproducibility Matrix

| Week | Artifact type | Clean-clone reproduction |
|---|---|---|
| 1 | Research + environment check | Run `W01_env_check.ipynb`; research documents are source-cited |
| 2 | Competitive research | Review cited matrix and memo; no model execution required |
| 3 | Synthetic data + EDA | Run deterministic generator and EDA notebook |
| 4 | ML benchmark | Run notebook against committed Week 3 sample |
| 5 | RL benchmark | Use Week 5 environment and `run_all.py` |
| 6 | Decision evaluation | Validate scenarios, run rule baseline, then seeded LLM judge |
| 7 | Dashboard | Run `./launch_dashboard.sh` |
| 8 | Capstone + follow-up | Report/deck are synthesis artifacts; run LSTM follow-up script |

---

# Selected Final Results

## Week 4 anomaly detection

Across 10 seeded splits, LOF was the strongest overall accuracy-oriented detector:

- AUROC: **0.985 ± 0.008**
- Recall: **0.970 ± 0.033**
- F1: **0.651 ± 0.055**

One-Class SVM was a close challenger.

## Week 5 reinforcement learning

The final benchmark showed a quality-versus-cost tradeoff:

- SAC reached substantially stronger final performance on BipedalWalker
- PPO required much less wall-clock time
- the multi-agent parameter-sharing baseline improved assignment distance relative to random behavior but increased collisions and did not reach the analyst-defined coordination threshold

These results support multi-metric policy evaluation rather than selecting an algorithm from reward alone.

## Week 6 decision evaluation

- Scenarios: **50**
- Seeded judgments: **150**
- Exact three-run agreement: **78%**
- Overall Krippendorff's alpha: **0.725**
- Aido Rover alpha: **0.905**
- Sentinel Prime AI alpha: **0.748**
- Fari alpha: **0.627**
- Senpai alpha: **0.366**

The LLM judge is not treated as the source of truth.

## Week 8 battery follow-up

The current LSTM Autoencoder should **not** be presented as a specialized accelerated-battery-degradation detector. Threshold relaxation and additional training epochs did not resolve the weakness. Battery-specific reconstruction channels contained more useful signal than the global reconstruction score, suggesting a future battery-specific formulation rather than simply adding more complexity to the existing model.

---

# Evidence and Data Boundaries

This repository intentionally separates several levels of evidence.

### Public research evidence

Competitive and market claims are based on public sources and are cited in the relevant Week 1, Week 2, and capstone documents.

### Synthetic fleet evidence

Aido Rover telemetry, fault injections, unit-health examples, and dashboard fleet state are synthetic.

### Simulation evidence

Week 5 PPO, SAC, and multi-agent results come from Gymnasium and PettingZoo environments. They are benchmark results, not field-robot results.

### Synthetic decision-evaluation evidence

Week 6 scenarios, target actions, thresholds, and rationales are analyst-defined evaluation contracts. They are not official clinical, educational, security, or autonomous-robot operating policies.

### Dashboard evidence

The Week 7 dashboard is an analyst-facing local prototype. It is not a production monitoring, clinical, safeguarding, or security-dispatch system.

---

# Reproducibility Principles

Across the quantitative work, the repository follows these principles:

- fixed seeds where stochasticity is material
- relative repository paths rather than local absolute paths
- committed configuration and manifest files
- explicit input and output locations
- train/test separation before fault injection or model evaluation
- result tables committed alongside notebooks and scripts when practical
- separate smoke-test and full-benchmark workflows
- environment files scoped to complex weekly pipelines
- public-safe synthetic data for operational examples
- explicit limitations around external validity

Machine-specific wall-clock measurements should be compared only on similar hardware.

Some stochastic ML and deep-learning outputs may show small numerical differences across operating systems, hardware, numerical libraries, or package versions even when the qualitative conclusions remain unchanged.

---

# Clean-Clone Validation Checklist

Before the final `v1.0` release is tagged, the repository can be validated with the following checklist:

```text
[ ] Clone the repository from scratch
[ ] Checkout v1.0, or main before the tag exists
[ ] Pull Git LFS objects
[ ] Create a Python 3.11 environment
[ ] Install root requirements
[ ] Run the Week 1 environment check
[ ] Run the Week 3 telemetry smoke test
[ ] Confirm the Week 4 notebook resolves the committed Week 3 input
[ ] Run the Week 5 environment check and smoke profile
[ ] Validate all 50 Week 6 scenarios
[ ] Run a Week 6 rule-only smoke test
[ ] Launch the Week 7 dashboard with ./launch_dashboard.sh
[ ] Run the Week 8 LSTM follow-up
[ ] Confirm reviewer-facing Markdown, PDF, CSV, and dashboard artifacts resolve from repository-relative paths
```

Check for accidental local paths:

```bash
grep -R "/Users/" . --exclude-dir=.git || true
grep -R "Downloads/" . --exclude-dir=.git || true
```

Notebook outputs can contain historical display text from local development, so any match should be reviewed rather than automatically treated as an active dependency.

---

# Final Release: v1.0

The final internship repository is intended to be frozen under the annotated Git tag:

```text
v1.0
```

The release represents the complete eight-week body of work, including:

- physical-AI research and competitive analysis
- synthetic telemetry framework
- anomaly-detection benchmark
- reinforcement-learning benchmarks
- decision-evaluation harness
- Streamlit dashboard and walkthrough
- supervisor-requested LSTM battery follow-up
- capstone report and executive deck
- weekly recaps and retrospective
- clean-clone reproduction instructions

After the final commit is validated:

```bash
git tag -a v1.0 -m "Final Applied AI Analyst Internship Capstone Release"
git push origin v1.0
```

A GitHub Release can then be created from the same `v1.0` tag.

---

## Author

**Xirui Chen**  
Applied AI Analyst Intern  
InGen Dynamics  
2026
