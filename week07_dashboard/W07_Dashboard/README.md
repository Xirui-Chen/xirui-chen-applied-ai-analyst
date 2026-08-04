# W07 Dashboard

## Overview

This Streamlit application synthesizes the final Weeks 3–6 outputs into one
analyst-facing dashboard for three personas:

| Persona | Primary question | Main control surface |
|---|---|---|
| Engineering Manager | Are models precise, stable, and operationally safe? | Detector/fault filters, precision-recall metrics, seeded stability, reliability drill-down |
| Product Manager | Which method or platform is performing best, and at what cost? | Algorithm/environment comparison, learning curves, wall-clock tradeoff, domain scorecards |
| Customer Success | Which fleet units need attention, and what should happen next? | Unit status, alert severity, unit selector, telemetry history, recommended action |

The dashboard has three workspaces:

1. **Fleet Health**  
   Week 4 anomaly results plus a five-unit Aido Rover operational snapshot
   derived from synthetic Week 3 telemetry.

2. **Policy Benchmark**  
   Week 5 PPO/SAC convergence, stability, threshold success, wall-clock cost,
   and multi-agent diagnostics.

3. **Decision Evaluation**  
   Week 6 rule accuracy, FLAN-T5-base verdict distributions, Krippendorff
   alpha, pairwise kappa, seed disagreements, and scenario-level review.

## One-command launch

From the `W07_Dashboard/` directory:

```bash
./launch_dashboard.sh
```

The script:

1. Finds Python 3.11 when available, otherwise uses `python3`.
2. Creates `.venv_w07/`.
3. Installs the pinned dependencies only when `requirements.txt` changes.
4. Validates all sample-data contracts.
5. Opens the app at `http://localhost:8501`.

To use another port:

```bash
PORT=8502 ./launch_dashboard.sh
```

A clean first launch requires internet access for package installation. After
the virtual environment is prepared, the dashboard uses only local sample data.

## Manual launch

```bash
python3.11 -m venv .venv_w07
source .venv_w07/bin/activate
python -m pip install -r requirements.txt
python scripts/validate_data.py
streamlit run app.py
```

## Repository structure

```text
W07_Dashboard/
├── app.py
├── dashboard/
│   ├── __init__.py
│   ├── charts.py
│   ├── data.py
│   └── ui.py
├── sample_data/
│   ├── README.md
│   ├── data_manifest.json
│   └── *.csv
├── scripts/
│   └── validate_data.py
├── .streamlit/
│   └── config.toml
├── .gitignore
├── launch_dashboard.sh
├── requirements.txt
└── README.md
```

## Persona design

### Engineering Manager

The Fleet Health view opens with precision, recall, F1, AUROC, a fault-method
heatmap, and seed-to-seed stability. The Policy Benchmark view exposes
algorithm and environment controls, convergence curves, final reward by seed,
and stability CV. The Decision Evaluation view emphasizes alpha, agreement,
criterion alignment, and seed-disagreement scenarios.

### Product Manager

The policy workspace compares PPO and SAC across two environments, showing
quality, runtime, sample efficiency, and threshold success. Cross-domain
decision scorecards help distinguish product areas with strong versus weak
judge reliability. Executive KPI cards make the main tradeoffs visible without
requiring notebook-level analysis.

### Customer Success

The Fleet Health view puts a unit-status table and unit selector first. Each
unit shows battery, mission, active alert, severity, telemetry history, and a
recommended next action. Alert examples are synthetic and should be treated as
a demonstration control surface, not a live support queue.

## Data provenance

The dashboard is self-contained and uses compact final outputs supplied from
Weeks 3–6. `sample_data/data_manifest.json` records row counts, SHA-256 hashes,
and descriptions.

Important boundaries:

- All fleet telemetry and fault events are synthetic.
- The Week 6 target labels are analyst-defined test contracts.
- The LLM judge is a secondary rubric, not the authoritative scorer.
- Exact match against `target_action` remains the decision-correctness source
  of truth.
- This dashboard is not a production monitoring, clinical, safeguarding, or
  security-dispatch system.

## Final measured results represented in the dashboard

- Week 4 overall anomaly benchmark: LOF achieved the strongest overall F1 and
  AUROC, with One-Class SVM as a close challenger.
- Week 5: SAC reached higher final rewards than PPO in both single-agent
  environments, but required materially more wall-clock time.
- Week 6: 50 scenarios and 150 seeded judgments produced overall nominal
  Krippendorff alpha of 0.725 and 78% exact three-run agreement.
- Reliability varied by domain, from 0.905 for Aido Rover to 0.366 for Senpai.

## Troubleshooting

### Permission denied

```bash
chmod +x launch_dashboard.sh
./launch_dashboard.sh
```

### Python 3.11 is installed in a nonstandard location

```bash
PYTHON_BIN=/path/to/python3.11 ./launch_dashboard.sh
```

### Port 8501 is already in use

```bash
PORT=8502 ./launch_dashboard.sh
```

### Reset the environment

```bash
rm -rf .venv_w07
./launch_dashboard.sh
```
