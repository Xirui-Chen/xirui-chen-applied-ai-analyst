# Xirui Chen Applied AI Analyst Internship

## Project Overview

This repository contains the public, reproducible deliverables for my 8-week Applied AI Analyst internship program with InGen Dynamics.

The program focuses on physical AI and embodied intelligence, with workstreams across robotics landscape research, competitive intelligence, synthetic robot telemetry, anomaly detection, reinforcement learning benchmarks, adaptive decision-model evaluation, and a final decision-support dashboard.

The project is anchored around InGen Dynamics’ public product portfolio and the Origami / PIC 2.0 physical intelligence framing. The main product anchors referenced throughout the program are:

* Fari, eldercare companion
* Senpai, educational robot
* Sentinel Prime AI, security and monitoring
* Aido Rover, outdoor patrol and inspection
* Aido Humanoid, bipedal research platform
* Origami / PIC 2.0, physical intelligence platform

All work in this repository is based on public information, open-source tooling, synthetic data, and reproducible experiments.

## Repository Structure

```text
├── README.md
├── requirements.txt
├── week01_landscape/
│   ├── W01_PhysicalAI_Landscape_Brief.md
│   ├── W01_PIC20_Conceptual_Map.md
│   └── W01_env_check.ipynb
├── week02_competitive/
│   ├── W02_Competitive_Matrix.xlsx
│   └── W02_Strategic_Gap_Memo.md
├── week03_telemetry/
│   ├── W03_Telemetry_Generator.py
│   ├── W03_Telemetry_EDA.ipynb
│   └── W03_Feature_Dictionary.md
├── week04_anomaly/
│   ├── W04_Anomaly_Benchmark.ipynb
│   ├── W04_Method_Recommendation_Memo.md
│   └── W04_MidPoint_Deck.pptx
├── week05_rl/
│   ├── W05_RL_Benchmarks/
│   └── W05_RL_Analyst_Memo.md
├── week06_decision_eval/
│   ├── W06_Decision_Eval_Harness/
│   └── W06_Eval_Methodology.md
├── week07_dashboard/
│   ├── W07_Dashboard/
│   ├── W07_Dashboard_Design_Doc.md
│   └── W07_Dashboard_Walkthrough.md
├── week08_capstone/
│   ├── W08_Capstone_Report.docx
│   ├── W08_Capstone_Deck.pptx
│   └── W08_Capstone_Presentation.md
├── weekly/
│   ├── Wk-01-Recap.md
│   ├── Wk-02-Recap.md
│   ├── Wk-03-Recap.md
│   ├── Wk-04-Recap.md
│   ├── Wk-05-Recap.md
│   ├── Wk-06-Recap.md
│   ├── Wk-07-Recap.md
│   └── Wk-08-Final-Recap.md
├── docs/
│   └── Internship_Plan.pdf
└── data/
```

## Environment

The project is designed for Python 3.11.

Core tools include:

* pandas
* NumPy
* scipy
* statsmodels
* scikit-learn
* PyTorch
* Gymnasium
* Stable-Baselines3
* PettingZoo
* PyOD
* plotly
* Streamlit
* PyYAML
* Hugging Face transformers

The full dependency list is maintained in `requirements.txt`.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/<your-username>/xirui-chen-applied-ai-analyst.git
cd xirui-chen-applied-ai-analyst
```

Create and activate a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment:

```bash
jupyter lab
```

Then open and run:

```text
week01_landscape/W01_env_check.ipynb
```

The environment check notebook imports the required packages and confirms that the local toolchain is ready for the Week 1 to Week 8 workflow.

## Reproduction Path

The intended reproduction path is:

```bash
git clone https://github.com/<your-username>/xirui-chen-applied-ai-analyst.git
cd xirui-chen-applied-ai-analyst
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

From JupyterLab, run the notebooks in the weekly folders in order.

As the project develops, each quantitative week will include its own reproduction instructions, seeds, configuration notes, and output locations.
