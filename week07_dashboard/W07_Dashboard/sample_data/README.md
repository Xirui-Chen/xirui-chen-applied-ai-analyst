# Sample Data

This folder contains compact, public-safe copies and derivatives of the final
Weeks 3–6 outputs used by the Streamlit dashboard.

- Week 3 contributes five-unit synthetic telemetry, sampled every five minutes.
- Week 4 contributes anomaly benchmark metrics and synthetic fault examples.
- Week 5 contributes PPO/SAC learning curves, seed results, runtime summaries,
  and multi-agent diagnostics.
- Week 6 contributes the 50-scenario decision scorecard, 150 seeded judge
  results, Krippendorff alpha, and pairwise Cohen kappa.

`fleet_unit_status.csv` is a dashboard demonstration snapshot. It combines the
latest Week 3 synthetic telemetry values with selected Week 4 synthetic fault
examples so that the customer-success view has unit-level alert states.

No real InGen fleet, customer, security, patient, or student data is included.
See `data_manifest.json` for row counts, hashes, and file-level provenance.
