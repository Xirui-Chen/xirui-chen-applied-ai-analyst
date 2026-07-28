# Scenario Bank Contract

This folder contains 50 individual YAML scenarios plus:

- `scenario_schema.yaml`: validation contract
- `action_taxonomy.yaml`: analyst-defined action labels
- `index.yaml`: expected files and domain counts

Every scenario includes structured `inputs`, scenario-specific `allowed_actions`, an authoritative `target_action`, and a short `target_rationale`.

To extend the bank, copy a scenario in the same domain, assign a unique ID, use labels from `action_taxonomy.yaml`, include the target in the allowed actions, then run:

```bash
python scorer/validate_scenarios.py
```

Update the expected counts in `config.yaml` and `index.yaml` when adding scenarios.
