# surgery-duration-estimator

Code for a surgery duration estimation AI, built on VitalDB clinical and lab data.

## Contents

- `preprocess.py` — data preprocessing pipeline
- `clinical_data.csv`, `clinical_parameters.csv` — clinical inputs
- `lab_data.csv`, `lab_parameters.csv` — lab inputs
- `processed/` — feature parquet, encoders, splits, schema, decisions log
- `models/v1/` — trained model artifacts
