# DriftPCA

**DriftPCA** is a method providing interpretable 2D visualizations of concept drift by inspecting changes on the level of data, model predictions and model explanations.

### Requirements

Tested on python 3.11.0

```
pip install -r "requirements.txt"
```

### Stream Generator Datasets Generation

```bash
python scripts/create_datasets.py
```

### Synthetic Data Generation

Define synthetic data properties in [synthetic_data.json](synthetic_data.json)

```bash
python scripts/generate_all_synthetic.py
```

### Visualization Generation

Define datasets to be visualized in [scripts/run_all_datasets.py](scripts/run_all_datasets.py)

```bash
python scripts/run_all_datasets.py --results_dir "results/synth_datasets"
```

### Synthetic Test App

```bash
python -m streamlit run app/main.py
```bash