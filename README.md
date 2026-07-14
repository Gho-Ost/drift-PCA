# DriftPCA: Interpretable 2D Concept Drift Visualizations

**DriftPCA** is a python-based methodology and codebase designed to provide highly interpretable 2D visualizations of concept drift. It inspects changes across three fundamental levels:
1. **Data Level**: Visualizes distributions of pre-drift (reference) and post-drift (current) samples.
2. **Model Predictions**: Projects and draws decision boundaries of classification models (trained exclusively on pre-drift data) evaluated on post-drift data, highlighting misclassified points.
3. **Model Explanations**: Integrates model feature importance into the feature loadings visualization.

---

## Methodology: Drift Component Analysis (DCA)

Traditional Principal Component Analysis (PCA) maps data along directions of maximum variance. However, when studying concept drift, standard PCA might focus on static variation in the dataset rather than the shift itself.

**Drift Component Analysis (DCA)** solves this by executing Singular Value Decomposition (SVD) on the **difference vectors** representing concept drift.

### Mathematical Intuition & Algorithm
1. **Group by Class**: For each class $c$ present in both the pre-drift (reference) and post-drift (current) datasets:
   - Compute the class-wise mean vector: $\mu_{c,\text{pre}}$ and $\mu_{c,\text{post}}$
   - Compute the class-wise standard deviation vector: $\sigma_{c,\text{pre}}$ and $\sigma_{c,\text{post}}$
2. **Compute Drift Differences**:
   - $\Delta \mu_c = \mu_{c,\text{post}} - \mu_{c,\text{pre}}$
   - $\Delta \sigma_c = \sigma_{c,\text{post}} - \sigma_{c,\text{pre}}$
3. **Construct the Drift Difference Matrix**:
   - Collect these vectors into a matrix $\mathbf{D} = [\Delta \mu_1, \Delta \sigma_1, \Delta \mu_2, \Delta \sigma_2, \dots]^T$ representing the dimensions of drift.
4. **Perform SVD**:
   - Execute Singular Value Decomposition on $\mathbf{D}$ to find the main component axes of concept drift.
5. **Project Samples**:
   - Project the high-dimensional pre-drift and post-drift samples onto the first two SVD components to visualize the dataset in a 2D space where concept drift is maximized.

### Interpretability Components
* **Biplot / Loadings Compass Rose**: Represents how individual features contribute to the first two drift components. Loading arrows can be scaled by the singular values to indicate the absolute magnitude of drift. Furthermore, they can be color-coded by model feature importance.
* **Drift Compass Rose**: Represents the projected mean and standard deviation difference vectors, showing the direction and magnitude of drift for each class in the 2D visualization space.

---

## Repository Structure

```
drift-PCA/
├── analysis_methods/            # Core methodology implementations
│   ├── drift_component_analysis.py  # DCA SVD mapping class
│   ├── dca_utils.py                 # Scatter plots and compass rose drawing
│   └── algorithm_comparison.py      # Fitter wrappers and plotters for benchmark methods
├── scripts/                     # Executable scripts
│   ├── dataset_generation/      # Data generation and preprocessing scripts
│   │   ├── create_generator_datasets.py # Stream-based generators (RBF, Hyp, Tree)
│   │   ├── generate_synthetic_drift.py  # Merged script to generate parameterized synthetic data (aggregate or class-specific)
│   │   ├── generate_all_synthetic.py    # Batch script generating all synthetic datasets
│   │   ├── create_real_datasets.py      # Real data preprocessing (Sliding Window & KS-Test)
│   │   └── generate_thu_splits.py       # Splits the THU concept drift stream datasets
│   ├── drift_detection/         # Sliding window drift search scripts
│   ├── run_experiments/         # Benchmarking and scenario testing scripts
│   │   ├── run_synthetic_scenarios_experiments.py # Batch runs DCA on aggregate & class-specific scenarios
│   │   ├── run_algorithm_comparison.py  # Compares DCA vs PCA vs UMAP vs SSNP
│   │   ├── run_dca_experiments.py       # Batch runs DCA on all generator datasets
│   │   ├── run_gradrec_experiments.py   # Batch runs DCA on gradual/recurrent THU datasets
│   │   ├── run_rbf_flexibility_scenarios.py # Runs RBF scenarios
│   │   ├── run_real_experiments.py      # Batch runs DCA on real-world datasets
│   │   └── run_speed_test.py            # Computational speed benchmarks
│   └── run_dca.py               # Main command-line script to run DCA on a dataset
├── notebooks/                   # Jupyter tutorial notebooks
│   ├── DriftPCA_Example.ipynb       # DriftPCA Simple Tutorial
│   └── PCA_Decision_Boundary.ipynb  # PCA Decision Boundary Drawing
├── data/                        # Datasets folder (populated by scripts)
│   ├── real/                    # Real-world datasets
│   └── synthetic/               # Generated synthetic datasets (agg/, class/, gen/, gradrec/)
├── results/                     # Output visualizations and reports (populated by scripts)
├── ssnp/                        # Semi-Supervised Neural Projection tool (benchmark comparison) (https://github.com/mespadoto/ssnp)
├── THU-Concept-Drift-Datasets-v1.0/ # THU Concept Drift Datasets generator tool (https://github.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0)
├── requirements.txt             # Project requirements
└── README.md                    # Project documentation
```

---

## Requirements & Installation

This project has been tested on **Python 3.11.0**.

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **External Benchmarking & Generation Tools**:
   This repository utilizes two separate external open-source projects for algorithm comparison and dataset generation. Make sure they are cloned into their respective directories:
   * **SSNP (Semi-Supervised Neural Projection)**: Cloned into the `ssnp/` directory.
     * URL: [https://github.com/mespadoto/ssnp](https://github.com/mespadoto/ssnp)
     * *Note: Requires `tensorflow` (defined in `requirements.txt`).*
   * **THU-Concept-Drift-Datasets-v1.0**: Cloned into the `THU-Concept-Drift-Datasets-v1.0/` directory.
     * URL: [https://github.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0](https://github.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0)
     * *Note: Used for generating gradual and recurrent rotation splits.*

---

## Running Dataset Generation

Before running DCA or experiments, generate the datasets:

### 1. Unified Parameterized Synthetic Generation (Aggregate & Class-specific)
To generate the complete batch of synthetic drift scenarios:
```bash
python scripts/dataset_generation/generate_all_synthetic.py
```
This runs the unified generator script (`generate_synthetic_drift.py`) twice to batch-generate from both aggregate configurations (outputting to `data/synthetic/agg/`) and class-specific configurations (outputting to `data/synthetic/class/`).

You can also run individual configurations via the CLI:
* **Batch generation using JSON files**:
  ```bash
  # Class-Aggregate definitions
  python scripts/dataset_generation/generate_synthetic_drift.py --json data/synthetic_data.json
  
  # Class-Specific definitions
  python scripts/dataset_generation/generate_synthetic_drift.py --json data/synthetic_data_classes.json
  ```
* **Single Scenario via Command-Line Flags**:
  ```bash
  # Class-Aggregate Mode
  python scripts/dataset_generation/generate_synthetic_drift.py \
    --name my_agg_scenario \
    --means_pre 0 0 \
    --stds_pre 1 1 \
    --means_post 1 0 \
    --stds_post 1 1
    
  # Class-Specific Mode
  python scripts/dataset_generation/generate_synthetic_drift.py \
    --name my_class_scenario \
    --means_pre_c0 0 0 --stds_pre_c0 1 1 \
    --means_pre_c1 0 0 --stds_pre_c1 1 1 \
    --means_post_c0 0 0 --stds_post_c0 1 1 \
    --means_post_c1 2 0 --stds_post_c1 1 1
  ```

### 2. Synthetic Stream Generation (River Generators)
Generates stream datasets using standard generator classes from `river.datasets.synth`:
```bash
python scripts/dataset_generation/create_generator_datasets.py
```

### 3. THU Dataset Splits
Generates gradual and recurrent concept drift dataset splits from the THU generator:
```bash
python scripts/dataset_generation/generate_thu_splits.py
```

### 4. Real-world Dataset Windows
Applies a sliding-window Kolmogorov-Smirnov (KS) test to divide real-world streams (e.g. Elec, NOAA) into pre-drift and post-drift pairs:
```bash
python scripts/dataset_generation/create_real_datasets.py
```

---

## Running Drift Component Analysis (DCA)

The `scripts/run_dca.py` script is the main entry point to visualize concept drift on any dataset.

### Command-Line Arguments
* `--data_dir`: Directory containing the dataset CSV files (default: `data\synthetic\gen`).
* `--results_dir`: Directory where the output visualization is saved (default: `results\synthetic\generator`).
* `--dataset`: Name prefix of the dataset (will load `{dataset}_pre.csv` and `{dataset}_post.csv`).
* `--pre_file`: Explicit path to a pre-drift CSV file (overrides `--dataset`).
* `--post_file`: Explicit path to a post-drift CSV file (overrides `--dataset`).
* `--model`: Classifier to train on pre-drift data for drawing decision boundaries (`svc` or `rf`, default: `svc`).
* `--no_boundary`: Disable decision boundary plotting.
* `--discrete_boundary`: Draw hard class boundaries instead of smooth probabilities.
* `--drift_mode`: Drift calculation: `data` (ignores class labels), `global` (global drift vectors), or `per-class` (computes drift per-class, default: `per-class`).
* `--no_target`: Treats all columns as features (unsupervised DCA).
* `--unscaled_loadings`: Disable loading arrow scaling by singular values.
* `--color_scheme`: Point color scheme: `class` (default) or `drift`.
* `--highlight_misclassifications`: Highlight post-drift points misclassified by the pre-drift model.
* `--feature_importance`: Highlight feature importances on the loadings rose.
* `--drift_type`: Change visualization mode for points: `sudden` (default) or `gradual` (gradient alpha).

### Runnable Examples

#### Example 1: Standard DCA on River RBF stream
Runs DCA using SVM boundary, per-class drift calculation, highlighting misclassifications, and computing feature importance:
```bash
python scripts/run_dca.py \
  --dataset rbf \
  --data_dir data/synthetic/gen \
  --results_dir results/synthetic/rbf_dca \
  --model svc \
  --drift_mode per-class \
  --highlight_misclassifications \
  --feature_importance
```

#### Example 2: DCA without Target/Classes (Unsupervised Mode)
```bash
python scripts/run_dca.py \
  --dataset hyp \
  --data_dir data/synthetic/gen \
  --results_dir results/synthetic/hyp_unsupervised \
  --no_target \
  --drift_mode data
```

---

## Running the Showcase Application

We provide an interactive **Streamlit Showcase Application** that allows users to:
1. Load pre/post drift datasets from any folder (scanning and rendering dataset files dynamically).
2. Configure all DCA parameter choices on the fly (model, mode, boundary grids, misclassification highlights, feature importances, etc.) and view the updated 2D mappings instantly.
3. Configure and generate custom class-specific synthetic drift datasets with specific feature-level means and standard deviations from a visual form interface.

To start the application:
```bash
python -m streamlit run app/main.py
```

---

## Running Benchmarks and Experiments

### 1. Individual Experiment Suites
You can run specific test suites individually:
* **Synthetic Scenarios batch**:
  ```bash
  python scripts/run_experiments/run_synthetic_scenarios_experiments.py
  ```
* **Algorithm Comparison**:
  ```bash
  python scripts/run_experiments/run_algorithm_comparison.py \
  --dataset noaa_drift2 \
    --data_dir data/real/comprehensive \
    --results_dir results/comparisons
  ```
* **Stream Generators batch**:
  ```bash
  python scripts/run_experiments/run_dca_experiments.py
  ```
* **THU splits batch**:
  ```bash
  python scripts/run_experiments/run_gradrec_experiments.py
  ```
* **Real benchmark datasets batch**:
  ```bash
  python scripts/run_experiments/run_real_experiments.py
  ```
* **Execution Speed benchmark**:
  ```bash
  python scripts/run_experiments/run_speed_test.py
  ```