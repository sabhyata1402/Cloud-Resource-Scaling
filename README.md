# Intelligent Proactive Resource Forecasting for Cloud-Native Systems

Public deployment repository for the Streamlit dashboard version of the project.

## Project Overview

This project implements and evaluates three ML models (XGBoost, Random Forest, LSTM) for proactive CPU and memory resource forecasting in cloud-native environments. Models are trained on the Alibaba 2018 Cluster Trace and evaluated for cross-cloud generalisation on Azure and Google datasets.

## Repository Structure

```
cloud-resource-forecasting/
├── app.py                        # Streamlit dashboard entry point
├── deployment_utils.py           # Model loading and inference helpers
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Optional container build
├── README.md                     # This file
└── data/
    ├── processed/                # Test splits, feature files, scalers
    ├── models/                   # Saved model artifacts used by the app
    ├── results/                  # JSON / pickle outputs for the dashboard
    └── upload_samples/           # Sample CSV for dashboard demos
```

## Quick Start

### Option 1: Run the dashboard locally

```bash
cd cloud-resource-forecasting
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Launch the Streamlit dashboard
streamlit run app.py
```

### Option 2: Reproduce the training notebook

The original training workflow is preserved in `Resource_Forecasting_Complete_Pipeline.ipynb`. It is not required for deployment, but it documents the full modelling pipeline used to generate the saved artifacts.

### Option 3: Docker

```bash
cd cloud-resource-forecasting
docker build -t h9mlai-forecasting .
docker run -p 8501:8501 h9mlai-forecasting
# Open http://localhost:8501
```

## Streamlit Cloud Deployment

To make the app available to anyone, deploy it from this GitHub repository to Streamlit Community Cloud.

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and sign in with GitHub.
3. Click **Create app**.
4. Select this repository and the `main` branch.
5. Set the entry file to `app.py`.
6. Choose a short app URL such as `cloud-resource-scaling`.

The app will be available at a public `streamlit.app` URL once deployment completes.

## Notebook Execution Order

The original notebook is intended to be run sequentially. Key sections:

| Section | Content | Output |
|---------|---------|--------|
| 1-2 | Setup + Data Loading | Raw datasets loaded |
| 3-4 | EDA + Cleaning | Preprocessed DataFrames |
| 5-6 | Feature Engineering + Selection | Selected feature lists |
| 7 | Temporal Train/Val/Test Split | No data leakage |
| 8 | Model Training (XGBoost, RF, LSTM) | Saved models in `data/models/` |
| 9 | Evaluation + Bootstrap CI | Results tables, confidence intervals |
| 10-11 | Multi-Horizon + Statistical Tests | Wilcoxon tests |
| 12-13 | Cross-Cloud + Transfer Learning | Generalisation results |
| 14 | SHAP Explainability | SHAP plots + saved values |
| 15 | Cost-Benefit Analysis (CPU + Memory) | EUR/day savings |
| 16 | Error Analysis + SLA Breach AUC-ROC | Classification metrics |
| 17 | Learning Curves | Data efficiency plots |
| 18 | Final Results Persistence | `all_results.json` |

## Models

| Model | Type | Parameters |
|-------|------|-----------|
| XGBoost | Gradient Boosted Trees | n_estimators=200, max_depth=5, lr=0.05 |
| Random Forest | Bagged Decision Trees | n_estimators=100, max_depth=12 |
| LSTM | Deep Learning (2-layer) | hidden=64, dropout=0.2, seq_len=6 |

## Datasets

| Dataset | Source | Role |
|---------|--------|------|
| Alibaba 2018 Cluster Trace | github.com/alibaba/clusterdata | Primary training |
| Azure Public Dataset | github.com/Azure/AzurePublicDataset | Cross-cloud validation |
| Google Cluster Traces v3 | research.google/tools/datasets | Cross-cloud testing |

## Evaluation Metrics

- **Accuracy:** RMSE, MAE, MAPE with 95% bootstrap confidence intervals
- **Efficiency:** Training time (s), inference latency (ms), throughput (pred/s), model size (MB)
- **Statistical:** Wilcoxon signed-rank tests for pairwise model comparisons
- **Classification:** Precision, Recall, F1, AUC-ROC for SLA breach detection
- **Cost:** EUR/day savings from proactive vs. reactive scaling (CPU + Memory)

## Streamlit Dashboard

The dashboard (`app.py`) provides 6 interactive tabs:

1. **Live Forecast** - Run batch predictions with interactive Plotly charts
2. **Model Results** - RMSE comparison, efficiency metrics, multi-horizon analysis
3. **Cost Analysis** - Proactive vs. reactive scaling savings with confidence intervals
4. **Cross-Cloud** - Generalisation heatmaps and transfer learning results
5. **Explainability** - SHAP feature importance and dependence plots
6. **About** - Project summary and setup instructions

## Reproducibility

- Fixed random seed (`SEED = 42`) throughout
- Temporal train/val/test split (no leakage)
- All model artifacts saved to `data/models/`
- All results persisted to `data/results/all_results.json`
- Docker container for environment reproducibility
- `requirements.txt` with pinned minimum versions
- This repository is intended to stay GitHub-friendly: the raw 13 GB telemetry and large training splits are excluded, while the dashboard-required test splits, model files, results, and sample inputs are kept.

## Data Note

The raw public cloud trace files are large and are not committed to this repository. The dashboard uses the saved processed artifacts and model files under `data/` for inference and visualisation.
