# CASAS Smart Home Activity Recognition — MLOps Pipeline

An end-to-end MLOps pipeline for real-time human activity classification using the [UCI CASAS Smart Home dataset](https://casas.wsu.edu/datasets/). Built with DVC, AWS SageMaker, Apache Airflow, and Grafana — demonstrating a production-grade ML lifecycle from raw sensor data to a monitored inference endpoint.

---

## Architecture

```
Raw Sensor Data (CASAS CSV)
        │
        ▼
  DVC Pipeline (dvc.yaml)
        │
        ├── ingest.py        → Download & validate raw sensor files
        ├── preprocess.py    → Sliding window feature extraction
        ├── train.py         → Submit XGBoost job to AWS SageMaker
        ├── evaluate.py      → Pull model artifact, compute metrics
        └── deploy.py        → Register & deploy SageMaker endpoint
        │
        ▼
  Apache Airflow DAG (Daily orchestration + accuracy gate)
        │
        ▼
  SageMaker Real-time Endpoint
        │
        ▼
  Grafana Dashboard (CloudWatch metrics monitoring)
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data versioning & pipeline | DVC + AWS S3 remote |
| Feature engineering | Python, Pandas |
| Model training | AWS SageMaker Training Jobs (ml.m5.large) |
| Model registry | SageMaker Model Registry |
| Classifier | XGBoost (multi-class, 40 activity classes) |
| Orchestration | Apache Airflow |
| Inference | SageMaker Real-time Endpoint (ml.t2.medium) |
| Monitoring | Grafana + AWS CloudWatch |
| Infrastructure | AWS (S3, IAM, CloudWatch) |

---

## Dataset

**UCI CASAS Smart Home Dataset** — timestamped ambient sensor events collected from real residential homes.

- **Source**: 6 labeled homes (`rw101`–`rw107`) from the CASAS Zenodo archive
- **Raw events**: ~792K sensor events after activity label forward-fill
- **Sliding windows**: 218K windows (60-second window, 30-second step)
- **Activity classes**: 40 (Sleep, Cook, Eat, Work, Relax, Personal_Hygiene, Watch_TV, etc.)
- **Features per window**: 28 (per-sensor event counts, hour of day, total events, unique sensors)

---

## Pipeline Stages

### Ingest
Downloads the labeled CASAS dataset from Zenodo and extracts CSV files to `data/raw/labeled/`. Output tracked by DVC.

### Preprocess
Parses sensor events, forward-fills activity labels, and builds sliding time windows. Each window becomes one training sample with a flat feature vector. Outputs versioned `features.csv` and train/val/test splits (70/15/15).

### Train
Uploads the training split to S3 and submits an XGBoost training job to AWS SageMaker. `train_entry.py` runs inside the SageMaker sklearn container on `ml.m5.large`. Hyperparameters controlled via `params.yaml`.

### Evaluate
Downloads the trained model artifact from S3, runs inference on the test split, and writes `metrics/eval_metrics.json`. Gates deployment on a 70% accuracy threshold.

### Deploy
Registers the approved model in SageMaker Model Registry and deploys it to a real-time inference endpoint. Skips deployment automatically if the accuracy threshold is not met.

---

## Experiment Tracking

Two versions trained and tracked with DVC:

```
Path                       Metric    v1      v2      Change
metrics/eval_metrics.json  accuracy  0.5961  0.6190  +0.0229
```

| | v1 (baseline) | v2 (tuned) |
|---|---|---|
| Window size | 30s | 60s |
| Step size | 10s | 30s |
| max_depth | 6 | 8 |
| n_estimators | 100 | 200 |
| Learning rate (eta) | 0.1 | 0.05 |
| Test accuracy | 59.61% | 61.90% |

Switching between versions:
```bash
git checkout v1 && dvc checkout   # restore v1 data + model
git checkout v2 && dvc checkout   # restore v2 data + model
```

---

## Airflow DAG

Daily orchestration pipeline with an accuracy gate:

```
ingest → preprocess → train → evaluate → accuracy_gate → deploy
                                                       └────────→ skip_deploy
```

The `accuracy_gate` is a `BranchPythonOperator` that reads `eval_metrics.json` and conditionally triggers deployment only when the accuracy threshold is met.

---

## Monitoring

After deployment the SageMaker endpoint emits metrics to CloudWatch automatically. The Grafana dashboard visualizes:

- Inference latency (p50, p95, p99)
- Prediction class distribution over time (drift proxy)
- Endpoint error rate (4XX + 5XX)
- Airflow DAG run history

---

## Getting Started

### Prerequisites

- Python 3.10+
- AWS account with configured credentials (`aws configure`)
- Docker (for Airflow and Grafana)
- Mac users: `brew install libomp` (required for XGBoost)

### Setup

```bash
git clone https://github.com/smitpatel-31/casas-mlops-pipeline.git
cd casas-mlops-pipeline

conda create -n casas-mlops python=3.10 -y
conda activate casas-mlops
pip install "dvc[s3]" sagemaker==2.232.2 boto3 pandas numpy scikit-learn xgboost pyyaml

dvc pull
```

### Run the pipeline

```bash
# full pipeline
dvc repro

# individual stages
python src/ingest.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/deploy.py
```

### Compare versions

```bash
dvc metrics diff v1 v2
```

### Start Airflow

```bash
docker compose up airflow
```

### Start Grafana

```bash
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

Open `http://localhost:3000` and import `monitoring/grafana_dashboard.json`.

---

## Project Structure

```
casas-mlops-pipeline/
├── .dvc/                          # DVC config + S3 remote
├── data/
│   ├── raw/labeled/               # CASAS CSV files (DVC tracked)
│   ├── processed/features.csv     # Feature windows (DVC tracked)
│   └── splits/                    # Train / val / test splits
├── src/
│   ├── ingest.py                  # Data download + validation
│   ├── preprocess.py              # Sliding window feature extraction
│   ├── train.py                   # SageMaker job launcher
│   ├── train_entry.py             # XGBoost script (runs on AWS)
│   ├── evaluate.py                # Evaluation + metrics gate
│   ├── deploy.py                  # SageMaker endpoint deployment
│   └── requirements.txt           # SageMaker container dependencies
├── airflow/
│   └── dags/
│       └── casas_pipeline_dag.py  # Airflow DAG
├── monitoring/
│   └── grafana_dashboard.json     # Importable Grafana dashboard
├── metrics/
│   └── eval_metrics.json          # DVC-tracked evaluation metrics
├── models/
│   ├── latest_job.txt             # SageMaker job name + model URI
│   └── endpoint.txt               # Deployed endpoint name
├── dvc.yaml                       # Pipeline stage definitions
├── params.yaml                    # Hyperparameters (DVC tracked)
└── README.md
```

---

## Cost

| Resource | Instance | Approx. cost |
|---|---|---|
| SageMaker Training | ml.m5.large | ~$0.02 per run |
| SageMaker Endpoint | ml.t2.medium | ~$0.056/hr |
| S3 Storage | — | <$0.01 |

Delete the endpoint when not in use:

```bash
aws sagemaker delete-endpoint --endpoint-name casas-activity-endpoint
```

---

## Citation

Cook, D., Crandall, A., Thomas, B., & Krishnan, N. (2013). CASAS: A smart home in a box. *IEEE Computer*, 46(7):62-69. https://doi.org/10.1109/MC.2012.328

---

## Author

**Smit Patel** — MS Software Engineering Systems, Northeastern University
[linkedin.com/in/smitpatel3107](https://linkedin.com/in/smitpatel3107) · [github.com/smitpatel-31](https://github.com/smitpatel-31)