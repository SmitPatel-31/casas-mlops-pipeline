from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import subprocess
import json
import os

def run_dvc_stage(stage):
    result = subprocess.run(
        ["dvc", "repro", stage],
        cwd="/opt/airflow/project",
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"DVC stage '{stage}' failed:\n{result.stderr}")

def run_training():
    result = subprocess.run(
        ["python", "src/train.py"],
        cwd="/opt/airflow/project",
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Training failed:\n{result.stderr}")

def run_evaluate():
    result = subprocess.run(
        ["python", "src/evaluate.py"],
        cwd="/opt/airflow/project",
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Evaluation failed:\n{result.stderr}")

def check_accuracy_gate(**context):
    metrics_path = "/opt/airflow/project/metrics/eval_metrics.json"
    if not os.path.exists(metrics_path):
        return "skip_deploy"
    metrics = json.load(open(metrics_path))
    print(f"Accuracy: {metrics['accuracy']} | Threshold: {metrics['threshold_passed']}")
    if metrics["threshold_passed"]:
        return "deploy"
    return "skip_deploy"

def run_deploy():
    result = subprocess.run(
        ["python", "src/deploy.py"],
        cwd="/opt/airflow/project",
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Deploy failed:\n{result.stderr}")

def skip_deploy():
    print("Accuracy below threshold — skipping deployment.")
    print("Retrain with better params to deploy.")

with DAG(
    dag_id="casas_mlops_pipeline",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "casas", "activity-recognition"],
    description="End-to-end MLOps pipeline for CASAS smart home activity classification"
) as dag:

    ingest = PythonOperator(
        task_id="ingest",
        python_callable=run_dvc_stage,
        op_args=["ingest"]
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=run_dvc_stage,
        op_args=["preprocess"]
    )

    train = PythonOperator(
        task_id="train",
        python_callable=run_training
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=run_evaluate
    )

    accuracy_gate = BranchPythonOperator(
        task_id="accuracy_gate",
        python_callable=check_accuracy_gate,
        provide_context=True
    )

    deploy = PythonOperator(
        task_id="deploy",
        python_callable=run_deploy
    )

    skip = PythonOperator(
        task_id="skip_deploy",
        python_callable=skip_deploy
    )

    ingest >> preprocess >> train >> evaluate >> accuracy_gate >> [deploy, skip]
