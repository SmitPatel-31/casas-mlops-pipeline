import boto3
import pickle
import json
import tarfile
import pathlib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

BUCKET = "casas-mlops-smit"
REGION = "us-east-1"

def get_latest_job():
    lines = open("models/latest_job.txt").readlines()
    job_name  = lines[0].strip()
    model_uri = lines[1].strip()
    return job_name, model_uri

def download_model(model_uri):
    pathlib.Path("/tmp/casas_model").mkdir(exist_ok=True)
    s3 = boto3.client("s3", region_name=REGION)
    
    # parse bucket and key from URI
    uri = model_uri.replace("s3://", "")
    bucket = uri.split("/")[0]
    key    = "/".join(uri.split("/")[1:])
    
    print(f"Downloading model from S3...")
    s3.download_file(bucket, key, "/tmp/casas_model/model.tar.gz")
    
    with tarfile.open("/tmp/casas_model/model.tar.gz") as t:
        t.extractall("/tmp/casas_model/")
    print("Model extracted.")

def evaluate():
    model   = pickle.load(open("/tmp/casas_model/model.pkl", "rb"))
    encoder = pickle.load(open("/tmp/casas_model/encoder.pkl", "rb"))

    df    = pd.read_csv("data/splits/test.csv")
    X     = df.drop("label", axis=1)
    y_raw = df["label"]

    # encode labels
    y = encoder.transform(y_raw)
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y, preds,
          target_names=encoder.classes_, zero_division=0))

    pathlib.Path("metrics").mkdir(exist_ok=True)
    metrics = {
        "accuracy": round(acc, 4),
        "threshold_passed": acc >= 0.70
    }
    json.dump(metrics, open("metrics/eval_metrics.json", "w"))
    print(f"\nMetrics saved → metrics/eval_metrics.json")
    print(f"Threshold passed: {metrics['threshold_passed']}")

if __name__ == "__main__":
    job_name, model_uri = get_latest_job()
    print(f"Evaluating job: {job_name}")
    download_model(model_uri)
    evaluate()
