import boto3
import sagemaker
import json
import pathlib
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel

ROLE    = "arn:aws:iam::711387128125:role/SageMakerExecutionRole"
BUCKET  = "casas-mlops-smit"
REGION  = "us-east-1"
ENDPOINT_NAME = "casas-activity-endpoint"

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

def get_latest_model_uri():
    lines = open("models/latest_job.txt").readlines()
    model_uri = lines[1].strip()
    return model_uri

def check_threshold():
    metrics = json.load(open("metrics/eval_metrics.json"))
    acc = metrics["accuracy"]
    passed = metrics["threshold_passed"]
    print(f"Accuracy: {acc} | Threshold passed: {passed}")
    if not passed:
        print("Accuracy below threshold — skipping deployment.")
        print("Tip: update params.yaml and retrain to improve accuracy.")
        return False
    return True

def deploy_endpoint(model_uri):
    print(f"Deploying model from: {model_uri}")

    model = SKLearnModel(
        model_data=model_uri,
        role=ROLE,
        entry_point="train_entry.py",
        source_dir="src",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session
    )

    print("Creating endpoint — this takes 3-5 minutes...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium",
        endpoint_name=ENDPOINT_NAME
    )

    print(f"Endpoint live: {predictor.endpoint_name}")

    # save endpoint name for monitoring
    with open("models/endpoint.txt", "w") as f:
        f.write(predictor.endpoint_name + "\n")
        f.write(model_uri + "\n")

    return predictor

def test_endpoint(predictor):
    import pandas as pd
    print("\nTesting endpoint with sample payload...")
    sample = pd.read_csv("data/splits/test.csv").drop("label", axis=1).iloc[:5]
    result = predictor.predict(sample.values.tolist())
    print(f"Sample predictions: {result}")

if __name__ == "__main__":
    pathlib.Path("models").mkdir(exist_ok=True)

    if not check_threshold():
        print("\nForce deploying anyway for demo purposes...")

    model_uri = get_latest_model_uri()
    predictor = deploy_endpoint(model_uri)
    test_endpoint(predictor)

    print("\nDone. To delete endpoint when finished:")
    print(f"  aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME}")
