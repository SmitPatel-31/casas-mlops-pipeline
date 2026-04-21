import boto3
import sagemaker
import yaml
import pathlib
from sagemaker.sklearn.estimator import SKLearn

ROLE    = "arn:aws:iam::711387128125:role/SageMakerExecutionRole"
BUCKET  = "casas-mlops-smit"
REGION  = "us-east-1"
PREFIX  = "casas"

params  = yaml.safe_load(open("params.yaml"))
xgb_params = params["xgboost"]

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

def upload_data():
    s3 = boto3.client("s3", region_name=REGION)
    print("Uploading train split to S3...")
    s3.upload_file("data/splits/train.csv",
                   BUCKET, f"{PREFIX}/splits/train.csv")
    print(f"Uploaded → s3://{BUCKET}/{PREFIX}/splits/train.csv")
    return f"s3://{BUCKET}/{PREFIX}/splits/"

def run_training(s3_input):
    estimator = SKLearn(
        entry_point="train_entry.py",
        source_dir="src",
        role=ROLE,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="1.2-1",
        py_version="py3",
        hyperparameters={
            "max_depth":    xgb_params["max_depth"],
            "eta":          xgb_params["eta"],
            "n_estimators": xgb_params["n_estimators"],
            "subsample":    xgb_params["subsample"],
        },
        output_path=f"s3://{BUCKET}/{PREFIX}/models/",
        base_job_name="casas-activity-classifier",
        sagemaker_session=session
    )

    train_input = sagemaker.inputs.TrainingInput(
        s3_input, content_type="text/csv"
    )

    print("Submitting SageMaker training job...")
    estimator.fit({"train": train_input}, wait=True, logs=True)

    job_name = estimator.latest_training_job.name
    model_uri = f"s3://{BUCKET}/{PREFIX}/models/{job_name}/output/model.tar.gz"
    print(f"Training complete.")
    print(f"Job name: {job_name}")
    print(f"Model URI: {model_uri}")

    # save job name for evaluate.py
    with open("models/latest_job.txt", "w") as f:
        f.write(job_name + "\n")
        f.write(model_uri + "\n")

if __name__ == "__main__":
    pathlib.Path("models").mkdir(exist_ok=True)
    s3_input = upload_data()
    run_training(s3_input)
