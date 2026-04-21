import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    model_dir = os.environ.get("SM_MODEL_DIR")

    print("Loading training data...")
    df = pd.read_csv(os.path.join(train_dir, "train.csv"))

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42)

    print(f"Training samples: {len(X_train):,} | Val samples: {len(X_val):,}")
    print(f"Classes: {len(le.classes_)}")

    model = xgb.XGBClassifier(
        max_depth=int(os.environ.get("max_depth", 6)),
        learning_rate=float(os.environ.get("eta", 0.1)),
        n_estimators=int(os.environ.get("n_estimators", 100)),
        subsample=float(os.environ.get("subsample", 0.8)),
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )

    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {acc:.4f}")

    # save model and encoder
    pickle.dump(model, open(os.path.join(model_dir, "model.pkl"), "wb"))
    pickle.dump(le,    open(os.path.join(model_dir, "encoder.pkl"), "wb"))

    # save metrics
    metrics = {"accuracy": round(acc, 4), "num_classes": len(le.classes_)}
    json.dump(metrics, open(os.path.join(model_dir, "metrics.json"), "w"))
    print("Done.")
