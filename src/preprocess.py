import pandas as pd
import numpy as np
import pathlib
import yaml
import glob

RAW_DIR = pathlib.Path("data/raw/labeled")
PROCESSED_DIR = pathlib.Path("data/processed")
SPLITS_DIR = pathlib.Path("data/splits")

params = yaml.safe_load(open("params.yaml"))
WINDOW_SIZE = params["window_size"]
STEP_SIZE = params["step_size"]

# Use only these well-labeled files to keep it fast and clean
TARGET_FILES = ["rw101", "rw103", "rw104", "rw105", "rw106", "rw107"]

def load_files():
    dfs = []
    for name in TARGET_FILES:
        filepath = RAW_DIR / f"{name}.csv"
        if not filepath.exists():
            continue
        df = pd.read_csv(filepath, header=None,
            names=["date", "time", "sensor", "value", "activity"],
            on_bad_lines="skip")
        df["source"] = name
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} events from {len(dfs)} files")
    return combined

def parse_activity(df):
    def clean(val):
        if pd.isna(val):
            return None
        val = str(val).strip()
        if "=" in val:
            val = val.split("=")[0]
        return val.strip('"').strip()
    df["activity"] = df["activity"].apply(clean)
    return df

def forward_fill_activity(df):
    df = df.sort_values(["source", "date", "time"]).reset_index(drop=True)
    df["activity"] = df.groupby("source")["activity"].ffill()
    df = df.dropna(subset=["activity"])
    return df

def make_windows(df, all_sensors):
    rows = []
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="mixed", dayfirst=False
    )
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    window_td = pd.Timedelta(seconds=WINDOW_SIZE)
    step_td   = pd.Timedelta(seconds=STEP_SIZE)

    # vectorized approach — much faster than row-by-row
    sensor_dummies = pd.get_dummies(df["sensor"]).reindex(
        columns=all_sensors, fill_value=0)
    df = pd.concat([df.reset_index(drop=True),
                    sensor_dummies.reset_index(drop=True)], axis=1)

    start = df["timestamp"].iloc[0]
    end   = df["timestamp"].iloc[-1]
    t = start
    total = int((end - start) / step_td)
    done  = 0

    while t + window_td <= end:
        mask = (df["timestamp"] >= t) & (df["timestamp"] < t + window_td)
        w = df[mask]
        if len(w) >= 2:
            feats = {}
            for s in all_sensors:
                feats[f"cnt_{s}"] = int(w[s].sum())
            feats["hour"]          = t.hour
            feats["total_events"]  = len(w)
            feats["unique_sensors"] = w["sensor"].nunique()
            feats["label"]         = w["activity"].mode()[0]
            rows.append(feats)
        t += step_td
        done += 1
        if done % 500 == 0:
            pct = done / total * 100
            print(f"\r  Progress: {pct:.1f}% ({len(rows):,} windows)", end="", flush=True)

    print()
    return pd.DataFrame(rows)

def save_splits(features):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    shuffled = features.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    shuffled[:train_end].to_csv(SPLITS_DIR / "train.csv", index=False)
    shuffled[train_end:val_end].to_csv(SPLITS_DIR / "val.csv", index=False)
    shuffled[val_end:].to_csv(SPLITS_DIR / "test.csv", index=False)
    print(f"Splits — train: {train_end:,} | val: {val_end-train_end:,} | test: {n-val_end:,}")

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_files()
    df = parse_activity(df)
    df = forward_fill_activity(df)

    print(f"Events after fill: {len(df):,}")
    print(f"Activity classes: {sorted(df['activity'].unique())}")

    all_sensors = sorted(df["sensor"].unique().tolist())
    print(f"Unique sensors: {len(all_sensors)}")

    print("Building sliding windows...")
    features = make_windows(df, all_sensors)
    print(f"Windows generated: {len(features):,} | Features: {features.shape[1]}")

    features.to_csv(PROCESSED_DIR / "features.csv", index=False)
    print(f"Saved → {PROCESSED_DIR}/features.csv")

    save_splits(features)

    # update params with actual class count
    num_classes = features["label"].nunique()
    print(f"Num activity classes: {num_classes} — update params.yaml if needed")
    print("Done.")
