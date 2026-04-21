import requests
import zipfile
import pathlib

RAW_DIR = pathlib.Path("data/raw")
DATASET_URL = "https://zenodo.org/records/15708568/files/labeled_data.zip?download=1"

def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "labeled_data.zip"

    print("Downloading CASAS labeled dataset (236MB)...")
    r = requests.get(DATASET_URL, stream=True)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB", end="", flush=True)
    print()

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    zip_path.unlink()
    print(f"Done — extracted to {RAW_DIR}")

def validate():
    files = list(RAW_DIR.rglob("*"))
    print("Files found:")
    for f in files[:15]:
        print(f"  {f}")

if __name__ == "__main__":
    download_dataset()
    validate()
