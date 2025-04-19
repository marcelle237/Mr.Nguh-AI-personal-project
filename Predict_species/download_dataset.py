import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_animals10(dataset_dir="./animals10"):
    os.makedirs(dataset_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print("Downloading Animals-10 dataset from Kaggle...")
    api.dataset_download_files('alessiocorrado99/animals10', path=dataset_dir, unzip=False)
    zip_path = os.path.join(dataset_dir, 'animals10.zip')
    # The file is named animals10.zip when downloaded
    if not os.path.exists(zip_path):
        # Sometimes Kaggle names it differently
        files = os.listdir(dataset_dir)
        for f in files:
            if f.endswith('.zip'):
                zip_path = os.path.join(dataset_dir, f)
                break
    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Dataset download and extraction complete.")

if __name__ == "__main__":
    download_animals10()
