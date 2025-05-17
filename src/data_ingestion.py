import os
import shutil
import kagglehub

def download_dataset(download_dir="data/raw/customer_support_twitter"):
    # If dataset folder already exists and is not empty, skip downloading
    if os.path.isdir(download_dir) and os.listdir(download_dir):
        print(f"Dataset already downloaded in: {download_dir}")
        return

    print("Downloading dataset...")
    # Download dataset, kagglehub returns a temporary path
    temp_path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    print(f"Downloaded dataset temporary path: {temp_path}")

    # Move dataset from temp path to desired folder
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)  # Remove if partially exists
    shutil.move(temp_path, download_dir)

    print(f"Dataset moved to: {download_dir}")

if __name__ == "__main__":
    download_dataset()
