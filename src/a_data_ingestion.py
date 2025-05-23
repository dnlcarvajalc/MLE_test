import os
import shutil
import kagglehub

def download_dataset(download_dir="data/raw/customer_support_twitter"):
    """
    Downloads the "Customer Support on Twitter" dataset from KaggleHub if it doesn't already exist.

    If the specified download directory already exists and is not empty, the function assumes
    the dataset is already downloaded and skips the download process.

    Args:
        download_dir (str): Path to the directory where the dataset should be saved.
            Defaults to "data/raw/customer_support_twitter".

    Returns:
        None
    """
    if os.path.isdir(download_dir) and os.listdir(download_dir):
        print(f"Dataset already downloaded in: {download_dir}")
        return

    print("Downloading dataset...")
    temp_path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    print(f"Downloaded dataset temporary path: {temp_path}")

    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    shutil.move(temp_path, download_dir)

    print(f"Dataset moved to: {download_dir}")

if __name__ == "__main__":
    download_dataset()
