import os
import requests
from tqdm import tqdm

# Subjects selected (efficient and publishable)
subjects = ["chb01", "chb02", "chb05"]

# Only seizure-containing files (optimized subset)
files = {
    "chb01": [
        "chb01_03.edf",
        "chb01_04.edf",
        "chb01_15.edf",
        "chb01_16.edf",
        "chb01_18.edf",
    ],
    "chb02": [
        "chb02_16.edf",
        "chb02_18.edf",
        "chb02_19.edf",
    ],
    "chb05": [
        "chb05_02.edf",
        "chb05_03.edf",
        "chb05_11.edf",
    ]
}

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"

save_dir = "data/raw"
os.makedirs(save_dir, exist_ok=True)

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(save_path)}",
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

for subject in subjects:
    for filename in files[subject]:
        url = f"{BASE_URL}/{subject}/{filename}"
        save_path = os.path.join(save_dir, filename)
        
        if not os.path.exists(save_path):
            print(f"Downloading: {filename}")
            download_file(url, save_path)
        else:
            print(f"Already exists: {filename}")

print("\nDownload complete! Files saved inside data/raw/")
