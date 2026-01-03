import os
import time
import requests
from tqdm import tqdm

FILES = [
    "chb01_01.edf",
    "chb01_02.edf",
]

BASE_URL = "https://physionet.org/files/chbmit/1.0.0/chb01/"
SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_with_resume(file):
    url = BASE_URL + file
    save_path = os.path.join(SAVE_DIR, file)

    # Get already downloaded file size
    downloaded = 0
    if os.path.exists(save_path):
        downloaded = os.path.getsize(save_path)

    headers = {"Range": f"bytes={downloaded}-"}
    with requests.get(url, headers=headers, stream=True) as r:
        if r.status_code not in (200, 206):
            print(f"❌ Error fetching: {file}")
            return

        # Total file size from server
        total = int(r.headers.get("content-length", 0)) + downloaded

        print(f"⬇ Resuming download: {file}")
        with open(save_path, "ab") as f, tqdm(
            total=total,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            desc=file,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

for f in FILES:
    done = False
    attempts = 0
    while not done and attempts < 5:
        try:
            download_with_resume(f)
            done = True
        except Exception as e:
            attempts += 1
            print(f"⚠ Error downloading {f}, retrying in 5s...")
            time.sleep(5)

print("\n🎉 All downloads completed (or resumed as far as possible)!")
