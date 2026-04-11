import os
import ssl
import urllib.request
import zipfile
import sys
import time

URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"
ZIP_PATH = "data/icbhi_dataset.zip"
DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
ANNOT_DIR = os.path.join(DATA_DIR, "annotations")

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(ZIP_PATH):
        print("ZIP file already exists. Skipping download.")
    else:
        print(f"Downloading ICBHI 2017 Dataset (~1.98 GB)...")
        print(f"Source: {URL}")
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx) as response, open(ZIP_PATH, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length').strip())
                block_size = 8192 * 4
                downloaded = 0
                start_time = time.time()
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    downloaded += len(buffer)
                    progress_size = int(50 * downloaded / total_size)
                    speed = downloaded / (time.time() - start_time + 1e-9) / (1024 * 1024)
                    sys.stdout.write(f"\r[{'=' * progress_size}{' ' * (50 - progress_size)}] {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({speed:.1f} MB/s)")
                    sys.stdout.flush()
            print("\nDownload complete.")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            return False

    print("Extracting ZIP file...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # The ZIP contains a folder "ICBHI_final_database"
            zip_ref.extractall(DATA_DIR)
            
        print("Extraction complete. Reorganizing files...")
        
        os.makedirs(AUDIO_DIR, exist_ok=True)
        os.makedirs(ANNOT_DIR, exist_ok=True)
        
        extracted_folder = os.path.join(DATA_DIR, "ICBHI_final_database")
        if os.path.exists(extracted_folder):
            import glob
            import shutil
            
            # Move .wav to audio/ and .txt to annotations/
            wav_files = glob.glob(os.path.join(extracted_folder, "*.wav"))
            txt_files = glob.glob(os.path.join(extracted_folder, "*.txt"))
            
            for f in wav_files:
                shutil.move(f, os.path.join(AUDIO_DIR, os.path.basename(f)))
            for f in txt_files:
                shutil.move(f, os.path.join(ANNOT_DIR, os.path.basename(f)))
            
            print(f"Moved {len(wav_files)} audio files to {AUDIO_DIR}")
            print(f"Moved {len(txt_files)} annotation files to {ANNOT_DIR}")
            
            # Move patient diagnosis
            patient_diag = os.path.join(extracted_folder, "patient_diagnosis.csv")
            if os.path.exists(patient_diag):
                shutil.move(patient_diag, os.path.join(DATA_DIR, "patient_diagnosis.csv"))
            
            # Cleanup
            try:
                shutil.rmtree(extracted_folder)
                print("Cleaned up extracted folder.")
            except:
                pass
                
        print("\nDataset is ready!")
        print("You can now run: python load_dataset.py")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
