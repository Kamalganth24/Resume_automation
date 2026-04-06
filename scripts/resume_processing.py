import os
import json
import shutil
from file_cleaning import process_resume
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent #get root directory

INCOMING_DIR = BASE_DIR / "Incoming_folder"
PROCESSED_DIR = BASE_DIR / "processed_folder"  

STATE_FILE = BASE_DIR / "state_file/processed_files.json"
OUTPUT_FILE = BASE_DIR / "resume_list.json"
 
def initialize_storage():
    """
    Ensure all directories and files exist.
    """
    # Create the main folders
    os.makedirs(INCOMING_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # NEW: Create the 'state_file' folder specifically
    os.makedirs(STATE_FILE.parent, exist_ok=True)

    # Create/reset state file if missing
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            json.dump({"processed": []}, f, indent=2)

    # Create output file if missing
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            json.dump([], f, indent=2)


def load_processed():
    with open(STATE_FILE, "r") as f:
        return json.load(f)["processed"]


def save_processed(processed):
    with open(STATE_FILE, "w") as f:
        json.dump({"processed": processed}, f, indent=2)


def save_output(data):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def main():

    print("Resume_processing is running")
    processed = set(load_processed())
    resume_text_list = []

    for file in os.listdir(INCOMING_DIR):

        if file.startswith("."):
            continue

        if file in processed:
            print("File processed already !")
            continue

        file_path = INCOMING_DIR / file

        if not file_path.is_file():
            continue

        print(f"Processing {file}")

        try:
            result = process_resume(file_path)

            if result:
                resume_text_list.append(result)

            shutil.move(file_path, PROCESSED_DIR / file)
            processed.add(file)


        except Exception as e:
            print(f"Error processing {file}: {e}")

    save_processed(list(processed))
    save_output(resume_text_list)

    return resume_text_list

    

     