import pandas as pd
import glob
import os

from joblib import Parallel, delayed

def load_data(folder):

    print("Searching inside:", folder)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    files = glob.glob(os.path.join(folder, "**", "*.psv"), recursive=True)

    print("Found .psv files:", len(files))

    if len(files) == 0:
        raise ValueError("No .psv files found inside the data folder.")

    def read_psv(f):
        df = pd.read_csv(f, sep="|")
        df["PatientID"] = os.path.basename(f).replace(".psv", "")
        return df

    print("Loading files in parallel...")
    all_data = Parallel(n_jobs=-1)(delayed(read_psv)(f) for f in files)

    data = pd.concat(all_data, ignore_index=True)
    return data

# prepare_sequence_data removed per user request (no DL)
