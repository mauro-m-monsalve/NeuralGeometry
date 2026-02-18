import pandas as pd
import pickle
import gzip
import urllib.request


# Session metadata mapping
SESSION_METADATA = {
    "S1": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-12-11"), 
           'TinC': [31, 51, 57, 58, 70, 85, 98, 99, 101, 102, 106, 114, 123, 140, 147, 150, 168], 
           'TinI': [2, 7, 14, 77, 115, 121, 132, 136], 
           'MinC': [10, 39, 60, 68, 78, 90, 93, 142, 154, 162], 
           'MinI': [15, 116] },
    "S2": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-10-30"), 
           'TinC': [1, 12, 22, 31, 37, 43, 45, 53, 56, 62, 63, 70, 71], 
           'TinI': [18, 30, 33, 66, 80], 
           'MinC': [9, 10, 64, 79, 85], 
           'MinI': [2, 68] },
    "S3": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-11-10"), 
           'TinC': [5, 21, 29, 30, 39, 42, 45, 48, 53], 
           'TinI': [2, 3, 22, 35, 38, 43, 46], 
           'MinC': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], 
           'MinI': [19, 31] },
    "S4": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-11-16"), 
           'TinC': [3, 10, 12, 17, 27, 33, 38, 57, 61, 65, 67, 73, 82, 84, 87, 96, 97, 101, 107, 109, 114], 
           'TinI': [11, 20, 29, 39, 45, 52, 59, 75, 88, 89, 92, 93, 95, 99, 102, 110, 111, 117, 125], 
           'MinC': [9, 16, 30, 32, 58, 68, 78], 
           'MinI': [34, 119] },
    "S5": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-12-08"), 
           'TinC': [1,3,8,24,27,28,31,33,34,43,44,50,51,54,57,60,62,79,84],
           'TinI': [0, 2, 5, 64, 68, 81, 90],
           'MinC': [13, 26, 89, 95],
           'MinI': [16, 48, 49]},
    "S6": {"Monkey": "Jones", "Date": pd.to_datetime("2021-10-11"), 
           'TinC': [1, 2, 4, 7, 14, 52, 77, 90, 92, 101, 116, 120], 
           'TinI': [11, 20, 25, 27, 28, 29, 30, 33, 34, 35, 41, 42, 51, 70, 105, 112, 128], 
           'MinC': [10, 31, 102, 115], 
           'MinI': [24, 36, 40, 48, 103] },
    "S7": {"Monkey": "Jones", "Date": pd.to_datetime("2021-10-15"), 
           'TinC': [1, 3, 4, 8, 10, 11, 15, 19, 28, 40, 50, 70, 73, 88, 90, 92, 96, 98, 102, 103, 107, 110, 123, 124, 126, 132, 135, 136, 137, 138, 140, 143, 144, 152], 
           'TinI': [37], 
           'MinC': [24, 26, 54, 100, 105, 108], 
           'MinI': [5, 52, 56, 134] },
    "S8": {"Monkey": "Jones", "Date": pd.to_datetime("2021-10-20"), 
           'TinC': [19, 22, 29, 37, 39, 46, 49, 56, 57, 59, 60, 64, 66, 68, 69, 73, 91, 92, 104, 108, 155, 163, 167, 168, 169, 172, 181], 
           'TinI': [28, 127, 182, 194], 
           'MinC': [7, 15, 53, 70, 71, 72, 111, 126, 130, 139, 146, 162, 173, 179, 185], 
           'MinI': [31, 114, 115, 142, 145, 151] }
}

def save_dataframe_with_metadata(df, session, filepath=None):
    """
    Save a DataFrame with attached session metadata using gzip-compressed pickle.

    Args:
        df (pd.DataFrame): The dataframe to save
        session (str): Session identifier (e.g. 'S6')
        filepath (str, optional): Destination file path, defaults to 'data/<session>.pkl.gz'
    """
    if filepath is None:
        filepath = f"data/{session}.pkl.gz"

    # Copy predefined metadata for the session
    metadata = SESSION_METADATA.get(session, {}).copy()

    # Add runtime fields
    metadata.update({
        "Session": session,
        "NCells": len(df['spCellPop'].iloc[0]) if 'spCellPop' in df.columns else None,
    })

    bundle = {'df': df, 'attrs': metadata}

    with gzip.open(filepath, 'wb') as f:
        pickle.dump(bundle, f)

def load_dataframe_with_metadata(session, filepath=None):
    """
    Load a DataFrame and its associated metadata from a gzip-compressed pickle file.

    TinC and TinI contain the indices of neurons classified as responsive to the contralateral and ipsilateral targets, respectively.
    MinC and MinI represent motion-responsive neurons for contralateral and ipsilateral directions, respectively.

    Args:
        session (str): Session identifier (e.g. 'S6')
        filepath (str, optional): Path to file, defaults to 'data/<session>.pkl.gz'

    Returns:
        df (pd.DataFrame): The dataframe
        metadata (dict): The metadata dictionary
    """
    if filepath is None:
        filepath = f"data/{session}.pkl.gz"
    with gzip.open(filepath, 'rb') as f:
        bundle = pickle.load(f)
    df = bundle['df']
    df.attrs.update(bundle.get('attrs', {}))
    return df


import os
import requests

ZENODO_DOI = "10.5281/zenodo.15093133"
ZENODO_API = f"https://doi.org/{ZENODO_DOI}"

def download_session(session: str, overwrite: bool = False) -> str:
    """
    Download the preprocessed LIP dataset for a given session from Zenodo.

    Args:
        session (str): Session name, one of 'S1' to 'S8' (without .pkl.gz extension)
        overwrite (bool): If True, overwrite existing file

    Returns:
        str: Path to the downloaded file (data/{session}.pkl.gz)
    """
    filename = f"{session}.pkl.gz"
    target_path = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)

    if os.path.exists(target_path) and not overwrite:
        print(f"File already exists: {target_path}")
        return target_path

    # Resolve DOI to latest version
    try:
        print(f"Resolving DOI {ZENODO_DOI} to fetch latest version info...")
        response = requests.get(ZENODO_API, allow_redirects=True)
        response.raise_for_status()

        # Follow redirect to final Zenodo record
        latest_url = response.url
        record_id = latest_url.strip("/").split("/")[-1]
        files_api = f"https://zenodo.org/api/records/{record_id}"
        
        # Get list of files
        files_response = requests.get(files_api)
        files_response.raise_for_status()
        files_info = files_response.json()["files"]

        # Find desired file
        file_entry = next((f for f in files_info if f["key"] == filename), None)
        if not file_entry:
            raise FileNotFoundError(f"{filename} not found in Zenodo record {record_id}.")

        file_url = file_entry["links"]["self"]

        # Download
        print(f"Downloading {filename} from {file_url}...")
        urllib.request.urlretrieve(file_url, target_path)
        print(f"Download complete: {target_path}")
        return target_path

    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None
    

import os
import sys
import contextlib

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
