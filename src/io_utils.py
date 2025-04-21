import pandas as pd
import pickle
import gzip


# Session metadata mapping
SESSION_METADATA = {
    "S1": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-12-11")},
    "S2": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-10-30")},
    "S3": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-11-10")},
    "S4": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-11-16")},
    "S5": {"Monkey": "Mars",  "Date": pd.to_datetime("2020-12-08")},
    "S6": {"Monkey": "Jones", "Date": pd.to_datetime("2021-10-11")},
    "S7": {"Monkey": "Jones", "Date": pd.to_datetime("2021-10-15")},
    "S8": {"Monkey": "Jones", "Date": pd.to_datetime("2021-10-20")},
}

def save_dataframe_with_metadata(df, session, filepath=None, TinC=None, TinI=None, MinC=None, MinI=None):
    """
    Save a DataFrame with attached session metadata using gzip-compressed pickle.

    Args:
        df (pd.DataFrame): The dataframe to save
        session (str): Session identifier (e.g. 'S6')
        filepath (str, optional): Destination file path, defaults to 'data/<session>.pkl.gz'
        TinC, TinI, MinC, MinI (array-like): neuron indices by condition
    """
    if filepath is None:
        filepath = f"data/{session}.pkl.gz"
    metadata = SESSION_METADATA.get(session, {}).copy()
    metadata.update({
        "Session": session,
        "NCells": len(df['spCellPop'].iloc[0]) if 'spCellPop' in df.columns else None,
        "TinC": TinC.tolist() if TinC is not None else None,
        "TinI": TinI.tolist() if TinI is not None else None,
        "MinC": MinC.tolist() if MinC is not None else None,
        "MinI": MinI.tolist() if MinI is not None else None
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
