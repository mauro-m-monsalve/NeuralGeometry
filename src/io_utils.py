

import pandas as pd
import pickle
import gzip
from datetime import datetime

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
import urllib.request

def download_session(session: str, overwrite: bool = False) -> str:
    """
    Download the preprocessed LIP dataset for a given session from Zenodo.

    Parameters:
        session (str): Session name, one of 'S1' to 'S8'.
        overwrite (bool): If True, overwrite existing file.

    Returns:
        str: Path to the downloaded file (data/{session}.pkl.gz)
    """
    assert session in [f"S{i}" for i in range(1, 9)], "Session must be one of 'S1' to 'S8'"

    zenodo_base = "https://zenodo.org/records/15093134/files"
    filename = f"{session}.pkl.gz"
    url = f"{zenodo_base}/{filename}"
    target_path = os.path.join("data", filename)

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(target_path) or overwrite:
        print(f"Downloading {filename} from Zenodo...")
        urllib.request.urlretrieve(url, target_path)
        print(f"Download complete: {target_path}")
    else:
        print(f"File already exists: {target_path}")

    return target_path
