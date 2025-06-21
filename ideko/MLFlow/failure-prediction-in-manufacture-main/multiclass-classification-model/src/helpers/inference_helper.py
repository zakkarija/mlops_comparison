import os
import random
import zipfile
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def _detect_delim(sample: str) -> str:
    """Pick the delimiter that splits sample into the most fields."""
    for d in [';', ',', '\t', ' ']:
        parts = sample.split(d)
        if len(parts) > 1:
            return d
    return ','  # fallback

def load_random_inference_data(data_root: str) -> Tuple[pd.DataFrame, str, str]:
    anomaly_types = ["electrical_anomalies", "mechanical_anomalies", "not_anomalous"]
    selected = random.choice(anomaly_types)
    data_dir = os.path.join(data_root, selected)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} missing or empty")

    rnd = random.choice(os.listdir(data_dir))
    dir_path = os.path.join(data_dir, rnd)
    zip_files = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No .zip in {dir_path}")

    zip_path = os.path.join(dir_path, zip_files[0])
    logger.info(f"Loading {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        csvs = [n for n in z.namelist() if n.endswith('.csv')]
        if not csvs:
            raise FileNotFoundError("No CSV inside zip")
        with z.open(csvs[0]) as f:
            # peek header line
            header = f.readline().decode(errors='ignore')
            delim = _detect_delim(header)
            f.seek(0)  # rewind
            df = pd.read_csv(f, sep=delim)
    logger.info(f"Detected delimiter='{delim}', columns={df.columns.tolist()}")

    fname = f"{zip_files[0]} from {rnd}"
    actual = selected.replace("_anomalies","").replace("not_anomalous","normal")
    return df, fname, actual

def preprocess_for_inference(
    df: pd.DataFrame,
    feature_columns: List[str],
    seq_len: int
) -> np.ndarray:
    logger.info("Performing feature engineering…")
    if 'f3' not in df.columns:
        raise ValueError(f"Missing f3 in {df.columns.tolist()}")

    df.rename(columns={'f3':'f3_current'}, inplace=True)
    df['f3_rolling_mean_10'] = df['f3_current'].rolling(10).mean()
    df['f3_rolling_std_10']  = df['f3_current'].rolling(10).std()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ensure id/timestamp
    if 'equipment_id' not in df: df['equipment_id']='inference_equipment'
    if 'event_timestamp' not in df:
        df['event_timestamp'] = pd.to_datetime(pd.RangeIndex(len(df)), unit='s')

    X = []
    for eq in df['equipment_id'].unique():
        sub = df[df['equipment_id']==eq].sort_values('event_timestamp')
        F = sub[feature_columns].values
        if len(F) < seq_len: continue
        for i in range(len(F)-seq_len+1):
            X.append(F[i:i+seq_len])
    X_arr = np.array(X)
    logger.info(f"Built {X_arr.shape[0]} sequences.")
    return X_arr
