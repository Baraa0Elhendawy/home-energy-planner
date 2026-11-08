import os
import zipfile
import urllib.request
import json
import logging
import pandas as pd
import numpy as np

# Deliverable 2: Reproducible preprocessing (fixed seed)
np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
DATA_DIR = "data"

def generate_config():
    """Generates the static configurations for simulated constraints."""
    config = {
        "battery": {
            "capacity_kwh": 5.0,
            "max_rate_kw": 2.5,
            "efficiency": 0.95
        },
        "power_limit_kw": 6.0,
        "appliances": {
            "washer": {"duration_h": 1, "load_kw": 1.2, "prereq": None, "shared_circuit": None},
            "dryer": {"duration_h": 1, "load_kw": 2.5, "prereq": "washer", "shared_circuit": None},
            "dishwasher": {"duration_h": 1, "load_kw": 1.5, "prereq": None, "shared_circuit": None},
            "ev_charger": {"duration_h": 3, "load_kw": 3.5, "prereq": None, "shared_circuit": "oven"},
            "oven": {"duration_h": 1, "load_kw": 2.2, "prereq": None, "shared_circuit": "ev_charger"}
        },
        "prices": {
            # 24h ToU: Cheap overnight (0.10), avg day (0.20), expensive peak 17:00-20:00 (0.50)
            "hourly": [0.1]*7 + [0.2]*10 + [0.5]*4 + [0.1]*3
        }
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    logging.info("Generated config.json")

def download_dataset():
    """Downloads and extracts the UCI dataset if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "power.zip")
    csv_path = os.path.join(DATA_DIR, "household_power_consumption.txt")
    
    if not os.path.exists(csv_path):
        logging.info(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        logging.info("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path)
    else:
        logging.info("Dataset already exists locally. Skipping download.")
    return csv_path

def preprocess_and_split(csv_path):
    """Processes raw minute data to hourly, engineers features, and splits chronologically."""
    logging.info("Parsing raw CSV (this may take a moment)...")
    df = pd.read_csv(csv_path, sep=';', na_values=['?'], low_memory=False)
    
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.set_index('Datetime', inplace=True)
    
    logging.info("Aggregating to hourly load...")
    hourly = df[['Global_active_power']].resample('h').mean()
    
    hourly['Global_active_power'] = hourly['Global_active_power'].interpolate(limit=3)
    hourly.dropna(inplace=True)
    
    logging.info("Engineering temporal and autoregressive features...")
    hourly['hour'] = hourly.index.hour
    hourly['weekday'] = hourly.index.weekday
    hourly['month'] = hourly.index.month
    hourly['prev_24h'] = hourly['Global_active_power'].shift(24)
    hourly['same_hour_last_week'] = hourly['Global_active_power'].shift(24 * 7)
    
    for i in range(1, 25):
        hourly[f'target_{i}'] = hourly['Global_active_power'].shift(-i)
        
    dataset = hourly.dropna()
    
    test_size = 365 * 24
    val_size = 90 * 24
    
    train_end = len(dataset) - test_size - val_size
    val_end = len(dataset) - test_size
    
    train = dataset.iloc[:train_end]
    val = dataset.iloc[train_end:val_end]
    test = dataset.iloc[val_end:]
    
    logging.info(f"Split sizes -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    train.to_csv(os.path.join(DATA_DIR, "train.csv"))
    val.to_csv(os.path.join(DATA_DIR, "val.csv"))
    test.to_csv(os.path.join(DATA_DIR, "test.csv"))
    logging.info("Exported train.csv, val.csv, test.csv to /data directory.")

if __name__ == "__main__":
    generate_config()
    raw_csv = download_dataset()
    preprocess_and_split(raw_csv)