# -*- coding: utf-8 -*-
import os, yaml, pandas as pd

def load_config(path="config.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_csv_smart(path: str) -> pd.DataFrame:
    # для больших файлов лучше chunksize, но оставим просто
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")
