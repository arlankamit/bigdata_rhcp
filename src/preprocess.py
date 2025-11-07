# -*- coding: utf-8 -*-
import os, re, hashlib, random, pathlib
import pandas as pd
from unidecode import unidecode

DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(exist_ok=True)
OUT_PARQUET = DATA_DIR / "complaints.parquet"
SEED = 42
random.seed(SEED)

_RE_SPACES = re.compile(r"\s+")
_RE_PII = re.compile(r"(\+?\d[\d\-\s]{8,}\d|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_RE_URL  = re.compile(r"https?://\S+|www\.\S+")

def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = _RE_URL.sub(" ", s)
    s = _RE_PII.sub(" ", s)
    s = _RE_SPACES.sub(" ", s)
    return s.strip()

def _norm_for_hash(s: str) -> str:
    s = unidecode((s or "").lower())
    s = re.sub(r"[^a-z0-9а-яёқңғүұіһәө\- ]+", " ", s)
    s = _RE_SPACES.sub(" ", s).strip()
    return s

def load_any():
    # подхватываем первый подходящий файл
    cands = [
        "data/complaints.csv",
        "data/transport_complaints_astana_almaty_100k.csv",
        "data/complaints.parquet",
    ]
    for p in cands:
        if os.path.exists(p):
            if p.endswith(".parquet"):
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            return df
    raise FileNotFoundError("Put CSV/Parquet with columns at least: text, priority, aspect")

def main():
    df = load_any()
    # ожидаем text/priority/aspect/route/time/place (что есть — берём)
    if "text" not in df.columns:
        raise RuntimeError("Dataset must contain column 'text'")
    df["text"] = df["text"].astype(str).map(_clean_text)
    df = df[df["text"].str.len() > 2].copy()

    # дедуп по нормализованному хэшу
    df["norm_hash"] = df["text"].map(_norm_for_hash).map(lambda x: hashlib.md5(x.encode()).hexdigest())
    df = df.drop_duplicates(subset=["norm_hash"]).drop(columns=["norm_hash"]).reset_index(drop=True)

    # баланс классов (если есть priority)
    if "priority" in df.columns:
        counts = df["priority"].value_counts()
        if len(counts) > 1:
            m = counts.min()
            parts = []
            for k, g in df.groupby("priority"):
                parts.append(g.sample(n=min(m*2, len(g)), random_state=SEED))  # легкое выравнивание
            df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # базовая колонка text_clean
    df["text_clean"] = df["text"]

    # сохраняем
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[preprocess] rows={len(df)} saved -> {OUT_PARQUET}")

if __name__ == "__main__":
    main()
