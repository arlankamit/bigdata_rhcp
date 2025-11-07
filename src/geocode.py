# -*- coding: utf-8 -*-
import os, csv
from rapidfuzz import process, fuzz

# ждём data/stops_kz.csv с колонками: name,lat,lon
_DB = []
if os.path.exists("data/stops_kz.csv"):
    with open("data/stops_kz.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                _DB.append({"name": row["name"], "lat": float(row["lat"]), "lon": float(row["lon"])})
            except Exception:
                pass

def geocode_stop(text: str, city_hint: str | None = None):
    """
    Ищем ближайшее имя в офлайн-таблице, возвращаем {name,lat,lon,score} или None.
    city_hint пока не фильтруем (можно расширить).
    """
    if not _DB or not text:
        return None
    names = [r["name"] for r in _DB]
    m = process.extractOne(text, names, scorer=fuzz.WRatio, score_cutoff=85)
    if not m:
        return None
    hit = next(r for r in _DB if r["name"] == m[0])
    return {"name": hit["name"], "lat": hit["lat"], "lon": hit["lon"], "score": m[1]}
