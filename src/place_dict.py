# -*- coding: utf-8 -*-
import os, glob, re, logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------- нормализация ----------
def _norm_text(s: str) -> str:
    try:
        from unidecode import unidecode
        s = unidecode(s or "")
    except Exception:
        s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9а-яёқңғүұіһәө\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _to_alias_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    logger.warning("place_dict: unexpected aliases type=%s value=%r -> []", type(val).__name__, val)
    return []

# ---------- загрузка YAML/JSON/CSV ----------
def _load_yaml_files() -> Dict[str, List[dict]]:
    """
    Ищем YAML вида:
      { city: "Almaty", stops: [ {name, aliases?, lat?, lon?}, ... ] }
    или просто список остановок в файле.
    При желании можно указать шаблоны через env STOPS_GLOB (через запятую).
    """
    import yaml
    patterns = os.getenv("STOPS_GLOB")
    if patterns:
        patterns = [p.strip() for p in patterns.split(",") if p.strip()]
    else:
        patterns = [
            "data/*.yaml",
            "data/stops/*.yaml",
            "stops/*.yaml",
            "*.yaml",
        ]

    result: Dict[str, List[dict]] = {}
    seen = set()

    for pat in patterns:
        for path in glob.glob(pat):
            if (path,) in seen:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except Exception as e:
                logger.warning("place_dict: skip file %s (%s)", path, e)
                continue

            # формат 1: {city:..., stops:[...]}
            if isinstance(data, dict) and "stops" in data:
                city = str(data.get("city") or "").strip() or os.path.splitext(os.path.basename(path))[0]
                stops = data.get("stops") or []
            # формат 2: [ {...}, "строка", ... ]
            elif isinstance(data, list):
                city = os.path.splitext(os.path.basename(path))[0]
                stops = data
            else:
                continue

            norm_stops: List[dict] = []
            for it in stops:
                if isinstance(it, str):
                    base = it.strip()
                    if not base:
                        continue
                    norm_stops.append({"name": base, "aliases": []})
                elif isinstance(it, dict):
                    base = (it.get("name") or it.get("base") or "").strip()
                    if not base:
                        continue
                    aliases = _to_alias_list(it.get("aliases"))
                    lat = it.get("lat"); lon = it.get("lon")
                    # best-effort к числам
                    try:
                        lat = float(lat) if lat not in (None, "") else None
                        lon = float(lon) if lon not in (None, "") else None
                    except Exception:
                        lat = lon = None
                    norm_stops.append({"name": base, "aliases": aliases, "lat": lat, "lon": lon})
                # иные типы пропускаем

            if norm_stops:
                result.setdefault(city, []).extend(norm_stops)
            seen.add((path,))
    return result

def _read_json(path: str) -> Optional[dict]:
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return None

def _read_csv_stops(path: str) -> List[dict]:
    """CSV: name,lat,lon → список записей без города (уйдут в GLOBAL)."""
    out: List[dict] = []
    try:
        import csv
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                lat = row.get("lat"); lon = row.get("lon")
                try:
                    lat = float(lat) if lat not in (None, "") else None
                    lon = float(lon) if lon not in (None, "") else None
                except Exception:
                    lat = lon = None
                out.append({"name": name, "aliases": [], "lat": lat, "lon": lon})
    except Exception:
        pass
    return out

# ---------- публичная загрузка ----------
def load_stop_dict() -> Dict[str, List[dict]]:
    # 1) YAML (могут быть несколько файлов)
    out = _load_yaml_files()
    if out:
        return out

    # 2) JSON (единый словарь вида {City: [..]})
    raw = _read_json("data/stops.json")
    if raw:
        return _normalize_raw_places(raw)

    # 3) Минимальный fallback + CSV
    base: Dict[str, List[dict]] = {
        "Astana": [
            {"name": "Сарыарка", "aliases": ["Сарыарқа"], "lat": 51.169, "lon": 71.449},
        ],
        "Almaty": [
            {"name": "Сайран", "aliases": [], "lat": 43.242, "lon": 76.882},
            {"name": "Ақсай",  "aliases": [], "lat": 43.220, "lon": 76.857},
        ],
    }
    csv_extra = _read_csv_stops("data/stops_kz.csv")
    if csv_extra:
        base.setdefault("GLOBAL", [])
        known = {r["name"] for lst in base.values() for r in lst}
        for rec in csv_extra:
            if rec["name"] not in known:
                base["GLOBAL"].append(rec)
    return base

def _normalize_raw_places(raw: dict) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for city, stops in (raw or {}).items():
        res: List[dict] = []
        for it in (stops or []):
            if isinstance(it, str):
                base = it.strip()
                if not base: continue
                res.append({"name": base, "aliases": []})
            elif isinstance(it, dict):
                base = (it.get("name") or it.get("base") or "").strip()
                if not base: continue
                aliases = _to_alias_list(it.get("aliases"))
                lat = it.get("lat"); lon = it.get("lon")
                try:
                    lat = float(lat) if lat not in (None, "") else None
                    lon = float(lon) if lon not in (None, "") else None
                except Exception:
                    lat = lon = None
                res.append({"name": base, "aliases": aliases, "lat": lat, "lon": lon})
        out[city] = res
    return out

# Держим прогруженный словать в модуле (используется во многих местах)
STOP_DICT: Dict[str, List[dict]] = load_stop_dict()

# ---------- генерация вариантов ----------
def _all_variants_for_city(stops: List[dict]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rec in stops:
        base = (rec.get("name") or "").strip()
        if base:
            out.append((_norm_text(base), base))
        for a in _to_alias_list(rec.get("aliases")):
            out.append((_norm_text(a), base))
    return out

def _all_variants_global(stop_dict: Dict[str, List[dict]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for _, stops in (stop_dict or {}).items():
        out.extend(_all_variants_for_city(stops))
    return out

# ---------- fuzzy-поиск ----------
def fuzzy_stop_match(text: str, city_hint: Optional[str] = None, threshold: int = 70) -> Tuple[Optional[str], int]:
    """
    (best_base_name, score 0..100). Сначала прямое вхождение, затем RapidFuzz, затем difflib.
    """
    q = _norm_text(text)
    if not q:
        return (None, 0)

    stop_dict = STOP_DICT or {}
    if city_hint and city_hint in stop_dict:
        variants = _all_variants_for_city(stop_dict[city_hint])
    else:
        variants = _all_variants_global(stop_dict)

    if not variants:
        return (None, 0)

    # прямое вхождение
    for v_norm, base in variants:
        if v_norm and v_norm in q:
            return (base, 100)

    candidates = [v for v, _ in variants]
    rev_map = {v: base for v, base in variants}

    try:
        from rapidfuzz import process, fuzz
        hit = process.extractOne(q, candidates, scorer=fuzz.WRatio)
        if hit:
            cand, score, _ = hit
            base = rev_map.get(cand)
            return (base if score >= threshold else None, int(score))
    except Exception:
        pass

    import difflib
    best = difflib.get_close_matches(q, candidates, n=1, cutoff=max(0.0, min(1.0, threshold/100.0)))
    if best:
        cand = best[0]
        base = rev_map.get(cand)
        score = int(100 * difflib.SequenceMatcher(None, q, cand).ratio())
        return (base if score >= threshold else None, score)
    return (None, 0)
