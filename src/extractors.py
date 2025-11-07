# -*- coding: utf-8 -*-
import re, pandas as pd
from typing import Optional, List, Dict, Iterable

from .constants import ASPECT_PATTERNS, STOP_HINTS
from .place_dict import STOP_DICT, load_stop_dict, fuzzy_stop_match

# geocode_stop — опционально: если модуля нет, просто пропускаем геокодинг
try:
    from .geocode import geocode_stop  # def geocode_stop(text, city_hint=None) -> {"name","lat","lon","score"}
except Exception:
    geocode_stop = None  # type: ignore

# === собираем плоский список имён остановок (база + алиасы) ===
def _iter_stop_strings(stop_dict: Dict[str, List[dict]]) -> Iterable[str]:
    for stops in (stop_dict or {}).values():
        for it in (stops or []):
            if isinstance(it, str):
                s = it.strip()
                if s: yield s
            elif isinstance(it, dict):
                base = (it.get("name") or it.get("base") or "").strip()
                if base: yield base
                for a in it.get("aliases") or []:
                    s = str(a).strip()
                    if s: yield s

ALL_STOPS: List[str] = sorted(set(_iter_stop_strings(STOP_DICT)))

# === регулярки ===
ROUTE_PATTERNS = [
    re.compile(r"(?:маршрут(?:а|ы)?|№|N)\s*([0-9]{1,4})", re.I),
    re.compile(r"([0-9]{1,4})\s*(?:маршрут(?:а|ы)?|автобус(?:а|ы)?)", re.I),
    re.compile(r"(?:автобус(?:а|ы)?|автобусы)\s*([0-9]{1,4})", re.I),
    re.compile(r"([0-9]{1,4})\s*[- ]?бағыт", re.I),
]
TIME_PAT  = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")

PLACE_PATTERNS = [
    re.compile(r"(?:на|у)\s+остановк\w+\s+([^\n]+)", re.I),
    re.compile(r"([A-Za-zА-Яа-яЁёӘәӨөҮүҰұҚқҒғІі]+(?:\s+[A-Za-zА-Яа-яЁёӘәӨөҮүҰұҚқҒғІі]+){0,3})\s+аялдамасына", re.I),
]

PARTICIPANT_PATTERNS = [
    (re.compile(r"\bводител[ьяюе]\b|\bжүргізуш[іi]\b", re.I), "driver"),
    (re.compile(r"\bкондуктор\w*|\bконтрол[её]р\w*", re.I), "conductor"),
    (re.compile(r"\bинспектор\w*\b", re.I), "inspector"),
    (re.compile(r"\bдиспетчер\w*\b|\bоператор\b", re.I), "dispatcher"),
    (re.compile(r"\bпассажир\w*\b", re.I), "passenger"),
]

NEGATE_SAFETY = re.compile(r"\bучени\w+|\bтренировочн\w+|\bпланов\w+|\bжоспарл\w+", re.I)

def is_negated_safety(text: str) -> bool:
    return bool(NEGATE_SAFETY.search(text or ""))

def _clean_place(p: str) -> str:
    p = re.sub(r"\s+(?:в\s+)?([01]?\d|2[0-3])(:[0-5]\d)?\b.*$", "", p)
    p = re.sub(r"\s*(остановк\w*|аялдамасы|аялдамасына)\s*$", "", p, flags=re.I)
    return p.strip(" ,.-")

# === аспекты по правилам ===
def detect_aspects(text: str) -> List[str]:
    text = text or ""
    found = set()
    for asp, pats in ASPECT_PATTERNS.items():
        for p in pats:
            if re.search(p, text, flags=re.I):
                found.add(asp); break
    return sorted(found) if found else ["other"]

# === маршрут/время ===
def extract_route(text: str) -> Optional[str]:
    t = text or ""
    for pat in ROUTE_PATTERNS:
        m = pat.search(t)
        if m:
            for g in m.groups():
                if g: return g
    return None

def extract_time(text: str) -> Optional[str]:
    if not text: return None
    m = TIME_PAT.search(text)
    if m: return m.group(0)
    if re.search(r"\bтаңертең\b|\bутром\b", text, flags=re.I): return "morning"
    if re.search(r"\bтүс\b|\bднем\b|\bтүскі\b", text, flags=re.I): return "noon"
    if re.search(r"\bкеш\b|\bвечером\b|\bкешке\b", text, flags=re.I): return "evening"
    return None

# === city hint ===
_CITY_PATTERNS = {
    "Astana": [r"\bастана\b", r"\bнур[-\s]?султан\b", r"\bнурсултан\b", r"\bastana\b", r"\bns\b"],
    "Almaty": [r"\bалматы\b", r"\bалмата\b", r"\bалма[-\s]?ата\b", r"\balmaty\b"],
}
def detect_city_hint(text: str) -> Optional[str]:
    t = (text or "").lower()
    for city, pats in _CITY_PATTERNS.items():
        for p in pats:
            if re.search(p, t, flags=re.I): return city
    return None

# === поиск города/координат по базе ===
def _find_city_latlon_for_base(base: str) -> Dict[str, Optional[float]]:
    for city, stops in (STOP_DICT or {}).items():
        for rec in (stops or []):
            if isinstance(rec, dict) and (rec.get("name") or "").strip() == base:
                return {
                    "city": city,
                    "lat": rec.get("lat"),
                    "lon": rec.get("lon"),
                }
    return {"city": None, "lat": None, "lon": None}

# === старый интерфейс (строка) ===
def extract_place(text: str) -> Optional[str]:
    t = text or ""
    # 1) явные шаблоны "на остановке ХХХ"
    for pat in PLACE_PATTERNS:
        m = pat.search(t)
        if m:
            return _clean_place(m.group(1))
    # 2) эвристика по стоп-стемам
    tl = t.lower()
    for stem in STOP_HINTS:
        idx = tl.find(stem)
        if idx != -1:
            tail = t[idx: idx + 140]
            m = re.search(r"([A-Za-zА-Яа-яЁёӘәӨөҮүҰұҚқҒғІі0-9\s\-\.,]{3,})", tail, flags=re.I)
            if m:
                return _clean_place(m.group(1))
    # 3) fuzzy по словарю
    hint = detect_city_hint(t)
    best, score = fuzzy_stop_match(t, city_hint=hint, threshold=87)
    if best:
        return best
    return None

# === структурный вывод (city/lat/lon/score) ===
def extract_place_struct(text: str) -> Optional[Dict]:
    hint = detect_city_hint(text)
    # 1) geocode (если доступен)
    if geocode_stop is not None:
        try:
            geo = geocode_stop(text, city_hint=hint)
        except Exception:
            geo = None
        if geo and geo.get("name"):
            return {
                "city_hint": hint,
                "name": geo.get("name"),
                "display": geo.get("name"),
                "lat": geo.get("lat"),
                "lon": geo.get("lon"),
                "score": geo.get("score", 0),
                "method": "geocode+fuzzy",
            }
    # 2) fuzzy по словарю
    best, score = fuzzy_stop_match(text, city_hint=hint, threshold=70)
    if best:
        meta = _find_city_latlon_for_base(best)
        return {
            "city_hint": hint,
            "name": best,
            "display": best,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "score": score,
            "method": "fuzzy-only",
        }
    # 3) fallback: вырезка кандидата по шаблону
    for pat in PLACE_PATTERNS:
        m = pat.search(text or "")
        if m:
            cand = _clean_place(m.group(1))
            return {"city_hint": hint, "name": cand, "display": cand,
                    "lat": None, "lon": None, "score": 0, "method": "candidate-only"}
    return None

# === участники ===
def extract_participant(text: str) -> Optional[Dict]:
    t = text or ""
    for pat, label in PARTICIPANT_PATTERNS:
        m = pat.search(t)
        if m: return {"role": label, "match": m.group(0)}
    return None

# === пакетная обработка ===
def batch_apply(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["route_extracted"] = df["text"].apply(extract_route)
    df["time_extracted"]  = df["text"].apply(extract_time)
    df["place_extracted"] = df["text"].apply(extract_place)
    df["participant"]     = df["text"].apply(lambda t: (extract_participant(t) or {}).get("role"))
    df["aspects_rule"]    = df["text"].apply(detect_aspects)
    return df
