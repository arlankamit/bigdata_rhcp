Haka Analyzer

FastAPI service that analyzes public transport complaints (ru/kk), extracts key signals (route, time, place, participant), predicts priority and aspect, and returns actionable advice in Kazakh.
Includes a one-file demo UI (/demo/index.html) and auto-generated reports (PNG charts).

Features

🔤 Robust to typos, mixed Russian/Kazakh, and incomplete texts

🧠 ML models: priority (classifier) + aspect (single-label) with TF-IDF (word + char)

🧭 Place detection with fuzzy matching + optional coordinates from local stop dictionaries (YAML/CSV)

🧑‍🤝‍🧑 Participant detection (driver/dispatcher/conductor/passenger/inspector)

✅ Every decision is accompanied by a recommended action (KZ)

📊 Visualizations: top routes, aspect frequency, priority over time, time-of-day histogram

🖥️ Demo UI (no build tools) and REST API (JSON)

Repo structure
.
├─ src/
│  ├─ api.py                 # FastAPI app, /analyze endpoint, static mounts, CORS
│  ├─ extractors.py          # route/time/place/participant/aspects extraction
│  ├─ place_dict.py          # stop dictionary loaders + fuzzy matching
│  ├─ geocode.py             # (optional) local geocode helpers
│  ├─ advice.py              # KZ recommendations
│  ├─ train_priority.py      # training script for priority model
│  ├─ train_aspect.py        # training script for aspect model
│  ├─ visualize.py           # report PNGs (matplotlib)
│  └─ ... utils/*
├─ demo/
│  └─ index.html             # single-file UI for the API
├─ reports/
│  ├─ aspects_hist.png
│  ├─ priority_over_time.png
│  ├─ routes_top.png
│  └─ time_of_day_hist.png
├─ tests/
│  └─ test_extractors*.py
├─ requirements.txt
├─ config.yml                # optional config (paths, thresholds)
├─ .gitattributes            # Git LFS for models & large assets
├─ .gitignore
└─ README.md

Quick start
1) Environment
python -m venv .venv && source .venv/bin/activate         # or conda
pip install -r requirements.txt


(If you use conda: conda activate base then pip install -r requirements.txt.)

2) Models

Place trained bundles under models/:

models/priority.joblib — dict with keys: clf, classes, vect_word (or vect), optional vect_char, optional base_word

models/aspect_lr.joblib — dict with keys: clf, classes, vect

Large binaries are tracked by Git LFS (*.joblib, *.parquet, *.pkl, reports/*.png).

3) Local stops / dictionaries (optional but recommended)

Put YAML/CSV with stops into data/, e.g.:

data/almaty_stops_with_aliases.yaml

data/astana_stops_with_aliases.yaml

data/stops_kz.csv with columns: name,lat,lon

place_dict.py auto-discovers YAMLs (you can override glob via STOPS_GLOB env).

4) Run API + Demo UI
export API_KEY=                             # optional
export BASIC_USER=                          # optional
export BASIC_PASS=                          # optional

# Run
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --log-level info


CORS is enabled for * by default in api.py

Static mounts:

/demo → demo/ (serves index.html)

/reports → reports/

Open the UI: http://localhost:8000/demo

5) API

Endpoint: POST /analyze

Request

{
  "text": "Алматы, у остановки Сайран валидатор не работает, очередь большая",
  "city_hint": "Almaty"
}


Response (example)

{
  "priority": "medium",
  "probs": {"low": 0.12, "medium": 0.58, "high": 0.22, "critical": 0.08},
  "participant": {"role": "driver", "match": "водитель"},
  "place": {"name": "Сайран", "city_hint": "Almaty", "lat": 43.242, "lon": 76.882, "score": 95, "method": "geocode+fuzzy"},
  "aspect": "payment",
  "recommendation_kz": "Төлем/валидатор: валидаторларды тексеріп, ақаулы құрылғыларды ауыстырыңыз.",
  "explain": {"model_top_tokens": ["валидатор", "очередь", "..."], "rules": []}
}


curl

curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"Маршрут 32 стабильно опаздывает вечером после 19:00"}'

6) Reports / Visualizations

Generate PNGs (examples already in reports/):

Aspect frequency

Priority over time (stacked)

Top routes by complaints

Complaints by time of day

Run (example):

python -m src.visualize --input data/transport_complaints.csv --outdir reports
# or see docstring in src/visualize.py for expected columns


Requirement: no paid APIs. Everything works offline on local data; OSM static map in the UI uses a public embed (can be disabled).

Security & Limits

API Key: set API_KEY env to enforce x-api-key header

Basic Auth: set BASIC_USER/BASIC_PASS

Rate limit: simple in-memory per-IP (default 120 req/min), tunable via RATE_LIMIT_PER_MIN

Training (optional)
# Priority
python -m src.train_priority --train data/train.csv --out models/priority.joblib

# Aspect
python -m src.train_aspect --train data/train.csv --out models/aspect_lr.joblib


Expected columns (example): text, priority, aspect, route, time_hint, city, …

Tests
pytest -q

Deployment notes

Production: run behind a reverse proxy (Caddy/Nginx).

Systemd: use a service file to keep uvicorn alive.

Docker: create a minimal image (not provided here to keep repo light).

FAQ

Q: Can I open the UI from a different domain/port?
A: Yes, CORS is enabled (allow_origins=["*"]). For stricter setups, update the list.

Q: Place isn’t detected?
A: Add/expand your city YAML with name and aliases. The fuzzy matcher uses rapidfuzz if available, else difflib.

Q: What if I only have Russian texts?
A: The pipeline is language-agnostic at the token level and includes mixed-lang patterns.
