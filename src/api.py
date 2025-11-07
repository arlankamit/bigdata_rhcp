# -*- coding: utf-8 -*-
import os, time, joblib
from typing import Dict, Optional
from collections import defaultdict, deque

from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# логирование: если есть logging_conf — используем, иначе noop
try:
    from .logging_conf import setup_logging
except Exception:
    def setup_logging(): pass

from .extractors import extract_place_struct, extract_participant

import structlog
logger = structlog.get_logger(__name__)

# -----------------------------------------------------------------------------
# FastAPI + logging
# -----------------------------------------------------------------------------
app = FastAPI(title="haka-analyzer")
setup_logging()

# (опционально) CORS, если UI будет на другом домене/порте
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статика с графиками (если каталога нет — не падаем)
try:
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")
except Exception:
    pass

# статика с html=True, чтобы /demo сразу отдавал index.html
try:
    app.mount("/demo", StaticFiles(directory="demo", html=True), name="demo")
except Exception:
    pass

# -----------------------------------------------------------------------------
# Security (API key и Basic — опциональны)
# -----------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "")
security = HTTPBasic(auto_error=False)
BASIC_USER = os.getenv("BASIC_USER", "")
BASIC_PASS = os.getenv("BASIC_PASS", "")

def _check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _check_basic(creds: Optional[HTTPBasicCredentials]):
    if (BASIC_USER or BASIC_PASS) and (
        creds is None or creds.username != BASIC_USER or creds.password != BASIC_PASS
    ):
        raise HTTPException(status_code=401, detail="Invalid basic auth")

# -----------------------------------------------------------------------------
# Простенький rate limit per IP
# -----------------------------------------------------------------------------
_hits: dict[str, deque] = defaultdict(deque)
_MAX_REQ_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))
def _rate_limit(ip: str):
    now = time.time()
    dq = _hits[ip]; dq.append(now)
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) > _MAX_REQ_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# -----------------------------------------------------------------------------
# PRIORITY model bundle (новый формат: word + char)
# -----------------------------------------------------------------------------
PRIORITY = joblib.load("models/priority.joblib")
_vect_word: TfidfVectorizer = PRIORITY.get("vect_word") or PRIORITY.get("vect")
_base_word = PRIORITY.get("base_word")  # базовый LinearSVC для explain (может быть None)
_vect_char: TfidfVectorizer = PRIORITY.get("vect_char")
_clf  = PRIORITY["clf"]
_classes = list(PRIORITY["classes"])

def _to_features(text: str):
    Xw = _vect_word.transform([text]) if _vect_word else None
    Xc = _vect_char.transform([text]) if _vect_char else None
    if Xw is not None and Xc is not None:
        return hstack([Xw, Xc], format="csr")
    return Xw or Xc

def _top_features_for_text(text: str, k: int = 8):
    # fallback по TF-IDF из самого текста (word-векторизатор)
    def _fallback():
        Xw = _vect_word.transform([text]) if _vect_word else None
        if Xw is None: return []
        feats = _vect_word.get_feature_names_out()
        idx = Xw.nonzero()[1]
        if idx.size == 0: return []
        weights = Xw.data
        order = weights.argsort()[-k:][::-1]
        return [feats[idx[i]] for i in order]

    try:
        if not (_base_word and hasattr(_base_word, "coef_") and _vect_word):
            return _fallback()
        feats = _vect_word.get_feature_names_out()
        pred = _clf.predict(_to_features(text))[0]
        cls_idx = list(getattr(_base_word, "classes_")).index(pred)
        import numpy as np
        top_idx = np.argsort(_base_word.coef_[cls_idx])[-k:][::-1]
        return [feats[i] for i in top_idx]
    except Exception:
        return _fallback()

def _predict_with_probs(text: str):
    X = _to_features(text)
    pred = _clf.predict(X)[0]
    proba = getattr(_clf, "predict_proba", None)
    probs = dict(zip(_classes, (proba(X)[0].tolist() if proba else [])))
    return str(pred), probs

# -----------------------------------------------------------------------------
# ASPECT model (single-label)
# -----------------------------------------------------------------------------
_ASPECT = None
try:
    _ASPECT = joblib.load("models/aspect_lr.joblib")
    _aspect_vect: TfidfVectorizer = _ASPECT["vect"]
    _aspect_clf = _ASPECT["clf"]
    _aspect_labels = list(_ASPECT["classes"])
except Exception:
    _ASPECT = None
    _aspect_vect = None
    _aspect_clf = None
    _aspect_labels = []

def _predict_aspect(text: str) -> Optional[str]:
    if not (_ASPECT and _aspect_vect and _aspect_clf):
        return None
    X = _aspect_vect.transform([text])
    return str(_aspect_clf.predict(X)[0])

# -----------------------------------------------------------------------------
# Простая рекомендация на казахском
# -----------------------------------------------------------------------------
def recommend_kz(aspect: Optional[str], priority: str) -> str:
    a = (aspect or "").lower()
    if priority == "critical" or a == "safety":
        return "Қауіпсіздікке қатысты шағым: жедел тексеріс жүргізіп, қауіпсіздікті қамтамасыз етіңіз."
    if a == "crowding":
        return "Толы автобус: шығыс уақыттарында қосымша рейстер қосып, интервалды азайтыңыз."
    if a == "punctuality":
        return "Кешігу: маршрут кестесін қайта қарап, диспетчерлеуді күшейтіңіз."
    if a == "staff_behavior":
        return "Қызметкерлердің тәртібі: қызметтік нұсқаулық бойынша түсіндіру жұмыстарын жүргізіңіз."
    if a == "vehicle_condition":
        return "Көліктің техникалық жағдайы: техникалық қарап-тексеруді жеделдетіңіз."
    if a == "payment":
        return "Төлем/валидатор: валидаторларды тексеріп, ақаулы құрылғыларды ауыстырыңыз."
    return "Жалпы ұсыныс: шағымды тіркеп, маршрут бойынша жоспарлы тексеріс жасаңыз."

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    city_hint: Optional[str] = Field(None, description="Astana/Almaty")

class AnalyzeResponse(BaseModel):
    priority: str
    probs: Dict[str, float] | None = None
    participant: Dict | None = None
    place: Dict | None = None
    explain: Dict | None = None
    aspect: str | None = None
    recommendation_kz: str | None = None

# -----------------------------------------------------------------------------
# /analyze
# -----------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    req: AnalyzeRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
    creds: Optional[HTTPBasicCredentials] = Depends(security)
):
    _check_api_key(x_api_key)
    _check_basic(creds)
    _rate_limit(request.client.host if request.client else "unknown")

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    if len(text) > 5000:
        raise HTTPException(status_code=413, detail="Text too long")

    participant = extract_participant(text)
    place_geo = extract_place_struct(text)
    pr, probs = _predict_with_probs(text)
    asp = _predict_aspect(text)
    top_feats = _top_features_for_text(text, k=8)
    rec = recommend_kz(asp, pr)

    logger.info(
        "analyze",
        ip=str(request.client.host),
        pr=pr,
        place=(place_geo or {}).get("name"),
        participant=(participant or {}).get("role"),
    )

    return AnalyzeResponse(
        priority=pr,
        probs=probs,
        participant=participant,
        place=place_geo,
        aspect=asp,
        recommendation_kz=rec,
        explain={"model_top_tokens": top_feats, "rules": []},
    )

# -----------------------------------------------------------------------------
# Простой демо-UI
# -----------------------------------------------------------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html><html lang="kk"><meta charset="utf-8">
<title>Haka Demo</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:24px;max-width:980px}
textarea{width:100%;height:120px}
pre{background:#111;color:#eee;padding:12px;border-radius:8px;white-space:pre-wrap}
.row{display:flex;gap:12px;flex-wrap:wrap}
.card{flex:1 1 300px;border:1px solid #ddd;border-radius:12px;padding:12px}
img{max-width:100%;border-radius:8px}
button{padding:10px 16px;border-radius:10px;border:1px solid #888;cursor:pointer}
</style>
<h1>Халық шағымдарын талдау — демо</h1>
<p>Кез келген шағым мәтінін енгізіңіз (қазақша/орысша).</p>
<textarea id="t"></textarea><br>
<button onclick="run()">Analyze</button>
<pre id="out">↳ Нәтиже осында шығады</pre>
<div class="row">
  <div class="card"><h3>Маршруттар (TOP)</h3><img src="/reports/routes_top.png" onerror="this.replaceWith(document.createTextNode('Нет отчёта'))"></div>
  <div class="card"><h3>Аспекттер</h3><img src="/reports/aspects_hist.png" onerror="this.replaceWith(document.createTextNode('Нет отчёта'))"></div>
  <div class="card"><h3>Время суток</h3><img src="/reports/time_of_day_hist.png" onerror="this.replaceWith(document.createTextNode('Нет отчёта'))"></div>
  <div class="card"><h3>Confusion (priority)</h3><img src="/reports/priority_confusion.png" onerror="this.replaceWith(document.createTextNode('Нет отчёта'))"></div>
</div>
<script>
async function run(){
  const text = document.getElementById('t').value;
  const r = await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j,null,2);
}
</script>
"""
