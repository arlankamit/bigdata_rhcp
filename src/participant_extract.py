# -*- coding: utf-8 -*-
import re
from typing import Optional, Dict
from unidecode import unidecode
from .participants_lexicon import LEXICON

def _norm(s: str) -> str:
    s = unidecode((s or "").lower())
    s = re.sub(r"[^a-z0-9а-яёқңғүұіһәө\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_participant(text: str) -> Optional[Dict]:
    t = _norm(text)
    if not t:
        return None
    for role, words in LEXICON.items():
        for w in words:
            if re.search(rf"\b{re.escape(_norm(w))}\b", t):
                return {"role": role, "match": w}
    return None
