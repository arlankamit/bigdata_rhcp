# -*- coding: utf-8 -*-
import regex as re

# Признаки казахских букв
KZ = "ӘәӨөҮүҰұҚқҒғІіҢңҺһ"
RU = "А-Яа-яЁё"

RE_KZ = re.compile(rf"[{KZ}]")
RE_RU = re.compile(rf"[{RU}]")

def token_lang_share(text: str):
    if not text: return {"kk":0.0, "ru":0.0}
    toks = re.findall(r"\p{L}+", text)
    if not toks: return {"kk":0.0, "ru":0.0}
    kk = sum(1 for t in toks if RE_KZ.search(t))
    ru = sum(1 for t in toks if RE_RU.search(t))
    total = max(1, kk+ru)
    return {"kk": kk/total, "ru": ru/total}
