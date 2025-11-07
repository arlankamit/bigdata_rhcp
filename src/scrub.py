import re
PHONE = re.compile(r"(?:\+?7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def scrub(text: str) -> str:
    if not text: return text
    t = PHONE.sub("<phone>", text)
    t = EMAIL.sub("<email>", t)
    return t[:2000]  # safety cap
