# -*- coding: utf-8 -*-
import argparse, random, re, pathlib, pandas as pd
from typing import List
from .utils import load_config

SEED = 42
random.seed(SEED)

KEYMAP = {
    # соседние клавиши (qwerty/рус/kk)
    "а": "фся", "с": "взы", "о": "лпщ", "е": "уку", "н": "гшть", "т": "ерь",
    "р": "еот", "у": "гек", "к": "лдж", "і": "қй", "қ": "өі", "ө": "қп",
}

def _typo(s: str) -> str:
    if not s: return s
    i = random.randrange(len(s))
    ch = s[i].lower()
    repl_pool = KEYMAP.get(ch, ch)
    repl = random.choice(list(repl_pool)) if isinstance(repl_pool, str) else ch
    return s[:i] + repl + s[i+1:]

def _drop_char(s: str) -> str:
    if len(s) < 2: return s
    i = random.randrange(len(s))
    return s[:i] + s[i+1:]

def _swap_adjacent(s: str) -> str:
    if len(s) < 2: return s
    i = random.randrange(len(s)-1)
    return s[:i] + s[i+1] + s[i] + s[i+2:]

OPS = [_typo, _drop_char, _swap_adjacent]

def augment_text(t: str, n: int = 2) -> str:
    out = t or ""
    for _ in range(n):
        out = random.choice(OPS)(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.10, help="доля шумных сэмплов (0..1)")
    args = ap.parse_args()

    cfg = load_config()
    parquet_path = pathlib.Path(cfg["data"]["processed_parquet"])
    df = pd.read_parquet(parquet_path)

    text_col = "text_clean" if "text_clean" in df.columns else ("text" if "text" in df.columns else None)
    if text_col is None:
        raise SystemExit("No 'text' or 'text_clean' column in dataset")

    k = max(1, int(args.frac * len(df)))
    aug = df.sample(n=k, random_state=SEED).copy()
    aug[text_col] = aug[text_col].map(lambda s: augment_text(s, n=2))

    out = pd.concat([df, aug], axis=0).reset_index(drop=True)
    out.to_parquet(parquet_path, index=False)
    print(f"[augment] added {k} noisy rows using column '{text_col}' -> {parquet_path}")

if __name__ == "__main__":
    main()
