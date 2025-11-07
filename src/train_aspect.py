# -*- coding: utf-8 -*-
import argparse, pathlib, re, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from .utils import load_config
from .constants import ASPECT_PATTERNS

SEED = 42

def _compile_patterns():
    comp = {}
    for asp, pats in ASPECT_PATTERNS.items():
        comp[asp] = [re.compile(p, flags=re.I) for p in pats]
    return comp

def _mask_rules(s: str, compiled):
    t = s
    for asp, pats in compiled.items():
        for p in pats:
            t = p.sub(" [MASK] ", t)
    return re.sub(r"\s+", " ", t).strip()

def _has_rule_hit(s: str, compiled):
    for asp, pats in compiled.items():
        for p in pats:
            if p.search(s):
                return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", type=int, default=0, help="1=маскировать срабатывания правил")
    ap.add_argument("--drop_rule_hits_train", type=int, default=0, help="1=выкинуть из train строки, где сработали правила")
    args = ap.parse_args()

    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["processed_parquet"])
    y = df["aspect"].astype(str).values
    text_col = "text_clean" if "text_clean" in df.columns else "text"
    texts_raw = df[text_col].astype(str).values

    compiled = _compile_patterns()

    # первичная маска (если надо)
    texts = np.array([_mask_rules(t, compiled) for t in texts_raw]) if args.mask else texts_raw

    X_train, X_test, y_train, y_test, raw_tr, raw_te = train_test_split(
        texts, y, texts_raw, test_size=0.2, random_state=SEED, stratify=y
    )

    # опционально — выбрасываем rule-hits из train (но НЕ из test)
    if args.drop_rule_hits_train:
        keep = [not _has_rule_hit(r, compiled) for r in raw_tr]
        X_train, y_train = X_train[keep], y_train[keep]

    vect = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,5), min_df=3, sublinear_tf=True
    )  # char_wb обычно «честнее» к утечкам
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)

    # лёгкий грид по C
    param_grid = {"C": [0.5, 1.0, 2.0]}
    base = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=200, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    gs = GridSearchCV(base, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(Xtr, y_train)
    clf = gs.best_estimator_

    y_pred = clf.predict(Xte)
    print(classification_report(y_test, y_pred))

    pathlib.Path("models").mkdir(exist_ok=True)
    joblib.dump({"vect": vect, "clf": clf, "classes": np.unique(y)}, "models/aspect_lr.joblib")
    print("[save] models/aspect_lr.joblib")

if __name__ == "__main__":
    main()
