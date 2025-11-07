# -*- coding: utf-8 -*-
import argparse, pathlib, numpy as np, pandas as pd, joblib
from collections import Counter
from scipy.sparse import hstack, vstack
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

from .utils import load_config

SEED = 42

def _oversample_minorities(X, y, factor=1.0):
    """
    Class-wise target: для каждого класса c с текущим n_c
    таргет = min(M, ceil(n_c * factor)), где M — размер мажорного класса.
    Дублируем до таргета. При factor<=1 — ничего не делаем.
    """
    if factor <= 1.0:
        return X, y
    y = np.asarray(y)
    cnt = Counter(y)
    M = max(cnt.values())
    X_parts, y_parts = [X], [y]
    for cls, n in cnt.items():
        target = min(M, int(np.ceil(n * factor)))
        add = max(0, target - n)
        if add > 0:
            idx = np.where(y == cls)[0]
            take = np.random.RandomState(SEED).choice(idx, size=add, replace=True)
            X_parts.append(X[take])
            y_parts.append(y[take])
    return vstack(X_parts), np.concatenate(y_parts)

def _save_hardcases(texts, y_true, y_pred, proba, classes, out_csv):
    """
    Сохраняем все предсказания на тесте, помечаем ошибки и уверенность.
    Удобно фильтровать потом: correct==False (FP/FN), смотреть conf.
    """
    df = pd.DataFrame({
        "text": texts,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    if proba is not None:
        conf = proba.max(axis=1)
        df["conf"] = conf
        # ещё можно сохранить топ-2 классов при желании
    else:
        df["conf"] = np.nan
    df["correct"] = (df["y_true"] == df["y_pred"])
    pathlib.Path("reports").mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=0, help="1 = подбирать C по сетке")
    ap.add_argument("--grid_cs", type=str,
                    default="0.25,0.5,0.75,1.0,1.5,2.0,4.0",
                    help="список C через запятую для грида")
    ap.add_argument("--oversample", type=float, default=1.0,
                    help=">1.0 = дублирование минорных классов (class-wise target)")
    ap.add_argument("--calib_split", type=float, default=0.15,
                    help="доля на калибровку (prefit)")
    args = ap.parse_args()

    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["processed_parquet"])

    # таргет и тексты
    y = df["priority"].astype(str).values
    texts_raw = (df["text_clean"] if "text_clean" in df.columns else df["text"]).astype(str).values

    # word + char (char n-grams уже помогают на шумных коротких жалобах)
    vect_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    vect_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, sublinear_tf=True)

    Xw = vect_word.fit_transform(texts_raw)
    Xc = vect_char.fit_transform(texts_raw)
    X_all = hstack([Xw, Xc], format="csr")

    # держим тексты при сплите — пригодятся для hardcases
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X_all, y, texts_raw, test_size=0.2, random_state=SEED, stratify=y
    )

    # для explain: базовый LinearSVC на словах
    base_word = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED)
    base_word.fit(Xw, y)

    # отдельный калибровочный сплит (prefit-калибровка → нет падений из-за CV)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=args.calib_split, random_state=SEED, stratify=y_train
    )

    # oversample на train_inner
    X_tr_os, y_tr_os = _oversample_minorities(X_tr, y_tr, factor=args.oversample)

    # grid по C (расширенный список)
    best_C = 1.0
    if args.grid:
        Cs = [float(c) for c in args.grid_cs.split(",")]
        inner = LinearSVC(class_weight="balanced", random_state=SEED)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        gs = GridSearchCV(inner, {"C": Cs}, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=0)
        gs.fit(X_tr_os, y_tr_os)
        best_C = float(gs.best_params_["C"])
        print(f"[grid] best C={best_C} (f1_macro={gs.best_score_:.4f})")

    # финальная модель на train_inner_os
    inner = LinearSVC(class_weight="balanced", C=best_C, random_state=SEED)
    inner.fit(X_tr_os, y_tr_os)

    # калибруем на отложенной части (sigmoid — стабильнее для малых классов)
    clf = CalibratedClassifierCV(inner, method="sigmoid", cv="prefit")
    clf.fit(X_cal, y_cal)

    # отчёт
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # proba для hardcases (может быть недоступно, но у CalibratedClassifierCV есть)
    proba = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)

    # сохраняем «тяжёлые» кейсы и confusion
    pathlib.Path("reports").mkdir(exist_ok=True)
    _save_hardcases(
        texts=t_test,
        y_true=y_test,
        y_pred=y_pred,
        proba=(proba if proba is not None else None),
        classes=np.unique(y),
        out_csv="reports/priority_hardcases.csv",
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y)).to_csv(
        "reports/priority_confusion.csv", encoding="utf-8"
    )
    # краткий текстовый отчёт (для архива защиты)
    with open("reports/priority_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred))

    # сохраняем бандл для API
    pathlib.Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {
            "vect_word": vect_word,
            "vect_char": vect_char,
            "base_word": base_word,  # для explain
            "clf": clf,              # calibr. (sigmoid)
            "classes": np.unique(y),
        },
        "models/priority.joblib",
    )
    print("[save] models/priority.joblib")

if __name__ == "__main__":
    main()
