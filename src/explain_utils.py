# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def load_priority_bundle(path: str = "models/priority.joblib"):
    bundle = joblib.load(path)
    return bundle  # {"vect","base","clf","classes"}

def top_features_for_text(text: str, bundle, k: int = 5) -> List[str]:
    vect: TfidfVectorizer = bundle["vect"]
    base = bundle["base"]
    clf  = bundle["clf"]
    X = vect.transform([text])
    pred = clf.predict(X)[0]
    cls_idx = list(base.classes_).index(pred)
    coefs = base.coef_[cls_idx]
    feats = vect.get_feature_names_out()
    top_idx = coefs.toarray().ravel().argsort()[-k:][::-1]
    return [feats[i] for i in top_idx]

def predict_with_probs(text: str, bundle) -> Tuple[str, Dict[str, float]]:
    vect: TfidfVectorizer = bundle["vect"]
    clf  = bundle["clf"]
    classes = list(bundle["classes"])
    X = vect.transform([text])
    pred = clf.predict(X)[0]
    proba = getattr(clf, "predict_proba", None)
    probs = dict(zip(classes, (proba(X)[0].tolist() if proba else [])))
    return str(pred), probs
