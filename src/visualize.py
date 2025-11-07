# -*- coding: utf-8 -*-
"""
routes_top.png, aspects_hist.png, priority_over_time.png, time_of_day_hist.png, participants_hist.png
"""
import os, pandas as pd, matplotlib.pyplot as plt, re
from collections import Counter
from .utils import load_config, ensure_dir

def plot_top_routes(df, outdir):
    if "route" not in df.columns: return
    c = Counter(df["route"].dropna().astype(str))
    if not c: return
    names, vals = zip(*c.most_common(15))
    plt.figure(figsize=(9,6))
    plt.barh(list(names)[::-1], list(vals)[::-1])
    plt.title("Top routes by complaints")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "routes_top.png"))
    plt.close()

def plot_aspects(df, outdir):
    src = None
    if "aspect" in df.columns and df["aspect"].notna().any():
        src = df["aspect"].astype(str).tolist()
    elif "aspects_rule" in df.columns and df["aspects_rule"].notna().any():
        src = [a[0] if isinstance(a, list) and a else "other" for a in df["aspects_rule"]]
    if not src: return
    c = Counter(src)
    names, vals = zip(*c.most_common())
    plt.figure(figsize=(9,6))
    plt.bar(names, vals)
    plt.xticks(rotation=30, ha="right")
    plt.title("Aspect frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "aspects_hist.png"))
    plt.close()

def plot_priority_over_time(df, outdir):
    if "priority" not in df.columns or "created_at" not in df.columns: return
    d = df.copy()
    d["created_at"] = pd.to_datetime(d["created_at"], errors="coerce")
    d["date"] = d["created_at"].dt.date
    pvt = d.pivot_table(index="date", columns="priority", values="text", aggfunc="count").fillna(0)
    ax = pvt.plot(kind="bar", stacked=True, figsize=(12,6))
    ax.set_ylabel("count")
    ax.set_title("Priority over time")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "priority_over_time.png"))
    plt.close()

def plot_time_of_day(df, outdir):
    d = df.copy()
    def hour_from_row(r):
        for col in ["time", "time_extracted"]:
            v = r.get(col)
            if isinstance(v, str) and re.match(r"^([01]?\d|2[0-3]):[0-5]\d$", v):
                return int(v.split(":")[0])
        if "created_at" in r and pd.notna(r["created_at"]):
            try: return pd.to_datetime(r["created_at"]).hour
            except Exception: return None
        return None
    hours = d.apply(hour_from_row, axis=1)
    bins = {"night":0, "morning":0, "day":0, "evening":0}
    for h in hours.dropna():
        h = int(h)
        if 0 <= h < 6: bins["night"] += 1
        elif 6 <= h < 12: bins["morning"] += 1
        elif 12 <= h < 18: bins["day"] += 1
        else: bins["evening"] += 1
    if sum(bins.values()) == 0: return
    names, vals = list(bins.keys()), list(bins.values())
    plt.figure(figsize=(8,5))
    plt.bar(names, vals)
    plt.title("Complaints by time of day")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_of_day_hist.png"))
    plt.close()

def plot_participants(df, outdir):
    if "participant" not in df.columns: return
    c = Counter([p for p in df["participant"].dropna()])
    if not c: return
    names, vals = zip(*c.most_common())
    plt.figure(figsize=(8,5))
    plt.bar(names, vals)
    plt.title("Complaints by participant")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "participants_hist.png"))
    plt.close()

def main():
    cfg = load_config()
    df = pd.read_parquet(cfg["data"]["processed_parquet"])
    out = cfg["visualization"]["out_dir"]
    ensure_dir(out)
    plot_top_routes(df, out)
    plot_aspects(df, out)
    plot_priority_over_time(df, out)
    plot_time_of_day(df, out)
    plot_participants(df, out)
    print(f"[viz] saved charts to {out}")

if __name__ == "__main__":
    main()
