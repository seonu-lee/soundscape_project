
import argparse
import pandas as pd
from urllib.parse import quote

DEFAULT_IN = "rule_mapping_v1.csv"
DEFAULT_OUT = "rule_mapping_v2.csv"

def _is_nan(x):
    try:
        return pd.isna(x)
    except Exception:
        return False

def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _tokenize(row):
    parts = []
    for col in ["genre_primary", "genre_secondary"]:
        val = row.get(col, "")
        if val and not _is_nan(val):
            parts.append(str(val).strip())

    goal = row.get("goal", "")
    if goal and not _is_nan(goal):
        g = str(goal).strip().lower()
        goal_map = {
            "focus": "deep focus",
            "reading": "reading",
            "relax": "relax & unwind",
            "active": "workout",
            "meditate": "meditation",
            "sleep": "sleep",
            "neutral": "background music"
        }
        parts.append(goal_map.get(g, g))

    mood = row.get("mood", "")
    if mood and not _is_nan(mood):
        parts.append(str(mood).strip())

    vocal = row.get("vocal", "")
    if vocal and not _is_nan(vocal):
        v = str(vocal).strip().lower()
        if "instrumental" in v:
            parts.extend(["instrumental", "-live", "-remix"])
        elif "no" in v:
            parts.extend(["ambient", "white noise", "-vocal", "-live", "-remix"])
        elif "vocal-heavy" in v:
            parts.extend(["vocal", "energetic"])

    e_min = _as_float(row.get("energy_min"))
    e_max = _as_float(row.get("energy_max"))
    if e_max is not None and e_min is not None:
        if e_max >= 0.65:
            parts.append("energetic")
        elif e_min <= 0.30:
            parts.append("calm")

    bmin = _as_float(row.get("bpm_min"))
    bmax = _as_float(row.get("bpm_max"))
    if bmin is not None and bmax is not None:
        if int(bmin) == int(bmax):
            parts.append(f"{int(bmin)} bpm")
        else:
            parts.append(f"{int(bmin)}-{int(bmax)} bpm")

    cleaned = []
    seen = set()
    for t in parts:
        t = str(t).strip()
        if not t:
            continue
        if t not in seen:
            cleaned.append(t)
            seen.add(t)
    return cleaned

def build_search_query(row):
    tokens = _tokenize(row)
    return " ".join(tokens)

def to_search_url(query: str) -> str:
    base = "https://open.spotify.com/search/"
    return base + quote(query, safe="")

def transform(in_path: str = DEFAULT_IN, out_path: str = DEFAULT_OUT):
    df = pd.read_csv(in_path)
    required = [
        "location","db_band","goal",
        "bpm_min","bpm_max",
        "energy_min","energy_max",
        "mood","genre_primary","genre_secondary","vocal"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = ""

    df["search_query"] = df.apply(lambda r: build_search_query(r.to_dict()), axis=1)
    df["search_url"]   = df["search_query"].apply(to_search_url)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    augmented_v1 = in_path.replace(".csv", "_augmented.csv")
    df.to_csv(augmented_v1, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    p.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    args = p.parse_args()
    transform(args.in_path, args.out_path)
