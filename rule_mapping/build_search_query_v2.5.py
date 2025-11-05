
"""
Build Search Query (v2.5)
- Input : rule_mapping_v1.csv
- Output: rule_mapping__v2.5.csv
- Purpose: Generate Spotify-friendly natural-language search queries & links
           from rule-based mapping (goal/decibel/location → music attributes).
- Notes :
  * Uses UTF-8-SIG for Windows/Excel-friendly CSVs.
  * Deduplicates tokens while preserving order.
  * Always adds negative filters to reduce noisy results: -live -remix -karaoke -cover
  * Does not require internet; purely transforms the CSV.
"""

import argparse
import pandas as pd
from urllib.parse import quote

# ---------- Config ----------
DEFAULT_IN  = "rule_mapping_v1.csv"
DEFAULT_OUT = "rule_mapping__v2.5.csv"

GOAL_MAP = {
    "focus": "for focus",
    "reading": "study music",
    "relax": "to relax",
    "active": "workout music",
    "meditate": "meditation music",
    "sleep": "sleep sounds",
    "neutral": "background music",
}

REQUIRED_COLS = [
    "location","db_band","goal",
    "bpm_min","bpm_max",
    "energy_min","energy_max",
    "mood","genre_primary","genre_secondary","vocal"
]

NEGATIVE_FILTERS = ["-live", "-remix", "-karaoke", "-cover"]


# ---------- Helpers ----------
def _safe(s):
    return "" if pd.isna(s) else str(s).strip()

def _float(x):
    try:
        return float(x)
    except Exception:
        return None

def _vocal_tokens(vocal: str):
    """Include only strong signals; skip light/balanced/unknown."""
    v = (vocal or "").lower().strip()
    if not v:
        return []
    if "instrumental" in v:
        return ["instrumental"]
    if "vocal-heavy" in v:
        return ["vocal"]
    if "female" in v:
        return ["female-vocal"]
    if "male" in v:
        return ["male-vocal"]
    # light-vocal / balanced / etc → 생략
    return []

def _energy_hint(e_min, e_max):
    """Average-based 4-level hint: soft / mellow / energetic / intense."""
    try:
        em = float(e_min); ex = float(e_max)
    except Exception:
        return ""
    avg = (em + ex) / 2.0
    if avg <= 0.30:
        return "soft"
    elif avg <= 0.50:
        return "mellow"
    elif avg <= 0.70:
        return "energetic"
    else:
        return "intense"


# ---------- Core ----------
def make_search_query_v2(row: dict) -> str:
    """Create a Spotify-friendly natural-language search query."""
    parts = []

    # 0) Genre tokens (primary → secondary, 동일 값 중복 제거)
    gp = _safe(row.get("genre_primary"))
    gs = _safe(row.get("genre_secondary"))
    if gp:
        parts.append(gp)
    if gs and gs.lower() != gp.lower():
        parts.append(gs)

    # 1) Vocal (조건부 포함)
    parts += _vocal_tokens(_safe(row.get("vocal")))

    # 2) Mood
    mood = _safe(row.get("mood"))
    if mood:
        parts.append(mood)

    # 3) Energy hint (평균값 기반 4단계)
    ehint = _energy_hint(row.get("energy_min"), row.get("energy_max"))
    if ehint:
        parts.append(ehint)

    # 4) Goal → 자연어 표현
    goal_raw = _safe(row.get("goal")).lower()
    parts.append(GOAL_MAP.get(goal_raw, goal_raw or "background music"))

    # 5) BPM hint
    bmin = _float(row.get("bpm_min"))
    bmax = _float(row.get("bpm_max"))
    if bmin is not None and bmax is not None:
        parts.append(f"{int(bmin)}-{int(bmax)} bpm" if int(bmin) != int(bmax) else f"{int(bmin)} bpm")

    # 6) 음질 제외 필터 (항상 부착)
    parts += NEGATIVE_FILTERS

    # 7) de-duplicate while preserving order
    clean, seen = [], set()
    for t in parts:
        t = t.strip()
        if t and t not in seen:
            clean.append(t)
            seen.add(t)

    return " ".join(clean)


def transform(in_path: str, out_path: str) -> pd.DataFrame:
    """Load v1 CSV, generate search_query & search_url, save v2.5 CSV."""
    df = pd.read_csv(in_path, encoding="utf-8-sig")

    # Ensure required columns exist (create blanks if missing to avoid crashes)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    for c in missing:
        df[c] = ""

    # Build search_query and deep links
    df["search_query"] = df.apply(lambda r: make_search_query_v2(r.to_dict()), axis=1)
    df["search_url"]   = df["search_query"].apply(lambda s: "https://open.spotify.com/search/" + quote(s, safe=""))

    # Save output (UTF-8-SIG for Excel compatibility)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def quick_validate(df: pd.DataFrame) -> None:
    """Lightweight validation for ranges; prints warnings instead of raising."""
    try:
        bad_energy = ~(
            (df["energy_min"] >= 0) & (df["energy_min"] <= 1) &
            (df["energy_max"] >= 0) & (df["energy_max"] <= 1) &
            (df["energy_min"] <= df["energy_max"])
        )
        bad_bpm = ~(df["bpm_min"] <= df["bpm_max"])
        if bad_energy.any():
            print(f"⚠️  Energy range warnings on rows: {df.index[bad_energy].tolist()}")
        if bad_bpm.any():
            print(f"⚠️  BPM range warnings on rows: {df.index[bad_bpm].tolist()}")
        if not bad_energy.any() and not bad_bpm.any():
            print("✅ quick_validate: OK")
    except Exception as e:
        print(f"⚠️  quick_validate skipped: {e}")


def main():
    p = argparse.ArgumentParser(description="Build Spotify search queries (v2.5) from rule_mapping_v1.csv")
    p.add_argument("--in",  dest="in_path",  default=DEFAULT_IN,  help="Input CSV (default: rule_mapping_v1.csv)")
    p.add_argument("--out", dest="out_path", default=DEFAULT_OUT, help="Output CSV (default: rule_mapping__v2.5.csv)")
    args = p.parse_args()

    df = transform(args.in_path, args.out_path)
    quick_validate(df)
    print("✅ Done.")
    print(f" - Input : {args.in_path}")
    print(f" - Output: {args.out_path}")
    print(f" - Rows  : {len(df)}")


if __name__ == "__main__":
    main()


