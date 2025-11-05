'''location → db_band → goal 3단 중첩 JSON 변환 스크립트'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV → Nested JSON (location > db_band > goal)

입력 CSV 컬럼(요청 사양):
- location | db_band | goal
- bpm_min, bpm_max
- energy_min, energy_max
- mood
- genre_primary, genre_secondary
- vocal
- search_query
- search_url

출력 JSON 구조 예:
{
  "meta": {...},
  "rules": {
    "cafe": {
      "66-80": {
        "focus": {
          "bpm_min": 80,
          "bpm_max": 90,
          "energy_min": 0.4,
          "energy_max": 0.55,
          "mood": "chill",
          "genre_primary": "lo-fi",
          "genre_secondary": "jazz",
          "vocal": "instrumental",
          "search_query": "lo-fi instrumental calm for focus 80-90 bpm -live -remix",
          "search_url": "https://open.spotify.com/search/..."
        }
      }
    }
  }
}
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone, timedelta

REQUIRED_COLUMNS = [
    "location", "db_band", "goal",
    "bpm_min", "bpm_max",
    "energy_min", "energy_max",
    "mood",
    "genre_primary", "genre_secondary",
    "vocal",
    "search_query", "search_url",
]

NUMERIC_HINT_PATTERNS = (
    r"(^|_)bpm(_|$)",           # bpm, bpm_min, bpm_max
    r"(^|_)energy(_|$)",        # energy, energy_min, energy_max
)

def guess_version_from_filename(path: str) -> str:
    m = re.search(r"(v?\d+(?:\.\d+)*)", os.path.basename(path))
    return m.group(1) if m else "v?"

def is_numeric_like(s: str) -> bool:
    if s is None: return False
    s = s.strip()
    if s == "": return False
    try:
        float(s)
        return True
    except ValueError:
        return False

def maybe_cast_value(col_name: str, value: str):
    if value is None: return None
    v = value.strip()
    if v == "": return None

    # bpm/energy 계열은 숫자 우선
    if any(re.search(pat, col_name, re.IGNORECASE) for pat in NUMERIC_HINT_PATTERNS):
        if is_numeric_like(v):
            n = float(v)
            return int(n) if n.is_integer() and "energy" not in col_name.lower() else n
        return v

    # 일반 숫자 모양도 캐스팅 (정수/실수)
    if is_numeric_like(v):
        n = float(v)
        return int(n) if n.is_integer() else n

    # 불리언 문자열 처리
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"

    return v

def norm_db_band(s: str) -> str:
    if s is None:
        return "unknown"
    # 다양한 대시를 ASCII '-'로 통일
    s = re.sub(r"[–—−―]", "-", s)
    # 공백 제거
    s = re.sub(r"\s+", "", s)
    return s

def norm_key(s: str) -> str:
    return (s or "").strip().lower()

def row_to_payload(row: dict, key_fields=("location", "db_band", "goal")) -> dict:
    payload = {}
    for k, v in row.items():
        if k is None:
            continue
        key = k.strip()
        if norm_key(key) in key_fields:
            continue
        payload[key] = maybe_cast_value(key, v)
    return payload

def csv_to_nested_json(csv_path: str, out_path: str = None):
    if out_path is None:
        out_path = os.path.splitext(csv_path)[0] + ".nested.json"

    KST = timezone(timedelta(hours=9))  # Asia/Seoul
    generated_at = datetime.now(tz=KST).isoformat(timespec="seconds")

    rules = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV 헤더를 읽을 수 없습니다.")

        header_lower = [h.strip().lower() for h in reader.fieldnames]
        # 필수 컬럼 검증
        missing = [c for c in REQUIRED_COLUMNS if c not in header_lower]
        if missing:
            raise ValueError(f"입력 CSV에 필수 컬럼 누락: {missing}\n헤더: {reader.fieldnames}")

        for raw_row in reader:
            # 키 통일(lower) + None → ""
            row = {norm_key(k): (v if v is not None else "") for k, v in raw_row.items()}

            location = norm_key(row.get("location", ""))
            db_band  = norm_db_band(row.get("db_band", ""))
            goal     = norm_key(row.get("goal", ""))

            if not location or not db_band or not goal:
                print(f"  키 누락으로 스킵: location='{location}', db_band='{db_band}', goal='{goal}'")
                continue

            rules.setdefault(location, {}).setdefault(db_band, {})
            payload = row_to_payload(row)

            # 중복 키 덮어쓰기 경고
            if goal in rules[location][db_band]:
                print(f"  덮어쓰기: ({location} / {db_band} / {goal})")

            rules[location][db_band][goal] = payload

    meta = {
        "source_csv": os.path.basename(csv_path),
        "version": guess_version_from_filename(csv_path),
        "generated_at": generated_at,
        "keys": ["location", "db_band", "goal"]
    }

    output = {"meta": meta, "rules": rules}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f" 변환 완료: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="CSV → Nested JSON (location > db_band > goal)")
    parser.add_argument("csv", help="입력 CSV 경로 (예: rule_mapping__v2.5.csv)")
    parser.add_argument("-o", "--out", help="출력 JSON 경로 (기본: 입력이름 + .nested.json)")
    args = parser.parse_args()
    csv_to_nested_json(args.csv, args.out)

if __name__ == "__main__":
    main()

'''터미널에 python convert_csv_to_nested_json.py rule_mapping_v2.5.csv
입력하면 → rule_mapping_v2.5.nested.json 생성 '''

