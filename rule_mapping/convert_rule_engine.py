#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule Engine for Soundscape
- 입력: location(str), decibel(float/int), goal(str)
- 규칙 소스: CSV→중첩JSON 변환 산출물 (location > db_band > goal)
- 출력: 추천 payload(dict) + 매칭 메타데이터

특징
1) dB 구간 파싱: "36-50", "66–80"(엔대시), "81+" 등 자동 파싱/매칭
2) 매칭 우선순위:
   (A) location / band / goal
   (B) location / band / neutral
   (C) default  / band / goal
   (D) default  / band / neutral
   (E) global   / default
3) 밴드 미포함 decibel → "가장 가까운 밴드"로 스냅(중심값 거리 최소)
4) 신뢰도 점수(confidence)와 경고 메시지(warnings) 제공
5) CLI 테스트 지원
"""

import json
import re
import math
import argparse
from typing import Dict, Tuple, Optional, Any

# -----------------------------
# 유틸
# -----------------------------
DASH_RE = re.compile(r"[–—−―]")  # 다양한 대시
NUM_RE = re.compile(r"^\s*\d+(\.\d+)?\s*$")

def norm_key(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def norm_band_key(s: Optional[str]) -> str:
    if s is None: 
        return "unknown"
    s = DASH_RE.sub("-", s)            # 엔대시 등 통일
    s = re.sub(r"\s+", "", s)          # 공백 제거
    return s

def parse_band(band: str) -> Tuple[float, float]:
    """
    "36-50" -> (36, 50)
    "81+"   -> (81, +inf)
    "0-35"  -> (0, 35)
    """
    b = norm_band_key(band)
    if b.endswith("+"):
        lo = float(b[:-1])
        return lo, float("inf")
    if "-" in b:
        lo, hi = b.split("-", 1)
        return float(lo), float(hi)
    # 안전장치: 숫자 하나만 오면 점으로 취급
    if NUM_RE.match(b):
        v = float(b)
        return v, v
    return float("-inf"), float("inf")

def band_contains(band: str, d: float) -> bool:
    lo, hi = parse_band(band)
    # 경계값 포함 규칙: 닫힌 구간 [lo, hi]
    return (d >= lo) and (d <= hi)

def band_center(band: str) -> float:
    lo, hi = parse_band(band)
    if math.isinf(hi):
        # 상한 무한대일 때, 중심을 lo+10 같은 유도값으로 근사
        return lo + 10.0
    return (lo + hi) / 2.0

# -----------------------------
# Rule Engine
# -----------------------------
class RuleEngine:
    def __init__(self, rules_json: Dict[str, Any]):
        """
        rules_json: CSV→중첩JSON 산출물 전체(dict)
          {
            "meta": {...},
            "rules": {
              "<location>": {
                "<db_band>": {
                  "<goal>": { ...payload... }
                }
              },
              "default": { ... }  # (있으면 사용)
            }
          }
        """
        self.meta = rules_json.get("meta", {})
        self.rules = rules_json.get("rules", {})

        # 키 정규화(소문자) + 밴드 키 통일
        self._normalize_inplace()

    def _normalize_inplace(self):
        normalized = {}
        for loc, bands in self.rules.items():
            nloc = norm_key(loc)
            if not isinstance(bands, dict):
                continue
            nbands = {}
            for band, goals in bands.items():
                nband = norm_band_key(band)
                if not isinstance(goals, dict):
                    continue
                ngoals = {}
                for goal, payload in goals.items():
                    ngoals[norm_key(goal)] = payload
                nbands[nband] = ngoals
            normalized[nloc] = nbands
        self.rules = normalized

    # ---------- 공개 API ----------
    def recommend(self, location: str, decibel: float, goal: str) -> Dict[str, Any]:
        """
        반환 예시:
        {
          "version": "v2.5",
          "input": {"location":"cafe","decibel":72,"goal":"focus"},
          "matched": {"location":"cafe","db_band":"66-80","goal_used":"focus","level":"A"},
          "confidence": 1.0,
          "rule": {...payload...},
          "warnings": []
        }
        """
        warnings = []
        loc = norm_key(location)
        gl  = norm_key(goal)

        # 1) 사용할 밴드 후보 계산
        band = self._select_band(loc, decibel, warnings)

        # 2) 매칭 우선순위
        # A. location / band / goal
        rule, level, goal_used = self._get(loc, band, gl)
        if rule:
            return self._mk_result(loc, decibel, gl, band, goal_used, "A", 1.0, rule, warnings)

        # B. location / band / neutral
        rule, level, goal_used = self._get(loc, band, "neutral")
        if rule:
            return self._mk_result(loc, decibel, gl, band, goal_used, "B", 0.9, rule, warnings)

        # C. default / band / goal
        rule, level, goal_used = self._get("default", band, gl)
        if rule:
            return self._mk_result(loc, decibel, gl, band, goal_used, "C", 0.8, rule, warnings)

        # D. default / band / neutral
        rule, level, goal_used = self._get("default", band, "neutral")
        if rule:
            return self._mk_result(loc, decibel, gl, band, goal_used, "D", 0.7, rule, warnings)

        # E. global default (rules["default"]["default"]["neutral"]) 같은 최후 수단 탐색
        rule = self._get_global_default()
        if rule:
            warnings.append("used global default rule")
            return self._mk_result(loc, decibel, gl, "default", "neutral", "E", 0.6, rule, warnings)

        # 실패
        return {
            "version": self.meta.get("version"),
            "input": {"location": location, "decibel": decibel, "goal": goal},
            "matched": None,
            "confidence": 0.0,
            "rule": None,
            "warnings": warnings + ["no matching rule found"]
        }

    # ---------- 내부 도우미 ----------
    def _get(self, loc: str, band: str, goal: str) -> Tuple[Optional[dict], str, str]:
        loc_map = self.rules.get(norm_key(loc))
        if not loc_map:
            return None, "", goal
        band_map = loc_map.get(norm_band_key(band))
        if not band_map:
            return None, "", goal
        rule = band_map.get(norm_key(goal))
        return rule, "", goal

    def _get_global_default(self) -> Optional[dict]:
        loc_map = self.rules.get("default")
        if not loc_map:
            return None
        # 우선순위 정하기 어려우므로, 하나라도 있으면 그중 neutral/첫 규칙 반환
        # 1) default > any band > neutral
        for band, goals in loc_map.items():
            if "neutral" in goals:
                return goals["neutral"]
        # 2) default > first band > first goal
        for band, goals in loc_map.items():
            for g, payload in goals.items():
                return payload
        return None

    def _select_band(self, loc: str, d: float, warnings: list) -> str:
        """
        location에 정의된 밴드 중:
          - d가 포함되는 밴드 우선
          - 없으면 중심값이 가장 가까운 밴드 선택(스냅)
        location에 밴드가 없으면, default의 밴드 풀로 동일 로직 수행
        둘 다 없으면 "default" 반환
        """
        loc_map = self.rules.get(norm_key(loc))
        band_pool = None
        source_loc = loc
        if isinstance(loc_map, dict) and loc_map:
            band_pool = list(loc_map.keys())
        else:
            def_map = self.rules.get("default", {})
            band_pool = list(def_map.keys()) if def_map else None
            source_loc = "default"

        if not band_pool:
            warnings.append("no bands defined; using 'default'")
            return "default"

        # 포함되는 밴드 먼저 탐색
        for b in band_pool:
            if band_contains(b, float(d)):
                return b

        # 스냅: 중심값 가장 가까운 밴드
        best_b = None
        best_dist = float("inf")
        for b in band_pool:
            c = band_center(b)
            dist = abs(float(d) - c)
            if dist < best_dist:
                best_dist = dist
                best_b = b
        warnings.append(f"decibel {d} not in any band of '{source_loc}', snapped to '{best_b}'")
        return best_b

    def _mk_result(self, loc, decibel, goal, band, goal_used, level, conf, rule, warnings):
        return {
            "version": self.meta.get("version"),
            "input": {"location": loc, "decibel": decibel, "goal": goal},
            "matched": {
                "location": loc,
                "db_band": band,
                "goal_used": goal_used,
                "level": level
            },
            "confidence": conf,
            "rule": rule,
            "warnings": warnings
        }

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Soundscape Rule Engine")
    parser.add_argument("rules_json", help="중첩 JSON 경로 (예: rule_mapping_v2.5.nested.json)")
    parser.add_argument("--location", "-l", required=True, help="예: cafe, library, office, home, outdoor, subway")
    parser.add_argument("--decibel", "-d", required=True, type=float, help="실수/정수 dB 값 (예: 72)")
    parser.add_argument("--goal", "-g", required=True, help="focus, relax, active, reading, meditate, sleep, neutral ...")
    args = parser.parse_args()

    with open(args.rules_json, "r", encoding="utf-8") as f:
        rules_json = json.load(f)

    engine = RuleEngine(rules_json)
    result = engine.recommend(args.location, args.decibel, args.goal)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

''' python convert_rule_engine.py rule_mapping_v2.5.nested.json -l cafe -d 72 -g focus
'''