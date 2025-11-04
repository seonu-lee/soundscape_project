Rule Mapping v2.5 — Spotify Search 기반 추천 규칙 매핑표

파일: rule_mapping__v2.5.csv
최종 업데이트: 2025.11.04

1. 개요

이 파일은 위치(Location), 소음(Decibel), 목표(Goal) 데이터를 기반으로
사용자 상황에 가장 적합한 음악을 추천하기 위한 규칙 기반 매핑표입니다.

Spotify API 권한 제한으로 Spotify의 AI 추천(/recommendations)대신
자체 규칙(rule-based mapping)을 기반으로 검색 기반(/search) 추천 전략을 적용했습니다.
즉, 각 조합별로 Spotify에서 바로 검색 가능한 query와 **링크(URL)**를 생성합니다.

2. 파일 구성
| 컬럼명                                 | 설명                                                                    |
| ----------------------------------- | --------------------------------------------------------------------- |
| `location`                          | 사용자의 위치 (예: cafe, library, office, home, outdoor, subway)             |
| `db_band`                           | 소음 구간 (예: 36–50, 51–65, 66–80, 81+)                                   |
| `goal`                              | 사용자의 심리적 목표 (focus, relax, active, reading, meditate, sleep, neutral) |
| `bpm_min` / `bpm_max`               | 해당 목표의 음악 템포 범위 (beats per minute)                                    |
| `energy_min` / `energy_max`         | 음악의 에너지 강도 (0.0~1.0)                                                  |
| `mood`                              | 음악의 정서적 분위기 (calm / chill / energetic / serene 등)                     |
| `genre_primary` / `genre_secondary` | 주요 장르 키워드 (lo-fi, jazz, edm 등)                                        |
| `vocal`                             | 보컬 비율 (instrumental / light-vocal / no-vocal / vocal-heavy)           |
| `search_query`                      | Spotify 검색 엔진에 최적화된 자연어 기반 검색어                                        |
| `search_url`                        | Spotify 웹에서 바로 열리는 검색 링크 (`https://open.spotify.com/search/...`)      |

3. 검색어 생성 로직 요약 (make_search_query_v2)
모든 행은 다음 규칙으로 검색어(search_query)를 구성합니다:
genre_primary + genre_secondary + [vocal 필터] + mood + [에너지 힌트] + [Goal 자연어 표현] + [BPM 힌트]
예시:
| Goal         | search_query 예시                                            | 의미                 |
| ------------ | ---------------------------------------------------------- | ------------------ |
| **focus**    | `lo-fi instrumental calm for focus 80-90 bpm -live -remix` | 집중용, 조용한 무보컬 Lo-Fi |
| **relax**    | `jazz acoustic chill to relax 60-75 bpm`                   | 휴식용, 부드러운 어쿠스틱 재즈  |
| **active**   | `edm pop energetic workout music 110-120 bpm`              | 활동용, 에너지 높은 음악     |
| **meditate** | `ambient drone meditation music 55-60 bpm`                 | 명상용, 반복적 저주파 사운드   |
| **sleep**    | `ambient white noise sleep sounds 55 bpm`                  | 수면 유도용, 화이트노이즈 중심  |

4. 목표(Goal) 자연어 매핑
| Goal     | Spotify 자연어 표현     | 설명                 |
| -------- | ------------------ | ------------------ |
| focus    | `for focus`        | 집중용, Deep Focus 계열 |
| reading  | `study music`      | 독서/공부 배경음          |
| relax    | `to relax`         | 휴식/이완              |
| active   | `workout music`    | 운동/활동              |
| meditate | `meditation music` | 명상/호흡              |
| sleep    | `sleep sounds`     | 수면 유도용             |
| neutral  | `background music` | 기본형 (fallback)     |

5. 백엔드/프론트 요청 사항

백엔드
이 파일의 search_query 값을 받아서 Spotify Search API 호출
응답 중 playlist를 우선 사용, 없을 경우 track fallback
결과는 title, image_url, external_url 형태로 프론트에 전달.

프론트엔드
받은 데이터를 기반으로 썸네일 카드 UI 구현.
카드 구성 예:
이미지: image_url
제목: title
하단 버튼: “Spotify에서 열기” → external_url

6. 개발자 노트

rule_mapping_v1.csv → 기본 규칙 (BPM/Energy/Mood/Genre/Vocal)

rule_mapping__v2.5.csv → 검색 기반 최적화 버전 (Spotify 자연어 쿼리 반영)

파일 인코딩: utf-8-sig

7. 샘플 코드 

import pandas as pd
df = pd.read_csv("rule_mapping__v2.5.csv", encoding="utf-8-sig")

# 예시 1: 특정 goal별 검색어 확인
print(df[df.goal == "focus"][["location", "search_query"]])

# 예시 2: Spotify 검색 링크 클릭
import webbrowser
webbrowser.open(df.loc[0, "search_url"])

8. 버전 이력

| 버전       | 내용                                             | 작성일            |
| -------- | ---------------------------------------------- | -------------- |
| v1.0     | 규칙 기반 매핑 (단순 문자열 seed)                         | 2025-10        |
| v2.0     | 검색형 쿼리 자동 생성 (`make_search_query`)             | 2025-11        |
| **v2.5** | Spotify 감성형 자연어 쿼리, 불필요 토큰 제거 (-live/-remix 등) | **2025-11-04** |
