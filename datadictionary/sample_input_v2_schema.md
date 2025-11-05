# Sample Data Schema — `sample_input_v2.csv`

본 데이터셋은 **위치(Location)**, **소음(Decibel)**, **사용자 목표(Goal)** 3가지 핵심 입력값을 기반으로,  
규칙 엔진이 추천하는 **음악 속성 및 메타데이터**를 담고 있습니다.  
이 스키마 문서는 데이터 구조 및 필드 정의를 명확히 하기 위한 참고용 문서입니다.

---

##  1. 파일 개요

| 항목 | 내용 |
|:---|:---|
| 파일명 | `sample_input_v2.csv` |
| 포맷 | CSV (UTF-8) |
| 데이터 수 | 88건  |
| 사용 목적 | 규칙 기반 음악 추천 모델(비학습형) 포맷 검증 |
| 주요 입력 요소 | 위치(location), 소음(decibel), 목표(goal) |
| 주요 출력 요소 | 추천 음악 속성 (playlist, mood, bpm, energy, vocal, genre) |

---

##  2. 컬럼 스키마 정의

| 필드명 | 타입 | 예시값 | 설명 |
|:---:|:---:|:---:|:---|
| `location` | `string` | `cafe`, `library`, `subway`, `home`, `office`, `outdoor` | 사용자의 위치 (환경 분류 단위) |
| `decibel` | `int` | `68` | 실시간 소음 세기 (dB 단위) |
| `goal` | `string` | `focus`, `relax`, `meditate`, `sleep`, `active`, `reading`, `neutral` | 사용자의 심리적 목표 |
| `playlist` | `string` | `Lo-Fi Beats`, `Ambient Journey` | 추천 플레이리스트명 (Spotify 시드 기반) |
| `mood` | `string` | `chill`, `calm`, `neutral`, `soothing` | 곡의 감정적 분위기 |
| `bpm` | `int` | `85` | 곡의 속도 (Beats Per Minute) |
| `energy` | `float` | `0.47` | Spotify 오디오 피처 기반 에너지값 (0~1) |
| `vocal` | `string` | `instrumental` / `with_vocal` | 보컬 유무 (집중 vs 활동 구분) |
| `genre` | `string` | `lofi`, `ambient`, `electronic`, `piano`, `acoustic` | 음악 장르 (규칙 매핑 결과) |

---

##  3. 컬럼 관계 요약

| 관계 | 해석 |
|:---|:---|
| `decibel ↔ bpm` | 목표별 소음이 높을수록 BPM이 증가하는 경향 (활동적 환경일수록 빠른 템포) |
| `goal ↔ mood` | 집중/휴식/활동 등 심리 목표에 따라 무드 분포가 명확히 구분됨 |

---

## 4. 파일 관리 정보

| 항목 | 내용 |
|:---|:---|
| 버전 | v2.0 |
| 작성일 | 2025-11 |
| 작성자 | 데이터분석 파트 |
| 용도 | 규칙 엔진 초기 학습·검증용 샘플 |
| 연동 포맷 | JSON 변환 후 백엔드 테스트 (`sample_input_v2.json`) |

---

>  **Note:**  
> 본 스키마는 규칙 엔진 설계의 “데이터 사전(Data Dictionary)” 역할을 하며,  
> 백엔드와의 연동 시 동일한 필드명·타입 구조를 유지해야 합니다.
