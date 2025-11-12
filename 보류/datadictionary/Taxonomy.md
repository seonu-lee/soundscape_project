#  Soundscape — Taxonomy v1  
> 위치(Location) × 데시벨 구간(Decibel Band) × 목표(Goal)를 기반으로 음악 속성 규칙을 설계하기 위한 분류 체계

---

##  1. 위치(Location)
| 분류 | 세부 예시 | 설명 |
|------|------------|------|
| `cafe` | 스타벅스, 투썸 등 | 주변 대화와 배경음이 섞인 중간 소음 환경 |
| `library` | 독서실, 공부방 | 정적·집중 환경 |
| `office` | 사무실, 코워킹 스페이스 | 일정한 백색소음, 업무 중심 환경 |
| `subway` | 지하철, 버스 | 간헐적 진동음·잡음이 있는 이동 환경 |
| `outdoor` | 공원, 거리, 해변 | 바람·자연음 등 불규칙한 야외 환경 |
| `home` | 거실, 방, 베란다 | 개인화된 편안한 환경 |

---

##  2. 데시벨 구간(Decibel Band)
| 구간 | dB 범위 | 환경 예시 | 분류명 |
|------|----------|------------|--------|
| `silent` | 0–35dB | 독서실, 새벽 주택가 | 정적 |
| `quiet` | 36–50dB | 사무실, 도서관 | 조용 |
| `moderate` | 51–65dB | 카페, 식당 | 보통 |
| `loud` | 66–80dB | 거리, 지하철 내부 | 시끄러움 |
| `very_loud` | 81-100dB  | 공연장, 공사장 | 매우 시끄러움 |

---

##  3. 목표(Goal)
| Goal | 설명 | 대표 상황 |
|------|------|------------|
| `focus` | 집중을 유지하거나 업무 효율을 높이기 위한 상태 | 공부, 업무, 코딩 |
| `relax` | 휴식과 안정, 스트레스 완화 | 카페, 퇴근 후 |
| `sleep` | 수면 유도 또는 취침 전 안정 | 야간, 숙면 모드 |
| `active` | 활동적 에너지, 운동/이동 중 사용 | 운동, 출근길 |
| `meditate` | 명상, 감정 정화, 내면 집중 | 마음챙김, 요가 |
| `reading` | 잔잔한 몰입 상태 | 독서, 글쓰기 |
| `neutral` | 목표 미설정 상태 | 일상생활 |

---

##  4. 음악 속성(Music Feature)
| 속성 | 설명 | 예시 값 |
|------|------|----------|
| `bpm_range` | 템포 범위 | 50–60(sleep) / 75–90(focus) / 110–120(active) |
| `energy_range` | 곡의 에너지 지수(0~1) | 0.2~0.8 |
| `mood` | 음악 분위기 | calm / chill / upbeat / ambient |
| `genre_primary` | 기본 장르 | lo-fi / pop / edm / jazz / ambient |
| `genre_secondary` | 보조 장르(선택) | soul / funk / acoustic 등 |
| `vocal` | 보컬 포함 여부 | instrumental / light-vocal / vocal-heavy |

---

##  5. 규칙 매핑 예시
| 입력(Location × Decibel × Goal) | 출력(Music Feature) |
|----------------------------------|----------------------|
| `cafe × 70dB × focus` | BPM: 80–90 / energy: 0-1/ Mood: chill / Genre: lo-fi / Vocal: instrumental |
| `library × 40dB × relax` | BPM: 60–70 / energy: 0-1/ Mood: calm / Genre: piano / Vocal: instrumental |
| `subway × 78dB × active` | BPM: 110–120 / energy: 0-1/ Mood: energetic / Genre: pop-edm / Vocal: vocal-heavy |
| `home × 45dB × sleep` | BPM: 50–60 / energy: 0-1/ Mood: ambient / Genre: ambient / Vocal: instrumental |

---

