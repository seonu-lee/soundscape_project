import operator
import json
import random
import os
import requests
from typing import Annotated, List, Tuple, Union, Literal, Optional, Dict, Any
from typing_extensions import TypedDict
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

# 0. 설정 로드
load_dotenv()


# Last.fm API Key
LASTFM_API_KEY = "2a3e04f203f980869fbd6d63c12cd96b" # 실제 키 사용
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

# OpenAI 설정

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# =========================================================
# 1. 데이터 모델 정의 (Agent State)
# =========================================================

# 사용자 입력 데이터 구조
class UserContext(TypedDict):
    location: str
    decibel_level: str
    goal: str
    current_time: str

class UserPreference(TypedDict):
    preferred_genres: List[str]
    preferred_artists: List[str]

# LangGraph 상태 (State)
class AgentState(TypedDict):
    # 입력
    user_context: UserContext
    user_preference: UserPreference
    
    # 내부 변수
    gpt_strategy: Dict[str, Any]       # GPT가 짠 전략 (Tag, Reasoning, Features)
    search_queries: List[str]          # Last.fm에 던질 실제 검색어 리스트
    candidate_tracks: List[Dict]       # 수집된 후보곡
    final_tracks: List[Dict]           # 최종 선정된 10곡
    
    # 평가용 메타데이터 (Reasoning 등 보존용)
    recommendation_meta: Dict[str, Any]

# =========================================================
# 2. 도구 함수 (Last.fm API & Helper)
# =========================================================

def call_lastfm_api(tag: str, limit: int = 50) -> List[Dict]:
    """Last.fm API 호출하여 태그 기반 곡 수집"""
    if not LASTFM_API_KEY: return []

    params = {
        "method": "tag.gettoptracks",
        "tag": tag,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        "limit": limit
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=5)
        if response.status_code != 200: return []
        
        data = response.json()
        raw_tracks = data.get("tracks", {}).get("track", [])
        
        cleaned = []
        for t in raw_tracks:
            # 아티스트 이름 안전하게 추출
            artist = t.get("artist", {})
            artist_name = artist.get("name") if isinstance(artist, dict) else str(artist)
            
            cleaned.append({
                "title": t.get("name"),
                "artist": artist_name,
                "url": t.get("url")
            })
        return cleaned
    except Exception as e:
        print(f"Last.fm Error ({tag}): {e}")
        return []

def generate_search_query(primary_tag: str) -> str:
    """
    GPT의 3단 태그를 Last.fm에 최적화된 '가변적 검색어'로 변환
    """
    try:
        parts = primary_tag.split('_') # [Goal, Genre, Vibe]
        if len(parts) != 3: return "pop" # 포맷 깨지면 기본값
        
        goal, genre, vibe = parts[0], parts[1], parts[2]
        
        # ---------------------------------------------------------
        # 전략 1. 기본은 [Vibe + Genre] 조합
        # ---------------------------------------------------------
        query = f"{vibe} {genre}" 
        
        # ---------------------------------------------------------
        # 전략 2. 특정 '목표(Goal)'가 강력할 땐 [Goal + Genre]로 덮어쓰기
        # (Last.fm의 '기능성 태그' 활용)
        # ---------------------------------------------------------
        if goal == "sleep":
            query = f"sleep {genre}"      # 예: sleep piano
        elif goal == "focus":
            query = f"study {genre}"      # 예: study lo-fi ('focus'보다 'study'가 결과 많음)
        elif goal == "active":
            query = f"workout {genre}"    # 예: workout k-pop ('active'보다 'workout'이 국룰)
        elif goal == "anger":
            query = f"angry {genre}"      # 예: angry rock
            
        # ---------------------------------------------------------
        # 전략 3. 장르별/바이브별 특수 매핑 (하드코딩 수정)
        # ---------------------------------------------------------
        # 락은 intense보다 hard rock, metal이 더 정확함
        if genre == "rock" and vibe == "intense":
            query = "hard rock"
            
        # ---------------------------------------------------------
        # 마무리. 띄어쓰기 및 특수문자 정리
        # ---------------------------------------------------------
        # Last.fm은 'k-pop'보다 'kpop', 'hip-hop'보다 'hip hop'을 선호하는 경향이 있음
        query = query.replace("k-pop", "kpop").replace("r-n-b", "rnb").replace("hip-hop", "hip hop")
        
        return query
        
    except Exception as e:
        print(f"Query Gen Error: {e}")
        return "pop"

# =========================================================
# 3. 노드 함수 정의 (Planner -> Fetcher -> Filter)
# =========================================================

# (1) Planner: GPT가 전략(Tag, Reason) 수립
def planner_node(state: AgentState):
    print("\n [Planner] 전략 수립 중...")
    
    ctx = state['user_context']
    pref = state['user_preference']
    
    # 프롬프트 (Logic Matrix 포함)
    system_prompt = system_prompt = """
    당신은 상황 맥락 인식 음악 추천 전문가입니다.
    사용자의 Context와 Preference를 분석하여 단 하나의 최적의 전략(Primary Tag)과 목표 오디오 수치를 수립하십시오.

    ### 1. Logic Matrix (기준표)

    (1) Goal -> Key Genres & Audio Features(Audio Features는 추천된 노래(태그)가 GPT가 의도한 음악속성과 일치하는지 보는 데이터분석용 변수입니다. 검색태그로는 들어가지 않습니다)
    목표에 따라 아래 장르와 오디오 수치 범위를 우선적으로 고려하되, 사용자 선호(Preference)가 있다면 유연하게 조정하십시오.

    - Focus / Sleep
    - Recommended Genres: [classical, jazz, ambient, piano, folk, lo-fi, new-age]
    - Audio Target:
     - Energy: 0.0 ~ 0.3 (매우 낮음)
     - Tempo (BPM): 60 ~ 90 (느림)
     - Instrumentalness: 0.7 ~ 1.0 (가사 없는 곡 위주)

    - Relax / Consolation
    - Recommended Genres: [indie-pop, r-n-b, soul, ballad, acoustic, jazz, c-pop]
    - Audio Target:
     - Energy: 0.3 ~ 0.6 (중간 이하)
     - Tempo (BPM): 70 ~ 110 (편안한 속도)
     - Instrumentalness: 0.0 ~ 0.5 (보컬 허용, 부드러운 음색)
     - Valence: 0.3 ~ 0.6 (차분함)
    
    - Active / Anger
    - Recommended Genres: [k-pop, pop, rock, hip-hop, edm, k-hiphop, j-rock, dance-pop]
    - Audio Target:
     - Energy: 0.7 ~ 1.0 (높음)
     - Tempo (BPM): 120 이상 (빠름)
     - Instrumentalness: 0.0 ~ 0.2 (강렬한 비트와 보컬)
     - Valence: 0.6 이상 (신나거나 강렬함)
    
    - Neutral
    - 사용자 선호 장르를 최우선으로 반영하며, 오디오 수치는 중간값(Moderate)을 기준으로 함.

    (2) Location -> Vibe Guidelines & inst
    장소에 어울리는 분위기를 선택하되, 반드시 아래 [Allowed Vibe List]에 있는 단어만 사용하십시오.
    - Library / Co-working: 
      - Recommended Vibes: [calm, chill, melancholy]
      - Inst: High (가사 지양)
    - Gym / Moving: 
      - Recommended Vibes: [intense, energetic, heavy, groovy]
      - Inst: Low (리듬감 필수)
    - Cafe / Home / Park: 
      - Recommended Vibes: [uplifting, dreamy, happy, groovy, chill, dark]
      - Inst: Moderate (분위기 중심)

    (3) Decibel -> Vibe & Energy fine-tuning
    소음 수준은 Vibe 선택과 Energy 목표치에 결정적인 영향을 줍니다.
    
    - Silent / Quiet (조용함): 
      - Strategy: 분위기를 유지하고 방해하지 않음.
      - Vibe Selection: [calm, chill, dreamy, melancholy] 중 선택.
      - Energy Target: 0.0 ~ 0.4 (Low)
      
    - Moderate (보통):
      - Strategy: 밸런스 유지.
      - Vibe Selection: [groovy, uplifting, happy, chill] 중 선택.
      - Energy Target: 0.4 ~ 0.7 (Mid)
      
    - Loud / Very Loud (시끄러움):
      - Strategy: 소음 마스킹 (Noise Masking). 외부 소음을 덮을 수 있는 강한 비트.
      - Vibe Selection: [intense, energetic, heavy, groovy] 중 선택.
      - Energy Target: 0.7 ~ 1.0 (High)

    ### 2. Allowed Genre List (허용된 장르 리스트)
    Primary Tag의 중간(Genre) 부분은 반드시 아래 시드(Seed) 중 하나를 선택해야 합니다.

    1. K-POP: k-pop, k-pop-boy-group, k-pop-girl-group
    2. Asian-pop: j-pop, anime, j-rock, c-pop, mandopop
    3. Classic: classical
    4. Jazz: jazz
    5. Indie: indie-pop, indie-rock, k-indie, folk
    6. Soul / R&B: r-n-b, soul, korean-rnb, neo-soul
    7. K-Hiphop: korean-hip-hop, k-rap
    8. Hip-hop: hip-hop, rap, trap, gangster-rap, lo-fi
    9. Rock: rock, alt-rock, k-rock, punk-rock
    10. EDM: edm, house, electro, trance
    11. Ballad: ballad, acoustic, piano, korean-ballad, ambient, new-age
    12. Pop: pop, dance-pop, teen-pop

    ### 3. Allowed VibeList (허용된 바이브 리스트, 너무 창의적인 바이브를 뽑으면 검색 안될 위험)
    Primary Tag의 마지막(Vibe) 부분은 반드시 아래 단어 중 하나여야 합니다.

    Options: [calm, chill, melancholy, intense, energetic, heavy, groovy, uplifting, dreamy, happy, dark]


    ### 4. 출력 포맷 (JSON)
    Raw JSON String만 반환하십시오.

    {
        "primary_tag": "{Goal}_{Genre}_{Vibe}",
        "reasoning": primary_tag를 설정한 이유를 1문장으로 서술하시오. 예: "사용자가 도서관(Library)에서 집중(Focus)를 할 수 있도록, 조용한(Silent) 환경을 고려하여 [장르(Genre)]를 선정했습니다.",
        "target_audio_features": { 
            "energy": 0.0~1.0, 
            "tempo": 0~200, 
            "valence": 0.0~1.0, 
            "instrumentalness": 0.0~1.0 
        }
    }

    ### 5. 규칙
    - primary_tag는 반드시 3단 구조(Goal_Genre_Vibe)를 지킬 것.
    - Genre는 위 Allowed Genre List에 있는 소문자 시드(seed)만 사용할 것.
    - 사용자 선호 장르가 Context와 충돌할 경우 Context(장소/목표)를 우선시하되 장르의 느낌을 최대한 살릴 것.
    """
    
    user_msg = f"Context: {ctx}\nPreference: {pref}"
    
    msg = [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)]
    res = llm.invoke(msg)
    
    try:
        content = res.content.replace("```json", "").replace("```", "").strip()
        strategy = json.loads(content)
        
        # Last.fm용 검색어 생성
        tag = strategy.get("primary_tag", "neutral_pop_calm")
        query = generate_search_query(tag)
        
        return {
            "gpt_strategy": strategy,
            "search_queries": [query], # 리스트로 저장 (확장성 고려)
            "recommendation_meta": {
                "reasoning": strategy.get("reasoning", ""),
                "primary_tag": tag
            }
        }
    except:
        # 에러 시 기본값
        return {
            "gpt_strategy": {},
            "search_queries": ["pop"],
            "recommendation_meta": {"reasoning": "Error", "primary_tag": "error"}
        }

# (2) Fetcher: 평행우주 전략 (한국 60% + 글로벌 40% 강제 확보)
def fetcher_node(state: AgentState):
    gpt_data = state['gpt_strategy']
    primary_tag = gpt_data.get('primary_tag', '') 
    
    # 태그 파싱
    try:
        parts = primary_tag.split('_') # [Goal, Genre, Vibe]
        if len(parts) == 3:
            goal, genre, vibe = parts
        else:
            goal, genre, vibe = "neutral", "pop", "calm"
    except:
        goal, genre, vibe = "neutral", "pop", "calm"

    # 기본 검색어 준비
    global_genre = genre.replace("k-pop", "kpop").replace("r-n-b", "rnb")
    
    # 목표(Goal) 매핑
    goal_map = {
        "focus": "study", "sleep": "sleep", "active": "workout", 
        "anger": "angry", "relax": "chill", "consolation": "sad"
    }
    q_goal = goal_map.get(goal, vibe)

    print(f"\n📡 [Fetcher] 이원화 검색 시작 (Tag: {primary_tag})")
    
    # 최종 결과를 담을 리스트
    final_candidates = []
    seen_keys = set()

    # =======================================================
    # 🇰🇷 [Track A] 한국 노래 채굴 (목표: 60곡)
    # 전략: 한국 쿼리는 '교집합'을 쓰면 글로벌 곡이 섞이므로, 
    #       '조합(Combo)'을 최우선으로 하고, 안되면 '장르'로 넓힘.
    # =======================================================
    def get_korean_query(base_genre):
        k_map = {
            "pop": "k-pop", "r-n-b": "krnb", "hip-hop": "korean hip-hop",
            "indie": "k-indie", "rock": "k-rock", "ballad": "korean ballad",
            "jazz": "korean jazz", "electronic": "korean electronic",
            "folk": "korean folk", "classical": "korean classical" 
        }
        return k_map.get(base_genre, f"korean {base_genre}")

    k_genre = get_korean_query(genre)
    q_k_combo = f"{vibe} {k_genre}" # 예: calm k-indie

    print(f"    🇰🇷 [Korea Batch] 목표 60곡 | 검색어: '{q_k_combo}'")
    
    k_tracks = []
    # 1순위: 조합 검색
    k_batch1 = call_lastfm_api(q_k_combo, limit=60)
    k_tracks.extend(k_batch1)
    
    # 2순위: 부족하면 장르 전체 검색 (한국 노래 확보가 최우선이라 교집합 안 씀)
    if len(k_tracks) < 60:
        print(f"       -> 부족({len(k_tracks)}). 광범위 검색('{k_genre}') 추가.")
        k_batch2 = call_lastfm_api(k_genre, limit=60)
        for t in k_batch2:
            if len(k_tracks) >= 60: break
            k_tracks.append(t)

    # 중복 제거 및 등록 (최대 60개)
    count_kr = 0
    for t in k_tracks:
        if count_kr >= 60: break
        key = f"{t['artist']}-{t['title']}".lower()
        if key not in seen_keys:
            seen_keys.add(key)
            t["relevance_score"] = 10 # 한국 노래는 무조건 상위권 (10점)
            final_candidates.append(t)
            count_kr += 1
            
    print(f"       -> 한국 노래 {count_kr}곡 확보.")

    # =======================================================
    # 🌍 [Track B] 글로벌 노래 채굴 (목표: 40곡 + alpha)
    # 전략: 글로벌은 데이터가 많으므로 '조합 -> 교집합' 고도화 로직 적용
    # =======================================================
    q_g_combo = f"{vibe} {global_genre}"
    print(f"    🌍 [Global Batch] 목표 40곡+ | 검색어: '{q_g_combo}' -> 교차검색")
    
    g_candidates = {} # {key: {data, score}}

    # 1순위: 조합 검색
    g_batch1 = call_lastfm_api(q_g_combo, limit=50)
    
    if len(g_batch1) >= 15:
        # 대성공 시
        for t in g_batch1:
            key = f"{t['artist']}-{t['title']}".lower()
            g_candidates[key] = {"data": t, "score": 5} # 5점 (한국보단 낮게)
    else:
        # 실패 시 교집합(Intersection) 가동
        # 1순위 결과 유지
        for t in g_batch1:
            key = f"{t['artist']}-{t['title']}".lower()
            g_candidates[key] = {"data": t, "score": 5}
            
        # 2순위 교차 검색
        queries = list(set([global_genre, vibe, q_goal]))
        for q in queries:
            sub_tracks = call_lastfm_api(q, limit=50)
            weight = 1
            if q == q_goal: weight = 3
            elif q == vibe: weight = 2
            
            for t in sub_tracks:
                key = f"{t['artist']}-{t['title']}".lower()
                if key not in g_candidates:
                    g_candidates[key] = {"data": t, "score": 0}
                g_candidates[key]["score"] += weight

    # 글로벌 노래 등록 (100개 찰 때까지)
    # 점수순 정렬 후 추가
    sorted_g = sorted(g_candidates.values(), key=lambda x: x["score"], reverse=True)
    
    count_gl = 0
    for item in sorted_g:
        if len(final_candidates) >= 100: break # 총 정원 100명 마감
        
        # 2점 이상(맥락 있음)이거나 콤보 성공작(5점)만
        if item["score"] >= 2:
            t = item["data"]
            key = f"{t['artist']}-{t['title']}".lower()
            
            if key not in seen_keys:
                seen_keys.add(key)
                t["relevance_score"] = item["score"] # 2~5점
                final_candidates.append(t)
                count_gl += 1

    # 혹시 부족하면 글로벌 인기곡 수혈
    if len(final_candidates) < 50:
         backup = call_lastfm_api(global_genre, limit=50)
         for t in backup:
             if len(final_candidates) >= 100: break
             key = f"{t['artist']}-{t['title']}".lower()
             if key not in seen_keys:
                 seen_keys.add(key)
                 t["relevance_score"] = 1
                 final_candidates.append(t)

    print(f"       -> 글로벌 노래 {count_gl}곡 확보.")
    print(f"    -> 최종 후보군 {len(final_candidates)}곡 (KR: {count_kr} / GL: {count_gl})")
    
    return {"candidate_tracks": final_candidates}

# (3) Filter: 룰 베이스 필터링 (선호 아티스트 30% + 랜덤 70%)
import re

def filter_node(state: AgentState):
    print(" [Filter] 선호 아티스트 및 한국 노래 우대 적용 중...")
    
    candidates = state['candidate_tracks']
    pref_artists = state['user_preference']['preferred_artists']
    pref_artists_lower = [a.lower() for a in pref_artists]
    
    # 한글 판별 함수 (정규식)
    def has_korean(text):
        return bool(re.search("[가-힣]", str(text)))

    # 그룹 분리
    my_picks = []      # 1순위: 선호 아티스트
    korean_picks = []  # 2순위: 한국 노래 (한글 포함)
    others = []        # 3순위: 나머지 (팝 등)
    
    for t in candidates:
        artist = t['artist']
        title = t['title']
        
        # 1. 선호 아티스트 체크
        if artist.lower() in pref_artists_lower:
            my_picks.append(t)
        # 2. 한국 노래 체크 (아티스트나 제목에 한글이 있으면)
        elif has_korean(artist) or has_korean(title):
            korean_picks.append(t)
        else:
            others.append(t)
            
    # --- 비율 구성 로직 ---
    final_list = []
    
    # 1. 선호 아티스트 (최대 3곡)
    final_list.extend(my_picks[:3])
    
    # 2. 한국 노래 우선 채우기 (남은 자리에 한국 노래 밀어넣기)
    needed = 10 - len(final_list)
    if needed > 0:
        random.shuffle(korean_picks)
        # 한국 노래를 최대한 넣음 (예: 5곡 정도)
        k_count = min(needed, len(korean_picks)) 
        final_list.extend(korean_picks[:k_count])
        
    # 3. 나머지는 글로벌 팝으로 채우기
    needed = 10 - len(final_list)
    if needed > 0:
        random.shuffle(others)
        final_list.extend(others[:needed])
    
    print(f"    -> 최종 {len(final_list)}곡 (My: {len(my_picks)}, KR: {len(korean_picks)}, Other: {len(others)})")
    
    return {"final_tracks": final_list}

# =========================================================
# 4. 그래프 구성 (Workflow)
# =========================================================
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("fetcher", fetcher_node)
workflow.add_node("filter", filter_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "fetcher")
workflow.add_edge("fetcher", "filter")
workflow.add_edge("filter", END)

app = workflow.compile()

# =========================================================
# 5. Bridge 함수 (KPI Evaluator와 연결)
# =========================================================
async def run_agent_bridge(inputs: dict):
    """
    KPI Evaluator가 호출하는 진입점.
    입력: 평가용 딕셔너리
    출력: 평가용 JSON 리스트 (String)
    """
    
    # 1. 입력 변환 (Flatten -> Nested)
    user_context = {
        "location": str(inputs.get('location', 'home')),
        "decibel_level": str(inputs.get('decibel', 'moderate')),
        "goal": str(inputs.get('goal', 'neutral')),
        "current_time": "14:00"
    }
    
    pref_genre = inputs.get('user_pref')
    pref_artist = inputs.get('user_artist')
    
    user_pref = {
        "preferred_genres": [pref_genre] if pref_genre and pref_genre != 'None' else [],
        "preferred_artists": [pref_artist] if pref_artist and pref_artist != 'None' else []
    }
    
    # 2. 초기 상태
    initial_state = {
        "user_context": user_context,
        "user_preference": user_pref,
        "gpt_strategy": {},
        "search_queries": [],
        "candidate_tracks": [],
        "final_tracks": [],
        "recommendation_meta": {}
    }
    
    # 3. 실행
    try:
        result = await app.ainvoke(initial_state)
        final_tracks = result.get('final_tracks', [])
        meta = result.get('recommendation_meta', {})
        strategy = result.get('gpt_strategy', {})
        
        # 4. 포맷 변환 (Logic -> Evaluation Format)
        formatted_output = []
        for t in final_tracks:
            item = {
                "recommendation_meta": meta, # Reasoning과 Tag는 10곡 모두 동일하게 적용
                "track_info": {
                    "artist_name": t['artist'],
                    "track_title": t['title']
                },
                # Audio Features는 GPT가 생각한 '목표치'를 넣어줌 (분석용)
                "target_audio_features": strategy.get("target_audio_features", {})
            }
            formatted_output.append(item)
            
        # JSON 문자열로 반환
        return json.dumps(formatted_output, ensure_ascii=False)
        
    except Exception as e:
        print(f" Agent Error: {e}")
        return json.dumps([])

# =========================================================
# 테스트 실행 (직접 실행 시)
# =========================================================
if __name__ == "__main__":
    import asyncio
    
    async def main():
        inputs = {
            "location": "library",
            "decibel": "silent",
            "goal": "focus",
            "user_pref": "k-pop",
            "user_artist": "BTS"
        }
        
        print(" 에이전트 실행 중...")
        res = await run_agent_bridge(inputs)
        print("\n 결과 확인 (JSON):")
        print(res[:500] + "...") # 너무 기니까 앞부분만

    asyncio.run(main())