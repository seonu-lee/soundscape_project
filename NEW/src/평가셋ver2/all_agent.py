#pip install langgraph langchain langchain_openai langchain_google_genai tavily-python langchain_community "httpx==0.27.2"

import operator
import json
from typing import Annotated, List, Tuple, Union, Literal, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START

import os
from dotenv import load_dotenv
import openai
from langchain_community.tools.tavily_search import TavilySearchResults #랭체인도구


# 0. 설정 로드 # .env 파일 로드
load_dotenv()

# OpenAI 클라이언트 설정 (KPI 평가용 등)
openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

# 1. 도구 및 LLM 설정

# Tavily 도구 설정 (API Key 환경변수 확인)
# LangChain 도구들은 보통 os.environ에 키가 있으면 자동으로 인식함.
# 명시적으로 확인하고 싶다면 아래처럼 작성.
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("Tavily API Key가 없습니다!")

tavily_tool = TavilySearchResults(max_results=5)
tools = [tavily_tool]

# LLM 설정 추천 로직이 복잡하므로 Planner와 Replanner에는 고성능 모델(GPT-4o) 권장
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
# llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Agent Executor 설정 #검색을 수행할 간단한 Agent (튜토리얼의 execute_step에서 사용)
agent_executor = create_react_agent(llm, tools)

# ---------------------------------------------------------
# 2. State 정의 (입력 데이터를 저장할 구조)
# ---------------------------------------------------------
class PlanExecute(TypedDict):
    input: str                         # 사용자의 자연어 요청 (또는 포맷팅된 문자열)
    user_context: dict                 # location, decibel_level, user_goal, current_time
    user_preference: dict              # preferred_genres, preferred_artists
    plan: List[str]                    # 실행 계획
    past_steps: Annotated[List[Tuple], operator.add] # 수행한 작업 기록
    response: str                      # 최종 JSON 응답

# 3. 데이터 모델 (Pydantic) 정의
class Plan(BaseModel):
    """검색 계획"""
    steps: List[str] = Field(description="검색 단계")

# class Response(BaseModel):
#     """최종 사용자 응답"""
#     response: str = Field(description="최종 추천 결과 (반드시 JSON 형식을 따를 것)")

class Act(BaseModel):
    """
    Replanner의 판단 결과:
    - 정보가 충분하면 response 필드에 최종 JSON을 담음
    - 부족하면 plan 필드에 추가 계획을 담음
    response와 plan 중 하나는 반드시 채워져야 함
    """

    # Response class랑 겹침
    response: Optional[str] = Field(
        default=None,
        description="최종 JSON 응답 (정보가 충분할 때 작성하는 필드)"
    )
    plan: Optional[List[str]] = Field(
        default=None,
        description="추가 계획 (정보가 부족할 때 작성하는 필드)"
    )

# ---------------------------------------------------------
# 4. Prompts 설정 (Taxonomy & Output Format 반영)
# ---------------------------------------------------------

# (1) 초기 계획 수립 프롬프트
planner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """당신은 음악 추천을 위한 '검색 계획가(Search Planner)'입니다.
     사용자의 상황(Context)과 취향(Preference)을 분석하여, Tavily로 검색할 구체적인 계획을 세우세요.

     ### [핵심 규칙]
     1. 복합 쿼리 생성: 반드시 'Context(장소/목표)'와 'Genre'를 결합하여 검색어를 만드세요.
        - (X) Bad: "Metallica 노래 검색"
        - (O) Good: "**도서관에서 듣기 좋은** Metallica의 **어쿠스틱 커버 곡** 검색"
     2. Taxonomy 참고: 아래의 '장소(Location)'와 '목표(Goal)' 분류를 반드시 준수하세요.
        - Location: cafe, library, co-working, moving, gym, home, park
        - Goal: focus, relax, sleep, active, anger, consolation, neutral
     """),
    ("user", "Context: {user_context}\nPreference: {user_preference}\nRequest: {input}")
])

first_planner = planner_prompt | llm.with_structured_output(Plan)

# (2) Replanner (검토 및 답변 생성) 프롬프트 
replanner_system_prompt = replanner_system_prompt = """
당신은 '상황 맥락 인식 음악 추천 전문가'입니다.
검색 결과와 사용자 정보를 바탕으로 최적의 음악을 추천하고, 분석용 데이터를 생성하십시오.

### 1. Taxonomy & Logic Matrix (기준표)
추천 로직은 아래의 정의를 엄격히 따릅니다.

(1-1) Goal (목표) → Audio Features 기본 범위
- Focus / Sleep: 
  - Energy: 0.0 ~ 0.4 (낮음) | Tempo: 60 ~ 90 BPM | Inst: 0.7 ~ 1.0 (가사 지양)
- Relax / Consolation: 
  - Acousticness: 0.6 ~ 1.0 | Valence: 0.3 ~ 0.6 (차분함)
- Active / Anger: 
  - Energy: 0.7 ~ 1.0 (높음) | Tempo: 120+ BPM | Valence: 0.6+ (강렬함)
  
(1-2) Goal (목표) → Recommended Genres (가이드라인)
목표에 따라 아래 장르를 우선적으로 고려하되, 사용자 선호(Preference)가 있다면 유연하게 조정하십시오.

- Focus / Sleep:
  - [Classical, Ambient, Lo-fi, Piano, Soundtrack, New-age]
  - (Note: 가사가 없고 차분한 장르 위주)

- Relax / Consolation: 
  - [Acoustic, Ballad, R-nb, Jazz, Indie, Folk]
  - (Note: 감성적이고 부드러운 장르 위주)

- Active / Anger: 
  - [Pop, K-Pop, Rock, Hip-hop, Electronic]
  - (Note: 비트가 강하고 에너지가 높은 장르 위주)

(2) Location (장소) → Vibe & Instrumentalness
- Strict (Library, Co-working): 
  - 가사 방해 금지 (Inst: 0.8~1.0). Vibe: `calm`, `melancholy`.
- Casual (Cafe, Home, Park): 
  - 허밍/적당한 가사 허용. Vibe: `groovy`, `uplifting`, `dreamy`.
- Active (Gym, Moving): 
  - 리듬감 필수. Vibe: `intense`, `groovy`.

(3) Decibel (소음) → Energy Fine-tuning
- Silent/Quiet: 범위 내 최솟값 설정 (분위기 유지).
- Loud/Very Loud: 범위 내 최댓값 설정 (소음 마스킹).

---

### 2. Reasoning Steps (생각의 순서)
값을 결정할 때는 반드시 아래 3단계를 거쳐야 합니다.

1. Step 1 (Goal): 위 'Logic Matrix'를 참고하여 `Genre`와 `target_audio_features`의 기본 범위를 잡으십시오.
2. Step 2 (Location): 장소의 사회적 맥락을 고려하여 `Vibe`와 `Instrumentalness`를 구체화하십시오.
3. Step 3 (Decibel): 소음도를 고려하여 `Energy` 수치를 미세 조정하십시오. (절대 기본 범위를 벗어나지 말 것)

---

### 3. Primary Tag 생성 규칙 (Strict 3-Tier)
통계 분석을 위해 태그는 반드시 **`{Goal}_{Genre}_{Vibe}`** 형식을 지켜야 합니다.

- Prefix (Goal): 사용자 입력 Goal 그대로 사용.
- Middle (Genre): 반드시 아래 리스트 중 하나 선택.
  - Options: [pop, k-pop, rock, hip-hop, r-nb, jazz, indie, folk, electronic, classical, ballad, acoustic, soundtrack, ambient, lo-fi, new-age, piano]
- Suffix (Vibe): 분위기를 가장 잘 나타내는 단어 1개 선택.
  - Options: [calm, groovy, intense, dreamy, uplifting, melancholy]

- Example: `focus_piano_calm`, `active_k-pop_uplifting`, `anger_rock_intense`

---

### 4. 절대적 규칙 (CRITICAL RULES)
1. 충돌 해결(Conflict Resolution): 사용자 선호(Genre)와 상황(Context)이 충돌하면, 상황을 우선시하되 장르의 느낌만 유지하세요. (예: 도서관+메탈 → 어쿠스틱 메탈/포스트 락)
2. 거짓말 금지: 실제 Spotify에 존재하는 곡만 추천하세요.
3. 언어 및 키워드 설정: 
   - 설명(Reasoning)은 한국어로 작성하되, 핵심 키워드(Goal, Location, Genre, Vibe)는 반드시 영어 원문을 괄호 `()` 안에 병기하세요.
   - Bad: "도서관이라서 조용한 피아노 곡을 골랐습니다."
   - Good: "사용자가 도서관(Library)에서 집중(Focus)할 수 있도록, 차분한(Calm) 분위기의 피아노(Piano) 곡을 선정했습니다."

---

### 5. 출력 포맷 (JSON Schema)
응답은 반드시 아래 JSON 리스트 포맷이어야 합니다. Markdown Block 없이 Raw String만 반환하세요.

[
  {
    "recommendation_meta": {
      "reasoning": "사용자가 [장소(Location)]에서 [목표(Goal)]를 달성할 수 있도록, [소음(Decibel)] 환경을 고려하여 [장르(Genre)]를 선정했습니다.",
      "primary_tag": "{Goal}_{Genre}_{Vibe}"
    },
    "track_info": {
      "artist_name": "Exact Artist Name",
      "track_title": "Exact Track Title"
    },
    "target_audio_features": {
      "min_tempo": (int),
      "max_tempo": (int),
      "target_energy": (float 0.0~1.0),
      "target_instrumentalness": (float 0.0~1.0),
      "target_valence": (float 0.0~1.0),
      "target_acousticness": (float 0.0~1.0)
    }
  }
]

### 6. 행동 지침 (Action Guideline)
- 이미 검색(past_steps)을 1회 이상 수행했고, 음악을 추천할 수 있는 정보가 조금이라도 있다면 **즉시 JSON을 생성하여 'response' 필드에 담으십시오.**
- 완벽한 정보를 찾기 위해 무의미한 재검색을 반복하지 마십시오.
- JSON 생성이 가능하다면 'plan'을 비우고 'response'만 반환하세요.

### 7. 입력 정보
- Context: {user_context}
- Preference: {user_preference}
"""

replanner_prompt = ChatPromptTemplate.from_template(
    """{system_prompt}

    원래 목표: {input}

    현재 계획: {plan}

    완료된 단계와 결과:
    {past_steps}

    위 정보를 바탕으로 판단하십시오:
    정보가 충분하면 'response'에 JSON을 작성하고, 부족하면 'plan'을 작성하세요.
   """
)

edited_planner = replanner_prompt | llm.with_structured_output(Act)

# 5. 노드 함수 정의

async def first_plan_step(state: PlanExecute):
    plan = await first_planner.ainvoke({
        "input": state["input"],
        "user_context": state["user_context"],
        "user_preference": state["user_preference"]
    })
    return {"plan": plan.steps}

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    # 다음 실행할 작업 (첫 번째 단계)
    task = plan[0]

    # 컨텍스트를 포함하여 검색 쿼리 최적화
    task_formatted = f"""
    사용자 정보: {state['user_context']}
    선호 정보: {state['user_preference']}

    위 정보를 참고하여 다음 검색 작업을 수행하세요: {task}
    """

    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

async def edited_plan_step(state: PlanExecute):
    output = await edited_planner.ainvoke({
        "system_prompt": replanner_system_prompt,
        "input": state["input"],
        "plan": state["plan"],
        "past_steps": state["past_steps"],
        "user_context": str(state["user_context"]),
        "user_preference": str(state["user_preference"])
    })

    # response 필드가 채워져 있으면 응답 반환
    if output.response:
        return {"response": output.response}
    # 아니면 plan 반환
    else:
        return {"plan": output.plan}

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"
    
# 6. 그래프 구성
workflow = StateGraph(PlanExecute)

workflow.add_node("planner", first_plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", edited_plan_step)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)

######### 외부(KPI평가)에서 import 할 객체 (이름을 app으로 통일하면 편함) #########
app = workflow.compile()

# =========================================================
# [add] KPI 코드와 연결하기 위한 '다리(Bridge)' 함수
# =========================================================
async def run_agent_bridge(inputs: dict):
    """
    KPI Evaluator에서 호출하는 함수입니다.
    inputs: dict 형태로 입력 데이터를 받습니다.
    """
    # 1. 입력 데이터 구조 변환 (Flat Dict -> Nested Dict)

    # 1. User Context 매핑 (기본값, 대소문자 처리 포함)
    user_context = {
        "location": str(inputs.get('location', 'home')).lower(), 
        "decibel_level": str(inputs.get('decibel', 'moderate')).lower(),
        "user_goal": str(inputs.get('goal', 'neutral')).lower(),
        "current_time": "14:00" 
    }

    # 2. User Preference 매핑
    # (A) 장르 처리
    # 값이 없거나 문자열 'None'이면 빈 값으로 취급
    input_genre = inputs.get('user_pref') 
    genre_list = []
    if input_genre and input_genre != 'None':
        genre_list = [input_genre]

    # (B) 아티스트 처리
    input_artist = inputs.get('user_artist')
    artist_list = []
    if input_artist and input_artist != 'None':
        # 만약 입력이 "Artist1, Artist2" 처럼 문자열 리스트일 수도 있으니 처리
        if isinstance(input_artist, list):
            artist_list = input_artist
        else:
            artist_list = [input_artist]

    user_preference = {
        "preferred_genres": genre_list,   # 없으면 [] (빈 리스트)
        "preferred_artists": artist_list  # 없으면 [] (빈 리스트)
    }
    
    # 2. 초기 State 구성
    initial_state = {
    "input": "Taxonomy와 Audio Features 포맷을 준수하여 음악 3곡을 추천해줘.",
    "user_context": user_context,
    "user_preference": user_preference,
    "plan": [],
    "past_steps": []
    }
    
    # 3. 그래프 실행 
    config = {"recursion_limit": 20}
    final_state = await app.ainvoke(initial_state, config=config)
    
    # 4. 최종 결과 반환
    return final_state.get('response', '{"error": "No response generated within limit"}')

# =========================================================
# 테스트 실행 (강력해진 파싱 기능 탑재)
# =========================================================
if __name__ == "__main__":
    import asyncio
    import json
    
    # 핵심 함수: 문자열에서 JSON 부분만 정밀하게 파싱
    def extract_json_core(text):
        try:
            # 1. 마크다운 제거
            text = text.replace("```json", "").replace("```", "").strip()
            
            # 2. 리스트('[')의 시작과 끝(']') 위치 찾기
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            
            # 대괄호가 둘 다 발견되면 그 사이만 잘라냄
            if start_idx != -1 and end_idx != -1:
                return text[start_idx : end_idx + 1]
            
            # 대괄호가 없으면 그대로 반환 (혹시 객체 '{}'일 수도 있으니)
            return text
        except Exception:
            return text

    async def main():
        print("🎧 [테스트 모드] 실행 중...")

        test_inputs = {
            "location": "library",   
            "decibel": "silent",     
            "goal": "focus",         
            "user_pref": "Heavy Metal", #충돌예시
            "user_artist": "Metallica" 
        }

        # 1. 결과 받기 (아직은 문자열)
        raw_result = await run_agent_bridge(test_inputs)
        print(f"\n--- 원본 문자열 길이: {len(raw_result)} ---")

        # 2. 파싱 시도
        try:
            # ★ 여기서 청소 함수 호출! (지저분한 꼬리표 떼기)
            clean_json_str = extract_json_core(raw_result)
            
            # 변환 (String -> List/Dict)
            parsed_data = json.loads(clean_json_str)
            
            print("\n✅ JSON 파싱 성공!")
            print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
            
            # 키 확인
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                print(f"\n🔑 키 확인: {list(parsed_data[0].keys())}")
            
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON 변환 실패: {e}")
            print("문제가 된 부분 근처:", clean_json_str[-50:]) # 끝부분 50자만 확인

    asyncio.run(main())