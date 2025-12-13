import os
import json
import asyncio
import logging
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ★ 태깅 에이전트 연결 (파일명: tagging_agent.py)
from tagging_agent import run_agent_bridge

# --------------------------------------------------------------------------
# 0. 환경 설정
# --------------------------------------------------------------------------
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tagging_eval_log.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 외부 라이브러리 로그 레벨 조정 (TMI 제거)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("spotipy").setLevel(logging.WARNING)

# OpenAI 클라이언트
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------------------------------
# [Class] 태깅 기반 음악 추천 시스템 평가기 (5대 KPI)
# --------------------------------------------------------------------------
class TaggingMusicEvaluator:
    def __init__(self):
        # Spotify API 연결 (검색 성공률/할루시네이션 검증용)
        try:
            auth_manager = SpotifyClientCredentials()
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            logger.info("✅ Spotify API Connected (검증용).")
        except Exception as e:
            logger.error(f"❌ Spotify Connection Failed: {e}")
            self.sp = None
            
        self.diversity_tracks = [] # 추천된 곡 제목 저장소
        self.diversity_tags = []   # 생성된 태그 저장소

    def _safe_parse_json(self, json_str):
        """JSON 파싱 헬퍼 (마크다운 제거 등)"""
        try:
            if isinstance(json_str, (dict, list)):
                return json_str
            # 마크다운 제거
            clean_str = json_str.replace("```json", "").replace("```", "").strip()
            # 리스트([]) 부분만 추출 시도
            start = clean_str.find('[')
            end = clean_str.rfind(']')
            if start != -1 and end != -1:
                clean_str = clean_str[start : end + 1]
            return json.loads(clean_str)
        except:
            return None

    def _extract_reasoning(self, parsed_data):
        """평가용 Reasoning 텍스트 추출"""
        try:
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                return parsed_data[0].get('recommendation_meta', {}).get('reasoning', '')
            return ""
        except:
            return ""
            
    def _extract_track_info_str(self, parsed_data):
        """로그 출력용 곡 정보 추출 (첫 번째 곡 기준)"""
        try:
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                info = parsed_data[0].get('track_info', {})
                artist = info.get('artist_name', 'Unknown')
                title = info.get('track_title', 'Unknown')
                return f"{artist} - {title}"
            return "Parsing Failed"
        except:
            return "Error"

    def _extract_primary_tag(self, parsed_data):
        """생성된 3단 태그 추출"""
        try:
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                return parsed_data[0].get('recommendation_meta', {}).get('primary_tag', 'unknown')
            return "Parsing Failed"
        except:
            return "Error"

    # ======================================================================
    # KPI 1. 정확성 (Accuracy) - Math & Logic (오디오 정합성 추가됨)
    # ======================================================================
    async def evaluate_accuracy(self, row, output_data):
        criteria = row['Evaluation Criteria']
        reasoning_text = self._extract_reasoning(output_data)
        tag = self._extract_primary_tag(output_data)
        
        # [NEW] GPT가 설정한 목표 오디오 수치 가져오기
        try:
            audio_target = output_data[0].get('target_audio_features', {})
        except:
            audio_target = "N/A"
        
        if not reasoning_text:
            return 0, 0

        # (A) Math Score (임베딩 유사도) - 그대로 유지
        score_math = 0
        try:
            resp = await aclient.embeddings.create(
                input=[criteria, reasoning_text], 
                model="text-embedding-3-small"
            )
            vec1 = resp.data[0].embedding
            vec2 = resp.data[1].embedding
            sim = cosine_similarity([vec1], [vec2])[0][0]
            score_math = max(0, sim * 100)
        except Exception:
            pass

        # (B) Logic Score (LLM 채점 - 심사 기준 강화!)
        system_prompt = """
        당신은 '음악 추천 품질 평가관'입니다. 
        사용자의 요구사항(Criteria)과 AI가 생성한 전략(Tag, Reasoning, Audio Features)을 종합적으로 평가하세요.

        [채점 기준 0~100점]
        1. Context 적합성: Reasoning이 장소와 목표를 잘 반영했는가?
        2. Tag 일치성: Primary Tag가 Reasoning과 모순되지 않는가?
        3. Audio 논리성 (중요): 'Primary Tag'와 'Target Audio Features'가 논리적으로 일치하는가?
           - 감점 예시 1: Tag가 'Sleep'(수면)인데 Energy가 0.8(높음)인 경우.
           - 감점 예시 2: Tag가 'Gym'(운동)인데 BPM(Tempo)이 60(느림)인 경우.
           - 감점 예시 3: Tag가 'Study'(공부)인데 Instrumentalness가 0.0(보컬 많음)인 경우.

        [출력 형식 (JSON)]
        { "score": 85, "reason": "태그는 적절하나, Sleep 태그에 비해 Energy 목표치가 0.6으로 다소 높게 설정되어 감점함." }
        """
        
        context_str = f"Location: {row['Location']}, Goal: {row['Goal']}, Pref: {row['User Pref']}"
        
        # 심사위원에게 보여줄 데이터 (오디오 타겟 포함)
        user_msg = f"""
        [User Input]
        Criteria: {criteria}
        Context: {context_str}

        [AI Output]
        Primary Tag: {tag}
        Reasoning: {reasoning_text}
        Target Audio Features: {audio_target}
        """
        
        score_logic = 0
        try:
            resp = await aclient.chat.completions.create(
                model="gpt-4o", 
                messages=[{"role":"system", "content":system_prompt}, {"role":"user", "content":user_msg}], 
                response_format={"type": "json_object"},
                temperature=0
            )
            eval_res = json.loads(resp.choices[0].message.content)
            score_logic = eval_res.get('score', 0)
        except Exception:
            pass

        return score_math, score_logic
    
    # ======================================================================
    # KPI 2. 안정성 (Stability) - 포맷 준수 여부
    # ======================================================================
    def evaluate_system_stability(self, parsed_data):
        if parsed_data is None: return 0 
        if not isinstance(parsed_data, list) or len(parsed_data) == 0: return 0 
        
        # 필수 키 확인
        required = ["recommendation_meta", "track_info", "target_audio_features"]
        # 첫 번째 아이템만 검사
        if all(key in parsed_data[0] for key in required):
            return 1 
        return 0

    # ======================================================================
    # KPI 3. 검색 성공률 (Search Success) - 실존 여부 검증 (Spotify)
    # ======================================================================
    def evaluate_search_success(self, parsed_data):
        if not self.evaluate_system_stability(parsed_data): return 0
        if self.sp is None: return 1 # API 없으면 패스 (1점 처리)
        
        try:
            # 리스트의 첫 번째 곡만 샘플링 검증 (속도 위해)
            info = parsed_data[0]['track_info']
            title = info.get('track_title', '').strip()
            artist = info.get('artist_name', '').strip()

            if not title or not artist or "unknown" in title.lower(): return 0

            # Last.fm 곡이 Spotify에 있는지 확인
            q = f"track:{title} artist:{artist}"
            res = self.sp.search(q=q, type='track', limit=1)
            
            if len(res['tracks']['items']) > 0: 
                return 1 # 재생 가능
            
            # 검색 안 되면 조금 느슨하게 다시 시도
            q_loose = f"{artist} {title}"
            res_loose = self.sp.search(q=q_loose, type='track', limit=1)
            if len(res_loose['tracks']['items']) > 0:
                return 1
            
            return 0 # Spotify에 없음 (재생 불가 = 실패)
        except:
            return 0

    # ======================================================================
    # KPI 4. 일관성 (Consistency) - [Primary Tag 동일성 검증]
    # ======================================================================
    async def evaluate_consistency(self, inputs, first_parsed_data):
        tags = []
        # 첫 번째 실행 결과 태그
        if first_parsed_data:
            tag1 = self._extract_primary_tag(first_parsed_data)
            tags.append(tag1)
        else:
            tags.append("error_1")

        try:
            # 두 번 더 실행해서 태그 비교 (총 3회)
            tasks = [run_agent_bridge(inputs) for _ in range(2)]
            results = await asyncio.gather(*tasks)
            
            for res in results:
                parsed = self._safe_parse_json(res)
                if parsed:
                    tag = self._extract_primary_tag(parsed)
                    tags.append(tag)
                else:
                    tags.append("error_run")
                    
        except Exception as e:
            logger.error(f"Consistency Check Error: {e}")
            return 0.0

        if not tags: return 0.0
        
        # 최빈값 비율 계산 (예: ['pop', 'pop', 'jazz'] -> 2/3 = 0.66)
        from collections import Counter
        counts = Counter(tags)
        most_common_count = counts.most_common(1)[0][1] 
        score = most_common_count / len(tags)
        
        if score < 1.0:
            logger.info(f"ℹ️ Tag Consistency Diff: {tags}")
            
        return score
    
    # ======================================================================
    # KPI 5. 다양성 (Diversity) - 태그 및 곡 중복도
    # ======================================================================
    def record_diversity(self, parsed_data):
        if parsed_data:
            # 태그 수집
            tag = self._extract_primary_tag(parsed_data)
            self.diversity_tags.append(tag)
            
            # 곡 제목 수집 (첫 곡 기준)
            t = parsed_data[0].get('track_info', {}).get('track_title', 'unknown')
            self.diversity_tracks.append(t)

    def calculate_diversity(self):
        # 태그 다양성과 곡 다양성을 평균냄
        if not self.diversity_tags: return 0.0
        
        tag_div = len(set(self.diversity_tags)) / len(self.diversity_tags)
        track_div = len(set(self.diversity_tracks)) / len(self.diversity_tracks)
        
        return ((tag_div + track_div) / 2) * 100

# --------------------------------------------------------------------------
# [Main] 실행부
# --------------------------------------------------------------------------
async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "evaluation_set_v2_criteria.csv")
    
    if not os.path.exists(csv_path):
        print("❌ 'evaluation_set_v2_criteria.csv' 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(csv_path)
    evaluator = TaggingMusicEvaluator()
    results = []
    
    print(f"\n🚀 [Tagging Model] 5대 KPI 평가 시작 (총 {len(df)}개 시나리오)")
    print("-" * 75)

    for idx, row in df.iterrows():
        inputs = {
            "location": row['Location'], "decibel": row['Decibel'],
            "goal": row['Goal'], "user_pref": row['User Pref'],
            "user_artist": row.get('User Artist', None)
        }
        
        print(f"▶ [{idx+1}/{len(df)}] ({row['Location']}/{row['Goal']}) ...", end=" ", flush=True)

        # 1. Agent 실행
        raw_out = await run_agent_bridge(inputs)
        parsed = evaluator._safe_parse_json(raw_out)
        
        # 2. KPI 측정
        s_math, s_logic = await evaluator.evaluate_accuracy(row, parsed)
        final_acc = (s_math * 0.3) + (s_logic * 0.7)
        s_stability = evaluator.evaluate_system_stability(parsed)
        s_search = evaluator.evaluate_search_success(parsed)
        
        # 일관성 (5번에 1번만 체크하여 속도 향상)
        s_consist = 1.0 
        if idx % 5 == 0: 
            s_consist = await evaluator.evaluate_consistency(inputs, parsed)
            
        evaluator.record_diversity(parsed)

        # 3. 결과 정리 및 출력
        track_info_str = evaluator._extract_track_info_str(parsed)
        primary_tag_str = evaluator._extract_primary_tag(parsed)
        hallucinated_track = ""
        
        # 색상 코드
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        
        # 할루시네이션 체크 (포맷은 맞는데 Spotify에 없는 경우)
        if s_search == 0 and s_stability == 1:
            hallucinated_track = track_info_str
            print(f"{RED}❌ Unplayable: {hallucinated_track}{RESET}", end=" ")
        else:
            print(f"{GREEN}✅ OK{RESET} (Acc:{final_acc:.0f})", end=" ")
            
        print(f"| Tag: {primary_tag_str}")
        
        results.append({
            "ID": row.get('ID', idx),
            "Context": f"{row['Location']}-{row['Goal']}",
            "Score_Total_Accuracy": round(final_acc, 1),
            "Score_Accuracy_Logic": s_logic,          
            "Score_Accuracy_Math": round(s_math, 1),  
            "Score_Stability": s_stability,             
            "Score_SearchSuccess": s_search,          
            "Score_Consistency": s_consist,
            "Primary_Tag": primary_tag_str,
            "Hallucination_Track": hallucinated_track, 
            "Output_Reasoning": evaluator._extract_reasoning(parsed),
            "Recommended_Track": track_info_str
        })

    # 4. 최종 리포트 생성
    res_df = pd.DataFrame(results)
    
    # 다양성 계산
    diversity = evaluator.calculate_diversity()
    res_df['Score_Diversity'] = round(diversity, 1)
    
    # 전체 성공률 (재생 가능률)
    overall_search_rate = res_df['Score_SearchSuccess'].mean() * 100
    res_df['Overall_Search_Success_Rate'] = f"{overall_search_rate:.1f}%"

    # 5. 콘솔 출력 및 CSV 저장
    print("\n" + "="*40)
    print("🏆  TAGGING MODEL KPI REPORT  🏆")
    print("="*40)
    
    avg_acc = res_df['Score_Total_Accuracy'].mean()
    avg_stab = res_df['Score_Stability'].mean() * 100
    avg_consist = res_df['Score_Consistency'].mean()
    
    print(f"1. 정확성 (Accuracy)       : {avg_acc:.1f}점")
    print(f"2. 안정성 (Stability)      : {avg_stab:.1f}%")
    print(f"3. 재생 성공률 (Success)    : {overall_search_rate:.1f}% (Spotify Valid)")
    print(f"4. 일관성 (Consistency)    : {avg_consist:.2f}")
    print(f"5. 다양성 (Diversity)      : {diversity:.1f}%")

    # (1) 상세 리포트
    detail_path = os.path.join(current_dir, "tagging_kpi_detail.csv")
    res_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 상세 리포트: {detail_path}")
    
    # (2) 요약 리포트
    summary_data = [
        {"KPI": "Accuracy", "Value": f"{avg_acc:.1f}"},
        {"KPI": "Stability", "Value": f"{avg_stab:.1f}%"},
        {"KPI": "Playability (Success)", "Value": f"{overall_search_rate:.1f}%"},
        {"KPI": "Consistency", "Value": f"{avg_consist:.2f}"},
        {"KPI": "Diversity", "Value": f"{diversity:.1f}%"}
    ]
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(current_dir, "tagging_kpi_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"✅ 요약 리포트: {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())