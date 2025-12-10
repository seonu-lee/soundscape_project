import os
import json
import asyncio
import logging
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ★ [연결] 랭그래프 엔진 가져오기
from my_agent import run_agent_bridge

# --------------------------------------------------------------------------
# 0. 환경 설정
# --------------------------------------------------------------------------
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_log.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 클라이언트 초기화
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------------------------------
# [Class] 음악 추천 시스템 통합 평가기 (5대 KPI)
# --------------------------------------------------------------------------
class MusicRecommendationEvaluator:
    def __init__(self):
        try:
            auth_manager = SpotifyClientCredentials()
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            logger.info("✅ Spotify API Connected.")
        except Exception as e:
            logger.error(f"❌ Spotify Connection Failed: {e}")
            self.sp = None
            
        self.diversity_pool = [] 

    def _safe_parse_json(self, json_str):
        try:
            if isinstance(json_str, dict) or isinstance(json_str, list):
                return json_str
            clean_str = json_str.replace("```json", "").replace("```", "").strip()
            start = clean_str.find('[')
            end = clean_str.rfind(']')
            if start != -1 and end != -1:
                clean_str = clean_str[start : end + 1]
            return json.loads(clean_str)
        except:
            return None

    def _extract_text_for_embedding(self, parsed_data):
        try:
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                return parsed_data[0].get('recommendation_meta', {}).get('reasoning', '')
            return ""
        except:
            return ""
            
    def _extract_track_info_str(self, parsed_data):
        try:
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                info = parsed_data[0].get('track_info', {})
                artist = info.get('artist_name', 'Unknown')
                title = info.get('track_title', 'Unknown')
                return f"{artist} - {title}"
            return "Parsing Failed"
        except:
            return "Error"

    # ======================================================================
    # KPI 1. 정확성 (Accuracy)
    # ======================================================================
    async def evaluate_accuracy(self, row, output_data):
        criteria = row['Evaluation Criteria']
        reasoning_text = self._extract_text_for_embedding(output_data)
        
        if not reasoning_text:
            return 0, 0

        # (A) Math
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
            
            # [디버깅] 점수가 낮으면 이유 확인용 로그
            if score_math < 40:
                logger.debug(f"[Low Math] Criteria: {criteria[:30]}... vs Reasoning: {reasoning_text[:30]}...")
        except Exception:
            pass

        # (B) Logic
        system_prompt = """
        당신은 '음악 추천 품질 평가관'입니다. 
        사용자의 요구사항(Criteria)과 AI의 추천 결과(Output)를 비교하여 점수를 매기세요.
        [채점 기준 0~100점]
        1. Context 적합성
        2. Preference 반영
        3. Conflict 해결
        [출력 형식 (JSON)]
        { "score": 85, "reason": "..." }
        """
        
        context_str = f"Location: {row['Location']}, Goal: {row['Goal']}, Pref: {row['User Pref']}"
        user_msg = f"Criteria: {criteria}\nUser Input: {context_str}\nOutput: {json.dumps(output_data, ensure_ascii=False)}"
        
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
    # KPI 2. 안정성 (Stability)
    # ======================================================================
    def evaluate_system_stability(self, parsed_data):
        if parsed_data is None: return 0 
        if not isinstance(parsed_data, list) or len(parsed_data) == 0: return 0 
        required = ["recommendation_meta", "track_info", "target_audio_features"]
        if all(key in parsed_data[0] for key in required):
            return 1 
        return 0

    # ======================================================================
    # KPI 3. 검색 성공률 (Search Success)
    # ======================================================================
    def evaluate_search_success(self, parsed_data):
        if not self.evaluate_system_stability(parsed_data): return 0
        
        try:
            info = parsed_data[0]['track_info']
            title = info.get('track_title', '').strip()
            artist = info.get('artist_name', '').strip()

            if not title or not artist or "unknown" in title.lower(): return 0
            if self.sp is None: return 1 

            # 1차 시도 (엄격)
            q_strict = f"track:{title} artist:{artist}"
            res = self.sp.search(q=q_strict, type='track', limit=1)
            if len(res['tracks']['items']) > 0: return 1 
            
            # 2차 시도 (느슨)
            q_loose = f"{title} {artist}"
            res_loose = self.sp.search(q=q_loose, type='track', limit=1)
            if len(res_loose['tracks']['items']) > 0: return 1
            
    
            return 0 
        except:
            return 0

# ======================================================================
    # KPI 4. 일관성 (Consistency) - [태그 내용 비교]
# ======================================================================
    async def evaluate_consistency(self, inputs, first_parsed_data):
        """
        동일 입력에 대해 Agent가 얼마나 유사한 'Primary Tag'를 내놓는지 평가 (3회)
        - 1회: 이미 실행한 결과(first_parsed_data) 사용
        - 2,3회: 추가 실행하여 비교
        """
        tags = []
        
        # 1. 첫 번째 실행 결과에서 태그 추출
        if first_parsed_data:
            tag1 = first_parsed_data[0].get('recommendation_meta', {}).get('primary_tag', 'error')
            tags.append(tag1)
        else:
            tags.append("error_1")

        # 2. 두 번 더 실행 (비동기 병렬 처리)
        try:
            tasks = [run_agent_bridge(inputs) for _ in range(2)]
            results = await asyncio.gather(*tasks)
            
            for res in results:
                parsed = self._safe_parse_json(res)
                if parsed:
                    tag = parsed[0].get('recommendation_meta', {}).get('primary_tag', 'error')
                    tags.append(tag)
                else:
                    tags.append("error_run")
                    
        except Exception as e:
            logger.error(f"Consistency Check Error: {e}")
            return 0.0

        # 3. 빈도 분석 (가장 많이 나온 태그가 전체의 몇 %인가?)
        # 예: ['A', 'A', 'B'] -> 'A'가 2번 -> 2/3 = 0.66
        # 예: ['A', 'B', 'C'] -> 'A'가 1번 -> 1/3 = 0.33
        
        if not tags: return 0.0
        
        from collections import Counter
        counts = Counter(tags)
        most_common_count = counts.most_common(1)[0][1] # 가장 많이 나온 횟수
        
        score = most_common_count / len(tags) # (최빈값 / 전체 시도 횟수)
        
        # [디버깅 로그] 태그가 어떻게 나왔는지 확인
        if score < 1.0:
            logger.info(f"ℹ️ Consistency Diff: {tags}")
            
        return score
    
# ======================================================================
    # KPI 5. 다양성 (diversity) 
# ======================================================================

    def record_diversity(self, parsed_data):
        if parsed_data:
            t = parsed_data[0].get('track_info', {}).get('track_title', 'unknown')
            self.diversity_pool.append(t)

    def calculate_diversity(self):
        if not self.diversity_pool: return 0.0
        return (len(set(self.diversity_pool)) / len(self.diversity_pool)) * 100

# --------------------------------------------------------------------------
# [Main] 실행
# --------------------------------------------------------------------------
async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "evaluation_set_v2_criteria.csv")
    
    if not os.path.exists(csv_path):
        print("❌ 평가셋 파일이 없습니다.")
        return

    df = pd.read_csv(csv_path)
    evaluator = MusicRecommendationEvaluator()
    results = []
    
    print(f"\n🚀 5대 KPI 평가 시작 (총 {len(df)}개 시나리오)")
    print("-" * 70)

    for idx, row in df.iterrows():
        inputs = {
            "location": row['Location'], "decibel": row['Decibel'],
            "goal": row['Goal'], "user_pref": row['User Pref'],
            "user_artist": row.get('User Artist', None)
        }
        
        print(f"▶ [{idx+1}/{len(df)}] ID {row.get('ID', idx)} ({row['Location']}) ...", end=" ")

        # 1. Agent 실행
        raw_out = await run_agent_bridge(inputs)
        parsed = evaluator._safe_parse_json(raw_out)
        
        # 2. KPI 측정
        s_math, s_logic = await evaluator.evaluate_accuracy(row, parsed)
        final_acc = (s_math * 0.3) + (s_logic * 0.7)
        s_stability = evaluator.evaluate_system_stability(parsed)
        s_search = evaluator.evaluate_search_success(parsed)
        
        s_consist = 1.0 #기본값
        if idx % 5 == 0: 
            # 첫 번째 결과(parsed)를 포함해서 비교하도록 수정
            s_consist = await evaluator.evaluate_consistency(inputs, parsed)
        evaluator.record_diversity(parsed)

        # 3. 할루시네이션 및 트랙 정보
        track_info_str = evaluator._extract_track_info_str(parsed)
        hallucinated_track = ""
        
        # 검색 실패 시 빨간색 강조 출력 (ANSI Code)
        RED = "\033[91m"
        RESET = "\033[0m"
        
        if s_search == 0 and s_stability == 1:
            hallucinated_track = track_info_str
            print(f"{RED}❌ Hallucination: {hallucinated_track}{RESET}", end=" ")
        
        # 결과 저장
        results.append({
            "ID": row.get('ID', idx),
            "Context": f"{row['Location']}-{row['Goal']}",
            "Score_Total_Accuracy": round(final_acc, 1),
            "Score_Accuracy_Logic": s_logic,          # 👈 요청하신 Logic 점수 칼럼
            "Score_Accuracy_Math": round(s_math, 1),  # 👈 요청하신 Math 점수 칼럼
            "Score_Stability": s_stability,             
            "Score_SearchSuccess": s_search,          # 개별 성공 여부 (0 or 1)
            "Score_Consistency": s_consist,
            "Hallucination_Track": hallucinated_track, 
            "Output_Reasoning": evaluator._extract_text_for_embedding(parsed),
            "Recommended_Track": track_info_str
        })
        
        if not hallucinated_track:
            print(f"✅ Acc:{final_acc:.0f}")

    # 4. 최종 집계 및 전체 성공률 계산
    res_df = pd.DataFrame(results)
    
    # 다양성 계산
    diversity = evaluator.calculate_diversity()
    res_df['Score_Diversity'] = round(diversity, 1)

    # ★ [요청하신 기능] 전체 검색 성공률 비율 칼럼 추가 (모든 행에 동일한 값 저장)
    # (성공한 횟수 / 전체 횟수) * 100
    overall_search_rate = res_df['Score_SearchSuccess'].mean() * 100
    res_df['Overall_Search_Success_Rate'] = f"{overall_search_rate:.1f}%" # 👈 한눈에 보는 성공률

    # 콘솔 리포트
    print("\n" + "="*40)
    print("🏆  FINAL 5-KPI REPORT  🏆")
    print("="*40)
    print(f"1. 정확성 (Accuracy)       : {res_df['Score_Total_Accuracy'].mean():.1f}점")
    print(f"   - Logic Avg             : {res_df['Score_Accuracy_Logic'].mean():.1f}점")
    print(f"   - Math Avg              : {res_df['Score_Accuracy_Math'].mean():.1f}점")
    print(f"2. 안정성 (Stability)      : {res_df['Score_Stability'].mean()*100:.1f}%")
    print(f"3. 검색 성공률 (Success)    : {overall_search_rate:.1f}% (Total Ratio)") # 콘솔에도 표시
    print(f"4. 일관성 (Consistency)    : {res_df['Score_Consistency'].mean():.2f}")
    print(f"5. 다양성 (Diversity)      : {diversity:.1f}%")

    output_path = os.path.join(current_dir, "final_kpi_report.csv")
    res_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 리포트 저장 완료: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())