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

# 외부 라이브러리 콘솔 제거 (TMI 제거) ▼
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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

    # [NEW] Primary Tag 추출 헬퍼 함수
    def _extract_primary_tag(self, parsed_data):
        try:
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                return parsed_data[0].get('recommendation_meta', {}).get('primary_tag', 'unknown')
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
        tags = []
        if first_parsed_data:
            tag1 = first_parsed_data[0].get('recommendation_meta', {}).get('primary_tag', 'error')
            tags.append(tag1)
        else:
            tags.append("error_1")

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

        if not tags: return 0.0
        
        from collections import Counter
        counts = Counter(tags)
        most_common_count = counts.most_common(1)[0][1] 
        score = most_common_count / len(tags)
        
        if score < 1.0:
            logger.info(f"ℹ️ Consistency Diff: {tags}")
            
        return score
    
    # ======================================================================
    # KPI 5. 다양성 (Diversity) 
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
        
        s_consist = 1.0 # 기본값
        if idx % 5 == 0: 
            s_consist = await evaluator.evaluate_consistency(inputs, parsed)
        evaluator.record_diversity(parsed)

        # 3. 할루시네이션 및 트랙 정보 추출
        track_info_str = evaluator._extract_track_info_str(parsed)
        primary_tag_str = evaluator._extract_primary_tag(parsed)  # 👈 Tag 추출
        hallucinated_track = ""
        
        RED = "\033[91m"
        RESET = "\033[0m"
        
        if s_search == 0 and s_stability == 1:
            hallucinated_track = track_info_str
            print(f"{RED}❌ Hallucination: {hallucinated_track}{RESET}", end=" ")
        
        # 결과 리스트에 추가
        results.append({
            "ID": row.get('ID', idx),
            "Context": f"{row['Location']}-{row['Goal']}",
            "Score_Total_Accuracy": round(final_acc, 1),
            "Score_Accuracy_Logic": s_logic,          
            "Score_Accuracy_Math": round(s_math, 1),  
            "Score_Stability": s_stability,             
            "Score_SearchSuccess": s_search,          
            "Score_Consistency": s_consist,
            "Primary_Tag": primary_tag_str,           # 👈 컬럼 추가됨!
            "Hallucination_Track": hallucinated_track, 
            "Output_Reasoning": evaluator._extract_text_for_embedding(parsed),
            "Recommended_Track": track_info_str
        })
        
        if not hallucinated_track:
            print(f"✅ Acc:{final_acc:.0f}")

    # 4. 최종 리포트 데이터프레임 생성
    res_df = pd.DataFrame(results)
    
    # 다양성 및 전체 성공률 계산
    diversity = evaluator.calculate_diversity()
    res_df['Score_Diversity'] = round(diversity, 1)
    
    overall_search_rate = res_df['Score_SearchSuccess'].mean() * 100
    res_df['Overall_Search_Success_Rate'] = f"{overall_search_rate:.1f}%"

    # 5. 콘솔 출력 및 요약 CSV 저장
    print("\n" + "="*40)
    print("🏆  FINAL 5-KPI REPORT  🏆")
    print("="*40)
    
    # 평균값 계산
    avg_accuracy = res_df['Score_Total_Accuracy'].mean()
    avg_logic = res_df['Score_Accuracy_Logic'].mean()
    avg_math = res_df['Score_Accuracy_Math'].mean()
    avg_stability = res_df['Score_Stability'].mean() * 100
    avg_consistency = res_df['Score_Consistency'].mean()
    
    print(f"1. 정확성 (Accuracy)       : {avg_accuracy:.1f}점")
    print(f"   - Logic Avg             : {avg_logic:.1f}점")
    print(f"   - Math Avg              : {avg_math:.1f}점")
    print(f"2. 안정성 (Stability)      : {avg_stability:.1f}%")
    print(f"3. 검색 성공률 (Success)    : {overall_search_rate:.1f}% (Total Ratio)")
    print(f"4. 일관성 (Consistency)    : {avg_consistency:.2f}")
    print(f"5. 다양성 (Diversity)      : {diversity:.1f}%")

    # (1) 상세 리포트 저장
    detail_path = os.path.join(current_dir, "final_kpi_report.csv")
    res_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 상세 리포트 저장 완료: {detail_path}")
    
    # (2) 요약 리포트 저장 (요청하신 기능)
    summary_data = [{
        "KPI_Name": "Accuracy (Total)", "Score": f"{avg_accuracy:.1f}"
    }, {
        "KPI_Name": "Accuracy (Logic)", "Score": f"{avg_logic:.1f}"
    }, {
        "KPI_Name": "Accuracy (Math)", "Score": f"{avg_math:.1f}"
    }, {
        "KPI_Name": "Stability", "Score": f"{avg_stability:.1f}%"
    }, {
        "KPI_Name": "Search Success Rate", "Score": f"{overall_search_rate:.1f}%"
    }, {
        "KPI_Name": "Consistency", "Score": f"{avg_consistency:.2f}"
    }, {
        "KPI_Name": "Diversity", "Score": f"{diversity:.1f}%"
    }]
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(current_dir, "summary_report.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"✅ 요약 리포트 저장 완료: {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())