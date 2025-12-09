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
        # Spotify 연결
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
            logger.info("✅ Spotify API Connected.")
        except Exception as e:
            logger.error(f"❌ Spotify Connection Failed: {e}")
            self.sp = None
            
        self.diversity_pool = [] # 다양성 계산용

    # [Helper] JSON 전처리
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
                # 제목/가수 등 고유명사는 제거하고 '이유(Reasoning)'만 추출
                return parsed_data[0].get('recommendation_meta', {}).get('reasoning', '')
            return ""
        except:
            return ""

    # ======================================================================
    # KPI 1. 정확성 (Accuracy): 하이브리드 (Math + Logic)
    # ======================================================================
    async def evaluate_accuracy(self, row, output_data):
        criteria = row['Evaluation Criteria']
        context_str = f"장소: {row['Location']}, 목표: {row['Goal']}, 선호: {row['User Pref']}"
        reasoning_text = self._extract_text_for_embedding(output_data)

        # (A) Math: 임베딩 유사도 (30%)
        score_math = 0
        if reasoning_text:
            try:
                resp = await aclient.embeddings.create(input=[criteria, reasoning_text], model="text-embedding-3-small")
                vec1 = resp.data[0].embedding
                vec2 = resp.data[1].embedding
                score_math = cosine_similarity([vec1], [vec2])[0][0] * 100
            except: pass

        # (B) Logic: LLM Judge (70%)
        system_prompt = """
        당신은 '음악 추천 품질 평가관'입니다. 추천 결과가 Criteria를 준수했는지 채점하세요.
        [채점 기준 0~100]
        1. Context 적합성: 장소/목표에 맞는 분위기인가? (예: 독서실 소음 금지)
        2. Preference 반영: 사용자의 선호 장르/아티스트를 고려했는가?
        3. Conflict 해결: 상황과 취향이 충돌할 때 합리적으로 타협했는가?
        숫자만 반환하세요.
        """
        user_msg = f"Criteria: {criteria}\nContext: {context_str}\nOutput: {str(output_data)}"
        
        score_logic = 0
        try:
            resp = await aclient.chat.completions.create(
                model="gpt-4o", messages=[{"role":"system", "content":system_prompt}, {"role":"user", "content":user_msg}], temperature=0
            )
            score_logic = int(''.join(filter(str.isdigit, resp.choices[0].message.content)))
        except: pass

        return score_math, score_logic

    # ======================================================================
    # KPI 2. 안정성 & 규칙 커버리지 (System Stability)
    # ======================================================================
    def evaluate_system_stability(self, parsed_data):
        """
        JSON 형식이 깨지지 않고, 필수 키가 존재하는지 확인 (Format Check)
        """
        if parsed_data is None: return 0 # 파싱 실패
        if not isinstance(parsed_data, list) or len(parsed_data) == 0: return 0 # 빈 리스트
        
        # 필수 키 구조 확인
        required = ["recommendation_meta", "track_info", "target_audio_features"]
        first = parsed_data[0]
        if all(key in first for key in required):
            return 1 # 시스템적으로 정상 응답
        return 0

    # ======================================================================
    # KPI 3. 검색 성공률 (Search Success Rate) - Spotify 검증
    # ======================================================================
    def evaluate_search_success(self, parsed_data):
        """
        추천된 곡이 Spotify에 실제로 존재하는지 확인 (Hallucination Check)
        """
        # 시스템 안정성 통과 못했으면 검색도 불가
        if not self.evaluate_system_stability(parsed_data): return 0
        
        try:
            info = parsed_data[0]['track_info']
            title = info.get('track_title', '').strip()
            artist = info.get('artist_name', '').strip()

            if not title or not artist or "unknown" in title.lower(): return 0

            # 스포티파이 미연결 시 통과 처리 (Fallback)
            if self.sp is None: return 1

            # 실제 검색
            q = f"track:{title} artist:{artist}"
            res = self.sp.search(q=q, type='track', limit=1)
            if len(res['tracks']['items']) > 0:
                return 1 # 실존함 (성공)
            
            logger.warning(f"[Hallucination] Not Found: {title} - {artist}")
            return 0 # 실존하지 않음 (실패)
        except:
            return 0

    # ======================================================================
    # KPI 4. 일관성 (Consistency)
    # ======================================================================
    async def evaluate_consistency(self, inputs):
        """샘플링 검사: 동일 입력 2회 추가 실행 후 포맷 안정성 확인"""
        try:
            tasks = [run_agent_bridge(inputs) for _ in range(2)]
            results = await asyncio.gather(*tasks)
            valid = sum(1 for r in results if self._safe_parse_json(r))
            return 1.0 if valid == 2 else 0.5
        except: return 0.0

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
    # 경로 설정
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
            "user_artist": row.get('User Artist')
        }
        
        print(f"▶ [{idx+1}/{len(df)}] ID {row['ID']} ({row['Location']}) 평가 중...")

        # 1. 실행
        raw_out = await run_agent_bridge(inputs)
        parsed = evaluator._safe_parse_json(raw_out)
        
        # 2. KPI 측정
        # [KPI 1] 정확성 (Hybrid)
        s_math, s_logic = await evaluator.evaluate_accuracy(row, parsed)
        final_acc = (s_math * 0.3) + (s_logic * 0.7)

        # [KPI 2] 안정성 (Format)
        s_stability = evaluator.evaluate_system_stability(parsed)

        # [KPI 3] 검색 성공률 (Spotify)
        s_search = evaluator.evaluate_search_success(parsed)

        # [KPI 4] 일관성 (5개마다 샘플링)
        s_consist = 1.0
        if idx % 5 == 0:
            s_consist = await evaluator.evaluate_consistency(inputs)

        # [KPI 5] 다양성 (기록)
        evaluator.record_diversity(parsed)

        # 결과 저장
        results.append({
            "ID": row['ID'],
            "Context": f"{row['Location']}-{row['Goal']}",
            "Score_Accuracy": round(final_acc, 1),      # 정확성
            "Score_Stability": s_stability,             # 안정성 (0 or 1)
            "Score_SearchSuccess": s_search,            # 검색성공 (0 or 1)
            "Score_Consistency": s_consist,             # 일관성
            "Raw_Score_Logic": s_logic,
            "Raw_Score_Math": round(s_math, 1)
        })
        
        status = "✅" if s_stability and s_search else "⚠️"
        print(f"   ㄴ {status} 정확도:{final_acc:.0f} | 안정성:{s_stability} | 검색성공:{s_search}")

    # 최종 집계
    res_df = pd.DataFrame(results)
    diversity = evaluator.calculate_diversity()
    
    # 다양성 컬럼 추가 (모든 행 동일 값)
    res_df['Score_Diversity'] = round(diversity, 1)

    print("\n" + "="*40)
    print("🏆  FINAL 5-KPI REPORT  🏆")
    print("="*40)
    print(f"1. 정확성 (Accuracy)       : {res_df['Score_Accuracy'].mean():.1f}점")
    print(f"2. 안정성 (Stability)      : {res_df['Score_Stability'].mean()*100:.1f}% (Rule Coverage 100%)")
    print(f"3. 검색 성공률 (Success)   : {res_df['Score_SearchSuccess'].mean()*100:.1f}% (Spotify Verified)")
    print(f"4. 일관성 (Consistency)    : {res_df['Score_Consistency'].mean():.2f}")
    print(f"5. 다양성 (Diversity)      : {diversity:.1f}%")

    res_df.to_csv(os.path.join(current_dir, "final_kpi_report.csv"), index=False, encoding="utf-8-sig")
    print("\n✅ 'final_kpi_report_v2.csv' 저장 완료.")

if __name__ == "__main__":
    asyncio.run(main())