import os
import pandas as pd
import numpy as np
import openai
import asyncio
import json
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ★ [연결] 님이 만든 랭그래프 엔진 가져오기
from my_agent import run_agent_bridge

# 1. 설정 로드
load_dotenv()
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------
# [KPI Class] 4대 관점 통합 평가기
# ---------------------------------------------------------
class UltimateEvaluator:
    def __init__(self):
        self.client = client
        self.all_recommendations = [] # 다양성 계산용 (모든 추천 곡 제목 저장)

    # --- 도구: JSON 문자열에서 텍스트 추출 (임베딩용) ---
    def _extract_content_for_embedding(self, json_str):
        try:
            # 마크다운 제거
            clean_str = json_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_str)
            
            # 리스트면 첫 번째 추천 곡의 이유와 제목만 추출
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                reason = item.get('recommendation_meta', {}).get('reasoning', '')
                title = item.get('track_info', {}).get('track_title', '')
                artist = item.get('track_info', {}).get('artist_name', '')
                return f"{reason} {title} {artist}"
            return clean_str # 파싱 실패하면 통째로 반환
        except:
            return json_str

    # --- 1. 정확성 (Accuracy): Hybrid (Math + LLM) ---
    def evaluate_accuracy(self, row, output):
        # (A) 수학적: 임베딩 유사도 (Topic Check)
        try:
            criteria = row['Evaluation Criteria']
            # JSON 전체가 아니라 핵심 내용(이유+곡명)만 발라내서 임베딩
            content_to_embed = self._extract_content_for_embedding(output)
            
            resp = self.client.embeddings.create(input=[criteria, content_to_embed], model="text-embedding-3-small")
            score_math = cosine_similarity([resp.data[0].embedding], [resp.data[1].embedding])[0][0] * 100
        except Exception as e:
            print(f"  Warning(Math): {e}")
            score_math = 0

        # (B) 논리적: LLM Judge (Context & Genre Check)
        system_prompt = """
        You are a strict 'Music Recommendation Auditor'.
        Compare the Agent's JSON Output with the Evaluation Criteria.
        
        [Criteria]
        1. Context Match: Does the song fit the Location/Goal? (e.g., No loud music in Library)
        2. Preference Match: Does it respect User's Genre preference?
        3. Conflict Resolution: If Context and Preference clash (e.g., Metal in Library), did the agent find a smart compromise?
        
        Score 0-100. Return ONLY the integer score.
        """
        
        user_msg = f"""
        [Context] {row['Location']} / {row['Goal']} (Decibel: {row['Decibel']})
        [User Pref] {row['User Pref']}
        [Evaluation Criteria] {row['Evaluation Criteria']}
        
        [Agent Output]
        {output}
        
        Score:
        """
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o", 
                messages=[{"role":"system","content":system_prompt}, {"role":"user","content":user_msg}],
                temperature=0
            )
            # 숫자만 추출
            score_llm = int(''.join(filter(str.isdigit, resp.choices[0].message.content)))
        except:
            score_llm = 0
            
        return score_math, score_llm

    # --- 2. 안정성 (Reliability): Success Rate ---
    def check_reliability(self, output):
        # 빈 값이 아니고, 에러 메시지가 없으며, 'track_info' 키가 포함되어 있는지 확인
        if output and len(output) > 10 and "error" not in output.lower():
            if "track_info" in output: # JSON 키 체크
                return 1 # Success
        return 0 # Fail

    # --- 4. 다양성 (Diversity): 전체 완료 후 계산 ---
    def add_to_diversity_pool(self, output):
        try:
            # 곡 제목만 추출해서 저장
            clean_str = output.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_str)
            if isinstance(data, list):
                for item in data:
                    title = item.get('track_info', {}).get('track_title', 'unknown')
                    self.all_recommendations.append(title)
        except:
            pass # 파싱 에러나면 다양성 집계 제외

    def calculate_final_diversity(self):
        # 중복되지 않은 추천 결과의 비율 (Unique / Total)
        if not self.all_recommendations: return 0
        unique_count = len(set(self.all_recommendations))
        total_count = len(self.all_recommendations)
        return (unique_count / total_count) * 100

# ---------------------------------------------------------
# [Main Loop] 실행 (비동기)
# ---------------------------------------------------------
async def main():
    # 1. 평가셋 로드
    csv_file = "evaluation_set_v2_criteria.csv"
    if not os.path.exists(csv_file):
        print(f"❌ '{csv_file}' 파일이 없습니다. 평가셋 생성 코드를 먼저 실행하세요.")
        return

    df = pd.read_csv(csv_file)
    evaluator = UltimateEvaluator()
    results = []
    
    print(f"🚀 평가 시작... (총 {len(df)}개 케이스)")
    print("-" * 60)

    # 2. 반복 실행
    for idx, row in df.iterrows():
        # Bridge에 넣을 입력값 구성
        inputs = {
            "location": row['Location'],
            "decibel": row['Decibel'],
            "goal": row['Goal'],
            "user_pref": row['User Pref'],
            # CSV에 'User Artist' 컬럼이 없으면 None 처리
            "user_artist": row['User Artist'] if 'User Artist' in row else None
        }
        
        print(f"▶ [{idx+1}/{len(df)}] ID {row['ID']} ({row['Location']}/{row['Goal']}) 평가 중...")
        
        # (1) 랭그래프 엔진 실행 (await 필수!)
        try:
            output = await run_agent_bridge(inputs)
        except Exception as e:
            print(f"  ❌ Engine Error: {e}")
            output = '{"error": "Runtime Error"}'

        # (2) 다양성 풀 저장
        evaluator.add_to_diversity_pool(output)
        
        # (3) KPI 측정
        score_math, score_llm = evaluator.evaluate_accuracy(row, output) # 정확성
        is_success = evaluator.check_reliability(output) # 안정성
        
        # (4) 통합 점수 (LLM 70% + Math 30%)
        final_score = (score_math * 0.3) + (score_llm * 0.7)

        results.append({
            "ID": row['ID'],
            "Context": f"{row['Location']}-{row['Goal']}",
            "Output_Snippet": output[:50] + "...", # 결과 요약
            "KPI_Math": round(score_math, 1),
            "KPI_Logic": score_llm,
            "KPI_Success": is_success,
            "Final_Score": round(final_score, 1)
        })
        
        print(f"  ㄴ 결과: {final_score:.1f}점 (Logic:{score_llm} / Math:{score_math:.0f}) | Success: {is_success}")

    # 3. 최종 집계 및 저장
    diversity_score = evaluator.calculate_final_diversity()
    result_df = pd.DataFrame(results)
    
    print("\n" + "="*30)
    print("🏆  ULTIMATE KPI REPORT  🏆")
    print("="*30)
    
    if len(result_df) > 0:
        avg_success = result_df['KPI_Success'].mean() * 100
        avg_logic = result_df['KPI_Logic'].mean()
        avg_math = result_df['KPI_Math'].mean()
        
        print(f"1. 시스템 안정성 (Success Rate) : {avg_success:.1f}%")
        print(f"2. 평균 정확도 (Logic + Math) : {avg_logic:.1f} (Logic) + {avg_math:.1f} (Math)")
        print(f"3. 추천 다양성 (Diversity)    : {diversity_score:.1f}%")
        
        # CSV 저장
        result_df.to_csv("final_kpi_report.csv", index=False, encoding="utf-8-sig")
        print(f"\n✅ 상세 결과가 'final_kpi_report.csv'에 저장되었습니다.")
    else:
        print("결과 데이터가 없습니다.")

if __name__ == "__main__":
    asyncio.run(main())