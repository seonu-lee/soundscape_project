### **전체 구조도 (Flow)**

>1. 평가셋 (CSV): 문제지 (Location, Goal 등 입력값) + 정답 기준(Criteria)
>
>2. 추천 엔진 (LangGraph): 문제를 풀어서 답을 도출 (추천 프롬프트)
>
>3. KPI 코드 (Evaluator): 답안지를 채점 (채점 프롬프트)