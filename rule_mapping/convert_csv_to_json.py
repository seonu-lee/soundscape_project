'''평면 JSON 변환 스크립트'''

import csv
import json
import os

# 파일 경로 설정
csv_file = "rule_mapping_v2.5.csv"
json_file = os.path.splitext(csv_file)[0] + ".json"

# CSV → JSON 변환 함수
def csv_to_json(csv_path, json_path):
    data = []
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 공백 키 제거 + 숫자형 변환
            clean_row = {}
            for k, v in row.items():
                key = k.strip()
                value = v.strip() if isinstance(v, str) else v
                # 숫자형 문자열을 자동 변환
                if value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                clean_row[key] = value
            data.append(clean_row)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f" 변환 완료: {json_path}")

# 실행
if __name__ == "__main__":
    csv_to_json(csv_file, json_file)

'''터미널 열어서 python convert_csv_to_json.py
실행하면 파일 생성됨'''