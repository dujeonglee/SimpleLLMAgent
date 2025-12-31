import json
import re
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

def extract_code_blocks(text: str) -> List[Dict]:
    """코드 블록 상세 정보 추출"""
    pattern = r'```(\w*)\n(.*?)```'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    return [
        {
            "language": match.group(1) or None,
            "content": match.group(2).strip(),
            "start": match.start(),
            "end": match.end()
        }
        for match in matches
    ]

def _save_failed_json(json_string: str, error_msg: str = ""):
    """
    파싱 실패한 JSON 문자열을 디버깅용 파일에 저장

    Args:
        json_string: 파싱 실패한 JSON 문자열
        error_msg: 에러 메시지 (선택)
    """
    try:
        # 디버그 디렉토리 생성
        debug_dir = os.path.join(os.path.dirname(__file__), "..", "debug", "failed_json")
        os.makedirs(debug_dir, exist_ok=True)

        # 타임스탬프 기반 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"failed_{timestamp}.txt"
        filepath = os.path.join(debug_dir, filename)

        # 파일 저장
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"Failed JSON Parsing - {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")

            if error_msg:
                f.write(f"Error: {error_msg}\n\n")

            f.write("Original String:\n")
            f.write("-" * 60 + "\n")
            f.write(json_string)
            f.write("\n" + "-" * 60 + "\n\n")

            f.write(f"Length: {len(json_string)} chars\n")
            f.write(f"Preview: {json_string[:200]}...\n" if len(json_string) > 200 else f"Content: {json_string}\n")

        print(f"[DEBUG] Failed JSON saved to: {filepath}")
    except Exception as e:
        # 저장 실패해도 파싱 로직에 영향 없도록
        print(f"[WARNING] Failed to save debug file: {e}")


def parse_json_robust(json_string: str) -> dict | list | Any | None:
    """
    특수 문자가 포함된 JSON 문자열을 안전하게 파싱합니다.

    처리 가능한 케이스:
    - 표준 JSON
    - 코드 블록으로 감싸진 JSON (```json ... ```)
    - Incomplete 코드 블록 (```json ... 으로 시작하고 끝이 없는 경우)
    - 이스케이프 안 된 제어 문자 (줄바꿈, 탭 등)
    - Python 스타일 딕셔너리 (작은따옴표)
    - 백틱, 작은따옴표 등 특수 문자 포함

    Args:
        json_string: 파싱할 JSON 문자열

    Returns:
        파싱된 Python 객체 (dict, list 등) 또는 실패시 None
    """
    if not json_string or not isinstance(json_string, str):
        return None

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # 파싱 실패 시 디버그 파일 저장
        _save_failed_json(json_string, str(e))
        pass
    
    try:
        code_block = extract_code_blocks(json_string)
        if len(code_block) > 1:
            _save_failed_json(json_string, f'More than one json objects in response : {len(code_block)}')
            return None
        return json.loads(code_block[0].get("content"))
    except json.JSONDecodeError as e:
        # 파싱 실패 시 디버그 파일 저장
        _save_failed_json(code_block[0].get("content"), str(e))
        pass

    return None
