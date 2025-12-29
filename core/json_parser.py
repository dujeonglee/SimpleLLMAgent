import json
import re
import ast
from typing import Any


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
    
    # 1. 먼저 그대로 파싱 시도
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass

    # 2. Incomplete code block 처리 (```로 시작했는데 끝이 없는 경우 등)
    sanitized = sanitize_code_blocks(json_string)

    # 3. 바깥쪽 코드 블록만 제거 (내부 ```는 보존)
    cleaned = _remove_outer_code_block(sanitized)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. 문자열 값 내부의 이스케이프 안 된 제어 문자 처리
    try:
        fixed = _escape_control_chars_in_strings(cleaned)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 5. 작은따옴표를 큰따옴표로 변환 (Python dict 스타일)
    try:
        converted = _convert_single_to_double_quotes(cleaned)
        return json.loads(converted)
    except json.JSONDecodeError:
        pass

    # 6. 4번 + 5번 조합
    try:
        converted = _convert_single_to_double_quotes(cleaned)
        fixed = _escape_control_chars_in_strings(converted)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 7. 마지막 시도: ast.literal_eval (Python 리터럴)
    try:
        return ast.literal_eval(json_string.strip())
    except (ValueError, SyntaxError):
        pass

    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass
    
    return None


def sanitize_code_blocks(text: str) -> str:
    """
    코드블럭 마크다운의 sanity check (incomplete code block 처리)

    - ``` 로 시작했는데 ``` 로 끝나지 않으면 마지막에 추가
    - ``` 로 끝났는데 ``` 시작이 없으면 처음에 추가

    Args:
        text: 검사할 텍스트

    Returns:
        sanitized 텍스트
    """
    if not text:
        return text

    # 모든 ``` 패턴 찾기
    patterns = list(re.finditer(r'```', text))

    if not patterns:
        return text

    # 첫 번째와 마지막 ``` 위치
    first_match = patterns[0]
    last_match = patterns[-1]

    result = text

    # 첫 번째 ``` 앞에 텍스트가 있는지 확인 (앞에 공백/줄바꿈만 있으면 시작으로 간주)
    before_first = text[:first_match.start()].strip()
    has_content_before_first = len(before_first) > 0

    # 마지막 ``` 뒤에 텍스트가 있는지 확인 (뒤에 공백/줄바꿈만 있으면 끝으로 간주)
    after_last = text[last_match.end():].strip()
    has_content_after_last = len(after_last) > 0

    # ``` 개수가 홀수면 짝이 맞지 않음
    if len(patterns) % 2 == 1:
        # 첫 번째 앞에 내용이 없으면 시작 블록, 끝이 없으므로 추가
        if not has_content_before_first:
            result = result.rstrip() + "\n```\n"
        # 첫 번째 앞에 내용이 있으면 끝 블록이 먼저 나온 것, 시작 추가
        else:
            result = "```\n" + result.lstrip()

    return result


def _remove_outer_code_block(s: str) -> str:
    """
    가장 바깥쪽 코드 블록만 제거 (첫 줄과 마지막 줄만 체크)
    내부에 있는 ```는 보존됨
    """
    lines = s.strip().split('\n')

    if len(lines) < 2:
        return s.strip()

    first_line = lines[0].strip()
    last_line = lines[-1].strip()

    # 첫 줄이 ```로 시작하고 (언어 태그 있어도 됨), 마지막 줄이 ```만 있는 경우
    if re.match(r'^```\w*$', first_line) and last_line == '```':
        return '\n'.join(lines[1:-1])

    return s.strip()


def _escape_control_chars_in_strings(s: str) -> str:
    """
    JSON 문자열 값 내부의 이스케이프 안 된 제어 문자를 처리
    
    처리 대상: \n, \r, \t, \b, \f 및 기타 제어 문자 (0x00-0x1F)
    """
    result = []
    in_string = False
    escape_next = False
    
    for char in s:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            result.append(char)
            continue
        
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        
        if in_string:
            # 제어 문자 이스케이프 (0x00 ~ 0x1F)
            if ord(char) < 0x20:
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif char == '\b':
                    result.append('\\b')
                elif char == '\f':
                    result.append('\\f')
                else:
                    # 기타 제어 문자는 \uXXXX 형식으로
                    result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        else:
            result.append(char)
    
    return ''.join(result)


def _convert_single_to_double_quotes(s: str) -> str:
    """
    Python 스타일 딕셔너리를 JSON으로 변환 (작은따옴표 -> 큰따옴표)
    문자열 내부의 따옴표는 적절히 이스케이프
    """
    result = []
    in_double_string = False
    in_single_string = False
    escape_next = False
    
    for i, char in enumerate(s):
        if escape_next:
            # 이스케이프된 작은따옴표를 일반 작은따옴표로
            if char == "'" and not in_double_string:
                result.append("'")
            # 이스케이프된 큰따옴표 유지
            elif char == '"':
                result.append('\\"')
            else:
                result.append(char)
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            result.append(char)
            continue
        
        if char == '"' and not in_single_string:
            in_double_string = not in_double_string
            result.append(char)
        elif char == "'" and not in_double_string:
            if not in_single_string:
                # 문자열 시작: 작은따옴표 -> 큰따옴표
                in_single_string = True
                result.append('"')
            else:
                # 문자열 끝: 작은따옴표 -> 큰따옴표
                in_single_string = False
                result.append('"')
        elif char == '"' and in_single_string:
            # 작은따옴표 문자열 내부의 큰따옴표는 이스케이프
            result.append('\\"')
        else:
            result.append(char)
    
    return ''.join(result)


def parse_json_strict(json_string: str) -> dict | list | Any:
    """
    파싱 실패시 예외를 발생시키는 버전
    
    Args:
        json_string: 파싱할 JSON 문자열
        
    Returns:
        파싱된 Python 객체
        
    Raises:
        ValueError: 파싱 실패시
    """
    result = parse_json_robust(json_string)
    if result is None:
        preview = json_string[:100] + '...' if len(json_string) > 100 else json_string
        raise ValueError(f"JSON 파싱 실패: {preview}")
    return result


# ============================================================
# 테스트
# ============================================================
if __name__ == "__main__":
    test_cases = [
        # 1. 기본 JSON
        ('{"name": "test", "value": 123}', "기본 JSON"),
        
        # 2. 특수 문자 포함
        ('{"message": "He said \\"Hello\\"", "code": "it\'s working"}', "이스케이프된 따옴표"),
        
        # 3. 백틱 포함
        ('{"template": "Use `code` here"}', "백틱 포함"),
        
        # 4. 코드 블록 (json)
        ('```json\n{"key": "value"}\n```', "JSON 코드 블록"),
        
        # 5. 코드 블록 (다른 언어)
        ('```python\n{"key": "value"}\n```', "Python 코드 블록"),
        
        # 6. 내부에 코드 블록 포함
        ('```json\n{"code": "Use ```python``` here"}\n```', "내부 코드 블록 포함"),
        
        # 7. Python 스타일 (작은따옴표)
        ("{'name': 'test', 'value': 123}", "Python 스타일"),
        
        # 8. 혼합 특수 문자
        ('{"text": "He said \\"it\'s `great`\\""}', "혼합 특수 문자"),
        
        # 9. 이스케이프 안 된 줄바꿈
        ('{"multiline": "line1\nline2"}', "줄바꿈 포함"),
        
        # 10. 이스케이프 안 된 탭
        ('{"data": "col1\tcol2"}', "탭 포함"),
        
        # 11. Python 스타일 + 내부 큰따옴표
        ("{'msg': 'He said \"hello\"'}", "Python 스타일 + 큰따옴표"),
        
        # 12. 빈 객체/배열
        ('{}', "빈 객체"),
        ('[]', "빈 배열"),
        
        # 13. 중첩 구조
        ('{"outer": {"inner": "value"}}', "중첩 객체"),
        
        # 14. 배열
        ('[1, 2, {"key": "value"}]', "배열"),
        
        # 15. 유니코드
        ('{"korean": "한글 테스트", "emoji": "😀"}', "유니코드"),

        # 16. Nested JSON 코드 블록
        ('```json\n{"key": "```json\ncodeA\n``` and ```json\ncodeB\n```"}\n```', "Nested JSON 코드 블록"),

        # 17. Nested 코드 블록
        ('```\n{"key": "```json\ncodeA\n``` and ```json\ncodeB\n```"}\n```', "Nested 코드 블록"),

        # 18. Incomplete 코드 블록 (json)
        ('```json\n{"key": "value"}\n', "Incomplete JSON 코드 블록"),

        # 19. Incomplete Nested 코드 블록
        ('```\n{"key": "```json\ncodeA\n``` and ```json\ncodeB\n```"}\n', "Incomplete Nested 코드 블록"),
    ]
    
    print("=" * 60)
    print("JSON 파서 테스트")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for json_str, description in test_cases:
        result = parse_json_robust(json_str)
        status = "✅" if result is not None else "❌"
        
        if result is not None:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} {description}")
        print(f"   입력: {json_str[:50]}{'...' if len(json_str) > 50 else ''}")
        print(f"   결과: {result}")
    
    print("\n" + "=" * 60)
    print(f"결과: {passed}/{len(test_cases)} 통과")
    print("=" * 60)