"""
Markdown Sanitizer Utilities
"""
import re


def sanitize_code_blocks(text: str) -> str:
    """
    코드블럭 마크다운의 간단한 sanity check

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


# =============================================================================
# Test Cases
# =============================================================================

def _run_tests():
    """테스트 케이스 실행"""
    test_cases = [
        # Case 1: 정상적인 코드블럭 (수정 불필요)
        {
            "input": "```python\nprint('hello')\n```",
            "expected": "```python\nprint('hello')\n```",
            "description": "정상적인 코드블럭"
        },

        # Case 2: 시작만 있고 끝이 없음
        {
            "input": "```python\nprint('hello')\n",
            "expected": "```python\nprint('hello')\n```\n",
            "description": "시작만 있고 끝이 없음"
        },

        # Case 3: 끝만 있고 시작이 없음
        {
            "input": "print('hello')\n```",
            "expected": "```\nprint('hello')\n```",
            "description": "끝만 있고 시작이 없음"
        },

        # Case 4: 앞뒤에 공백이 있는 경우
        {
            "input": "  ```python\nprint('hello')\n",
            "expected": "  ```python\nprint('hello')\n```\n",
            "description": "앞에 공백이 있고 끝이 없음"
        },

        # Case 5: 뒤에 공백이 있는 경우
        {
            "input": "```python\nprint('hello')\n```  ",
            "expected": "```python\nprint('hello')\n```  ",
            "description": "정상 코드블럭 (뒤에 공백)"
        },

        # Case 6: 빈 문자열
        {
            "input": "",
            "expected": "",
            "description": "빈 문자열"
        },

        # Case 7: None
        {
            "input": None,
            "expected": None,
            "description": "None 입력"
        },

        # Case 8: 코드블럭이 없는 일반 텍스트
        {
            "input": "This is just normal text",
            "expected": "This is just normal text",
            "description": "코드블럭 없는 일반 텍스트"
        },

        # Case 9: 여러 줄에 걸친 시작 (줄바꿈 포함)
        {
            "input": "\n\n```python\ndef hello():\n    pass\n",
            "expected": "\n\n```python\ndef hello():\n    pass\n```\n",
            "description": "줄바꿈 후 시작, 끝 없음"
        },

        # Case 10: 끝에 줄바꿈 포함
        {
            "input": "def hello():\n    pass\n```\n\n",
            "expected": "```\ndef hello():\n    pass\n```\n\n",
            "description": "시작 없음, 끝에 줄바꿈 포함"
        },
    ]

    print("=" * 60)
    print("Testing sanitize_code_blocks()")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        input_text = test["input"]
        expected = test["expected"]
        description = test["description"]

        result = sanitize_code_blocks(input_text)

        if result == expected:
            print(f"[PASS] Test {i}: {description}")
            passed += 1
        else:
            print(f"[FAIL] Test {i}: {description}")
            print(f"   Input:    {repr(input_text)}")
            print(f"   Expected: {repr(expected)}")
            print(f"   Got:      {repr(result)}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    _run_tests()
