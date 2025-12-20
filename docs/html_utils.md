# HTML Utilities 설계 문서

## 개요

`html_utils.py`는 HTML 파싱, 요약, 미리보기 기능을 제공합니다.
web_tool.fetch로 가져온 HTML을 안전하게 처리하고 표시하는 데 사용됩니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| HTML 감지 | 콘텐츠가 HTML인지 판단 |
| 요약 추출 | Title, Description, 텍스트 미리보기 |
| 보안 처리 | 스크립트, 이벤트 핸들러 제거 |
| 저장/조회 | 미리보기용 HTML 임시 저장 |

## 클래스 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                      HTMLParser                             │
├─────────────────────────────────────────────────────────────┤
│ + is_html(content) -> bool                                  │
│ + extract_title(html) -> str                                │
│ + extract_description(html) -> str                          │
│ + strip_tags(html) -> str                                   │
│ + extract_text_preview(html, max_length) -> str             │
│ + generate_hash(content) -> str                             │
│ + has_dangerous_content(html) -> Tuple[bool, bool]          │
│ + summarize(html) -> HTMLSummary                            │
│ + sanitize_for_iframe(html) -> str                          │
│ + format_for_chat(html, hash) -> str                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ creates
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      HTMLSummary                            │
├─────────────────────────────────────────────────────────────┤
│ + title: str                                                │
│ + text_preview: str                                         │
│ + html_hash: str                                            │
│ + char_count: int                                           │
│ + has_forms: bool                                           │
│ + has_scripts: bool                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 HTML Storage (Module-level)                 │
├─────────────────────────────────────────────────────────────┤
│ store_html(html) -> str (hash)                              │
│ get_html(hash) -> Optional[str]                             │
│ clear_html_storage()                                        │
└─────────────────────────────────────────────────────────────┘
```

## HTMLParser 클래스

### is_html(content: str) -> bool

콘텐츠가 HTML인지 판단합니다.

```python
HTMLParser.is_html("<!DOCTYPE html><html>...")  # True
HTMLParser.is_html("<html><body>...")           # True
HTMLParser.is_html("<div>content</div>")        # True
HTMLParser.is_html("plain text")                # False
```

판단 기준:
- `<!doctype html` 로 시작
- `<html` 로 시작
- 처음 500자 내에 `<html` 포함
- 처음 1000자 내에 HTML 태그 (`<head>`, `<body>`, `<div>` 등) 포함

### extract_title(html: str) -> str

HTML에서 제목을 추출합니다.

우선순위:
1. `<title>` 태그
2. `og:title` 메타 태그
3. 첫 번째 `<h1>` 태그
4. "(제목 없음)" 반환

```python
HTMLParser.extract_title("<title>My Page</title>")  # "My Page"
```

### extract_description(html: str) -> str

HTML에서 설명을 추출합니다.

우선순위:
1. `meta name="description"` 태그
2. `og:description` 메타 태그
3. 빈 문자열 반환

### strip_tags(html: str) -> str

HTML 태그를 제거하고 텍스트만 추출합니다.

처리 순서:
1. `<script>`, `<style>` 태그 내용 제거
2. HTML 주석 제거
3. 모든 태그 제거
4. HTML 엔티티 디코딩
5. 연속 공백 정리

```python
HTMLParser.strip_tags("<p>Hello <b>World</b></p>")  # "Hello World"
```

### extract_text_preview(html: str, max_length: int = 300) -> str

HTML에서 텍스트 미리보기를 추출합니다.

1. description이 있고 50자 이상이면 사용
2. 없으면 body 내용에서 텍스트 추출
3. max_length로 자르고 "..." 추가

### generate_hash(content: str) -> str

콘텐츠의 MD5 해시를 생성합니다 (앞 12자리).

```python
HTMLParser.generate_hash("<html>...")  # "a1b2c3d4e5f6"
```

### has_dangerous_content(html: str) -> Tuple[bool, bool]

위험한 콘텐츠 포함 여부를 확인합니다.

```python
has_scripts, has_forms = HTMLParser.has_dangerous_content(html)
# has_scripts: <script> 태그 포함 여부
# has_forms: <form> 태그 포함 여부
```

### summarize(html: str) -> HTMLSummary

HTML 요약 정보를 생성합니다.

```python
summary = HTMLParser.summarize(html)
print(summary.title)        # "Google"
print(summary.text_preview) # "Google 서비스에 오신 것을..."
print(summary.html_hash)    # "a1b2c3d4e5f6"
print(summary.char_count)   # 45234
print(summary.has_scripts)  # True
print(summary.has_forms)    # False
```

### sanitize_for_iframe(html: str) -> str

iframe에서 안전하게 표시하기 위해 HTML을 정리합니다.

제거 항목:
- `<script>` 태그 및 내용
- `on*` 이벤트 핸들러 (`onclick`, `onload` 등)
- `javascript:` URL

```python
sanitized = HTMLParser.sanitize_for_iframe(html)
# <script>alert('xss')</script> → 제거됨
# onclick="..." → 제거됨
# href="javascript:..." → href="#"
```

### format_for_chat(html: str, html_hash: str) -> str

채팅에 표시할 HTML 요약 포맷을 생성합니다.

```
📄 **Google**

Google 서비스에 오신 것을 환영합니다. 검색, 메일, 지도...

📊 크기: 45.2KB ⚠️ 스크립트 포함

🔑 HTML ID: `a1b2c3d4e5f6` (미리보기/복사에 사용)
```

## HTML Storage

메모리 기반 HTML 임시 저장소입니다.

### store_html(html: str) -> str

HTML을 저장하고 해시를 반환합니다.

```python
html_hash = store_html("<html>...</html>")
# "a1b2c3d4e5f6"
```

### get_html(html_hash: str) -> Optional[str]

저장된 HTML을 가져옵니다.

```python
html = get_html("a1b2c3d4e5f6")
if html:
    print(html)  # "<html>...</html>"
```

### clear_html_storage()

저장소를 비웁니다.

```python
clear_html_storage()
```

## 사용 예시

### web_tool.fetch 결과 처리

```python
from core.html_utils import HTMLParser, store_html, get_html

# fetch 결과가 HTML인지 확인
result = web_tool.fetch(url="https://example.com")
content = result.output

if HTMLParser.is_html(content):
    # HTML 저장
    html_hash = store_html(content)
    
    # 요약 생성
    summary = HTMLParser.summarize(content)
    
    # 채팅에 표시할 포맷
    chat_display = f"""
📄 **{summary.title}**

{summary.text_preview}

📊 크기: {summary.char_count / 1024:.1f}KB
🔑 HTML ID: `{html_hash}`
"""
```

### HTML 미리보기

```python
# 저장된 HTML 가져오기
html = get_html("a1b2c3d4e5f6")

if html:
    # iframe용으로 정리
    safe_html = HTMLParser.sanitize_for_iframe(html)
    
    # iframe 생성
    iframe = f'<iframe srcdoc="{safe_html}" sandbox="allow-same-origin"></iframe>'
```

## 보안 고려사항

### 왜 sanitize가 필요한가?

외부 웹페이지의 HTML을 그대로 렌더링하면:
- XSS 공격 가능 (악성 스크립트 실행)
- 피싱 폼 표시 가능
- 사용자 정보 탈취 가능

### sanitize_for_iframe이 제거하는 것

| 항목 | 예시 | 위험 |
|------|------|------|
| script 태그 | `<script>alert('xss')</script>` | 스크립트 실행 |
| 이벤트 핸들러 | `onclick="..."`, `onload="..."` | 스크립트 실행 |
| javascript URL | `href="javascript:..."` | 스크립트 실행 |

### iframe sandbox 속성

```html
<iframe 
    srcdoc="..." 
    sandbox="allow-same-origin"
>
```

sandbox 제한:
- 스크립트 실행 차단 (allow-scripts 없음)
- 폼 제출 차단 (allow-forms 없음)
- 팝업 차단 (allow-popups 없음)

## 파일 위치

```
multi_agent_chatbot/
├── core/
│   ├── html_utils.py      ← HTML 유틸리티
│   └── __init__.py        ← HTMLParser, store_html 등 export
├── app.py                 ← format_tool_result, preview_html에서 사용
└── docs/
    └── html_utils.md      ← 이 문서
```

## 테스트

```python
from core.html_utils import HTMLParser, store_html, get_html

# HTML 감지 테스트
assert HTMLParser.is_html("<!DOCTYPE html><html></html>") == True
assert HTMLParser.is_html("plain text") == False

# 제목 추출 테스트
assert HTMLParser.extract_title("<title>Test</title>") == "Test"

# 저장/조회 테스트
html = "<html><body>Hello</body></html>"
hash = store_html(html)
assert get_html(hash) == html

# sanitize 테스트
dirty = '<div onclick="alert()">content</div>'
clean = HTMLParser.sanitize_for_iframe(dirty)
assert 'onclick' not in clean

print("All tests passed!")
```