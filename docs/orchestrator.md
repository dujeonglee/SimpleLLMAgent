# Orchestrator 설계 문서

## 개요

Orchestrator는 Multi-Agent Chatbot의 **두뇌** 역할을 합니다.
사용자 쿼리를 받아 Tool들을 조율하고 최종 응답을 생성합니다.

## 핵심 원칙

```
ReAct (Reasoning + Acting) 패턴
- 매 step마다 현재 상태를 보고 다음 행동 결정
- 초기 계획 없이 단계별 진행
- 결과를 보고 유연하게 대응
```

## 전체 흐름

```
User Query
    ↓
┌─────────────────────────────────────────────┐
│  Orchestrator (ReAct Loop)                  │
│                                             │
│  while not (complete or max_step):          │
│      1. LLM에게 다음 행동 질문               │
│      2. Tool 실행                           │
│      3. 결과 저장 + 실시간 출력 (Streaming)  │
│      4. 완료 여부 판단                       │
│                                             │
└─────────────────────────────────────────────┘
    ↓
Final Response
```

## 설계 결정 사항

| 항목 | 결정 | 비고 |
|------|------|------|
| 계획 방식 | ReAct (단계별 결정) | 매 step마다 LLM 판단 |
| LLM 응답 형식 | Function Calling 스타일 | tool_calls 형식 |
| 최대 step 제한 | 제한 + LLM 판단 | 기본 10 step |
| 규칙 기반 결정 | 나중에 추가 | 패턴 발견되면 |
| Streaming | 실시간 출력 | Generator 방식 |
| LLM 설정 | 동적 변경 가능 | UI에서 실시간 조절 |

## 클래스 다이어그램

```
┌─────────────────────────────────────────────────────────┐
│                     Orchestrator                        │
├─────────────────────────────────────────────────────────┤
│ - tools: ToolRegistry                                   │
│ - storage: SharedStorage                                │
│ - llm_config: LLMConfig                                 │
│ - max_steps: int                                        │
│ - logger: DebugLogger                                   │
├─────────────────────────────────────────────────────────┤
│ + run(user_query) -> str                                │
│ + run_stream(user_query) -> Generator[StepInfo]         │
│ + update_llm_config(**kwargs)                           │
│ + stop()                                                │
├─────────────────────────────────────────────────────────┤
│ - _ask_llm(user_query) -> LLMResponse                   │
│ - _parse_llm_response(raw) -> LLMResponse               │
│ - _extract_json(text) -> Optional[str]                  │
│ - _sanitize_json_string(text) -> str                    │
│ - _fix_triple_quotes(text) -> str                       │
│ - _execute_tool(tool_call) -> ToolResult                │
└─────────────────────────────────────────────────────────┘
                          │
                          │ uses
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      LLMConfig                          │
├─────────────────────────────────────────────────────────┤
│ + model: str = "llama3.2"                               │
│ + temperature: float = 0.7                              │
│ + top_p: float = 0.9                                    │
│ + max_tokens: int = 2048                                │
│ + base_url: str = "http://localhost:11434"              │
│ + timeout: int = 120                                    │
│ + repeat_penalty: float = 1.1                           │
│ + num_ctx: int = 4096                                   │
└─────────────────────────────────────────────────────────┘
                          │
                          │ creates
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      ToolCall                           │
├─────────────────────────────────────────────────────────┤
│ + name: str                                             │
│ + arguments: Dict                                       │
├─────────────────────────────────────────────────────────┤
│ + action: str (property)                                │
│ + params: Dict (property)                               │
│ - _infer_action() -> Optional[str]                      │
├─────────────────────────────────────────────────────────┤
│ ACTION_INFERENCE_RULES: Dict  # action 자동 추론 규칙   │
└─────────────────────────────────────────────────────────┘
```

## LLM 응답 형식 (Function Calling)

### Tool 호출 시

```json
{
    "thought": "Why I'm taking this action",
    "tool_calls": [
        {
            "name": "file_tool",
            "arguments": {
                "action": "read",
                "path": "wifi.log"
            }
        }
    ]
}
```

### 최종 답변 시

```json
{
    "thought": "I have enough information to answer",
    "tool_calls": null,
    "content": "분석 결과: DMA timeout이 주요 원인입니다..."
}
```

## JSON 파싱 강화

LLM이 잘못된 JSON을 보낼 때 자동 복구합니다.

### 1. 삼중 따옴표 처리

```python
# LLM이 이렇게 보내면 (잘못된 JSON)
"content": """
def fibonacci(n):
    return n
"""

# 자동으로 변환
"content": "\ndef fibonacci(n):\n    return n\n"
```

### 2. 제어 문자 이스케이프

```python
# 문자열 내 줄바꿈이 있으면
"content": "line1
line2"

# 자동으로 변환
"content": "line1\nline2"
```

### 3. 괄호 매칭 JSON 추출

```python
# 텍스트 앞뒤에 불필요한 내용이 있어도
Here's my response:
{"thought": "...", "tool_calls": [...]}

# 정확히 JSON만 추출
```

### 파싱 순서

```
raw_response
    ↓
_fix_triple_quotes()    # """ → " 변환
    ↓
_sanitize_json_string() # 제어 문자 이스케이프  
    ↓
_extract_json()         # 괄호 매칭으로 JSON 추출
    ↓
json.loads()            # 파싱
```

## Action 자동 추론

LLM이 action을 빠뜨렸을 때 arguments 기반으로 자동 추론합니다.

### 추론 규칙

| Tool | Arguments 키 | 추론된 Action |
|------|-------------|---------------|
| **file_tool** | `content` | write |
| | `path` (only) | read |
| | `pattern` | list_dir |
| **web_tool** | `url` | fetch |
| | `keyword` / `query` | search |
| **llm_tool** | `prompt` | ask |
| | `content` | summarize |
| | `text` | analyze |

### 예시

```python
# LLM이 action 없이 보내면
{"name": "llm_tool", "arguments": {"prompt": "Explain this"}}

# 자동 추론: action = "ask" (prompt 키 기반)
```

### tool_name.action 형식 지원

```python
# 이렇게 보내도 처리됨
{"name": "file_tool.read", "arguments": {"path": "test.txt"}}

# 자동 분리: name="file_tool", action="read"
```

## LLM 동적 설정

사용자가 UI에서 실시간 변경 가능:

```python
@dataclass
class LLMConfig:
    # 모델 선택
    model: str = "llama3.2"           # Ollama 서버에서 조회
    
    # 생성 파라미터  
    temperature: float = 0.7          # 창의성 (0.0 ~ 2.0)
    top_p: float = 0.9                # nucleus sampling
    max_tokens: int = 2048            # 최대 출력 길이
    
    # 연결 설정
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    
    # 고급 옵션
    repeat_penalty: float = 1.1       # 반복 억제
    num_ctx: int = 4096               # context window 크기
```

### 설정 시나리오

```python
# 코드 분석 - 정확성 중시
config.model = "codellama"
config.temperature = 0.2

# 창의적 요약 - 창의성 높임
config.model = "llama3.2"
config.temperature = 0.8

# 긴 로그 처리 - context 확장
config.num_ctx = 8192
config.max_tokens = 4096
```

## Streaming 출력

### Generator 방식 (Gradio 연동)

```python
def run_stream(user_query: str) -> Generator[StepInfo, None, None]:
    """각 step 결과를 yield"""
    while not complete:
        # Thinking
        yield StepInfo(type=StepType.THINKING, content="분석 중...")
        
        # Tool 호출
        yield StepInfo(type=StepType.TOOL_CALL, tool_name="file_tool", action="read")
        
        # Tool 결과
        yield StepInfo(type=StepType.TOOL_RESULT, content="파일 내용...")
    
    # 최종 답변
    yield StepInfo(type=StepType.FINAL_ANSWER, content="분석 결과입니다...")
```

### StepType 종류

| Type | 설명 |
|------|------|
| THINKING | LLM 사고 과정 |
| TOOL_CALL | Tool 호출 시작 |
| TOOL_RESULT | Tool 실행 결과 |
| FINAL_ANSWER | 최종 답변 |
| ERROR | 오류 발생 |

## 종료 조건

```python
def is_complete(self) -> bool:
    # 1. LLM이 final_answer 반환
    if last_response.tool_calls is None:
        return True
    
    # 2. 최대 step 도달
    if current_step >= self.max_steps:
        return True
    
    # 3. 수동 중지
    if self.stopped:
        return True
    
    return False
```

## System Prompt 설계

```
You are an AI assistant that helps users by using available tools.

## Available Tools
{tools_schema}

## Response Format
You must respond in valid JSON format only. No other text before or after the JSON.

1. To use a tool:
{
    "thought": "Why I'm taking this action",
    "tool_calls": [
        {
            "name": "tool_name",
            "arguments": {
                "action": "action_name",
                "param1": "value1"
            }
        }
    ]
}

2. To give final answer (when you have enough information):
{
    "thought": "I have enough information to answer",
    "tool_calls": null,
    "content": "Your final answer here"
}

## Rules
- Always respond with valid JSON only
- Use tools when you need external information
- Give final answer when you have enough information
- Be concise and accurate
- One tool call at a time is recommended for clarity

## IMPORTANT: JSON String Format
- NEVER use triple quotes (""") in JSON - they are invalid
- Use \n for newlines inside strings
- Escape double quotes as \"
- Example for code content: "content": "def hello():\n    print(\"Hello\")"
```

## 파일 위치

```
multi_agent_chatbot/
├── core/
│   ├── shared_storage.py
│   ├── base_tool.py
│   ├── orchestrator.py
│   ├── workspace_manager.py
│   └── html_utils.py
├── tools/
│   ├── file_tool.py
│   ├── web_tool.py
│   └── llm_tool.py
├── docs/
│   ├── shared_storage.md
│   ├── base_tool.md
│   ├── orchestrator.md
│   └── ui.md
└── tests/
    ├── test_shared_storage.py
    ├── test_tools.py
    ├── test_orchestrator.py
    └── test_workspace.py
```

## 사용 예시

```python
from core.orchestrator import Orchestrator, LLMConfig
from core.shared_storage import SharedStorage
from core.base_tool import ToolRegistry
from tools import FileTool, WebTool, LLMTool

# 1. 설정
storage = SharedStorage()
registry = ToolRegistry()
registry.register(FileTool())
registry.register(WebTool())
registry.register(LLMTool())

# 2. Orchestrator 생성
orchestrator = Orchestrator(
    tools=registry,
    storage=storage,
    max_steps=10
)

# 3. LLM 설정 변경 (선택)
orchestrator.update_llm_config(
    model="codellama",
    temperature=0.3
)

# 4. 실행 (Streaming)
for step_info in orchestrator.run_stream("wifi.log 분석해줘"):
    if step_info.type == StepType.THINKING:
        print(f"💭 {step_info.content}")
    elif step_info.type == StepType.TOOL_CALL:
        print(f"🔧 {step_info.tool_name}.{step_info.action}")
    elif step_info.type == StepType.TOOL_RESULT:
        print(f"📄 {step_info.content[:100]}...")
    elif step_info.type == StepType.FINAL_ANSWER:
        print(f"✅ {step_info.content}")
```