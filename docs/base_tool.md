# BaseTool 설계 문서

## 개요

BaseTool은 모든 Tool의 **공통 인터페이스**를 정의하는 추상 클래스입니다.
file_tool, web_tool, llm_tool 등이 이를 상속받아 구현합니다.

## 핵심 원칙

```
Tool = 순수 실행기
- Input 받음 → 실행 → Output 반환
- 다른 Tool 모름 (의존성 없음)
- SharedStorage 직접 접근 안 함
```

**Tool 간 의존성은 Orchestrator가 관리합니다.**

```
Orchestrator
    ├─→ Step 1: WebTool.fetch() → output: "원본 텍스트"
    └─→ Step 2: LLMTool.summarize(input: Step 1 output) → output: "요약"
```

## 클래스 다이어그램

```
┌─────────────────────────────────────────────────────────┐
│                      BaseTool                           │
│  - name: str                                            │
│  - description: str                                     │
│  - actions: Dict[str, ActionSchema]                     │
│                                                         │
│  + execute(action, params) → ToolResult                 │
│  + get_schema() → Dict  (LLM에게 tool 설명용)           │
│  + validate_params(action, params) → bool               │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ 상속
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
   │FileTool │      │WebTool  │      │LLMTool  │
   │         │      │         │      │         │
   │-read    │      │-search  │      │-ask     │
   │-write   │      │-fetch   │      │-analyze │
   │-update  │      │-summary │      │         │
   │-delete  │      │         │      │         │
   └─────────┘      └─────────┘      └─────────┘
```

## 데이터 구조

### 1. ToolResult (실행 결과)

```python
@dataclass
class ToolResult:
    success: bool            # 성공 여부
    output: Any              # 다음 step에서 사용할 데이터
    error: Optional[str]     # 에러 메시지 (실패 시)
    metadata: Dict           # 부가 정보 (실행 시간 등)
```

**SharedStorage.add_result()와 매핑:**
```python
storage.add_result(
    executor=tool.name,
    executor_type="tool",
    action=action,
    input_data=params,
    output=result.output,
    status="success" if result.success else "error",
    error_message=result.error
)
```

### 2. ActionParam (파라미터 정의)

```python
@dataclass
class ActionParam:
    name: str                # 파라미터 이름
    param_type: str          # "str", "int", "bool", "dict", "list"
    required: bool           # 필수 여부
    description: str         # 설명
    default: Any = None      # 기본값 (optional일 때)
```

### 3. ActionSchema (액션 정의)

```python
@dataclass
class ActionSchema:
    name: str                    # 액션 이름
    description: str             # 액션 설명
    params: List[ActionParam]    # 파라미터 목록
    output_type: str             # "str", "dict", "list", "None"
    output_description: str      # Output 설명 (한 줄)
```

**Output 정의 설계 결정:**
- `output_type`: 타입만 간단히 명시
- `output_description`: 자연어로 한 줄 설명
- 상세 스키마는 정의하지 않음 (LLM 환각 방지)

```python
# ✓ 좋은 예
ActionSchema(
    name="analyze",
    output_type="dict",
    output_description="분석 결과 (에러 목록, 주요 이슈, 검색 키워드 포함)"
)

# ❌ 피해야 할 예 (너무 상세)
ActionSchema(
    output_schema={
        "errors_found": "int",
        "main_issue": "str",
        ...
    }
)
```

## BaseTool 클래스

### 추상 메서드 (자식 클래스 필수 구현)

```python
@abstractmethod
def _define_actions(self) -> Dict[str, ActionSchema]:
    """Tool이 지원하는 action들 정의"""
    pass

@abstractmethod
def _execute_action(self, action: str, params: Dict) -> ToolResult:
    """실제 action 실행 로직"""
    pass
```

### 공통 메서드 (BaseTool 제공)

```python
def execute(self, action: str, params: Dict) -> ToolResult:
    """
    공통 실행 로직
    1. action 존재 확인
    2. params 검증
    3. _execute_action 호출
    4. 결과 로깅
    """

def get_schema(self) -> Dict:
    """LLM에게 전달할 tool 스키마 반환"""

def validate_params(self, action: str, params: Dict) -> Tuple[bool, str]:
    """파라미터 유효성 검증"""

def get_action_names(self) -> List[str]:
    """지원하는 action 이름 목록"""
```

## get_schema() 출력 형식

LLM이 Tool을 이해하고 선택할 수 있도록 구조화된 정보 제공:

```json
{
  "name": "file_tool",
  "description": "파일 읽기, 쓰기, 수정, 삭제",
  "actions": {
    "read": {
      "description": "파일 내용 읽기",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "파일 경로"
          }
        },
        "required": ["path"]
      },
      "output": {
        "type": "string",
        "description": "파일 내용"
      }
    },
    "write": {
      "description": "파일에 내용 쓰기",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "파일 경로"
          },
          "content": {
            "type": "string",
            "description": "작성할 내용"
          },
          "overwrite": {
            "type": "boolean",
            "description": "덮어쓰기 여부",
            "default": false
          }
        },
        "required": ["path", "content"]
      },
      "output": {
        "type": "string",
        "description": "작성된 파일 경로"
      }
    }
  }
}
```

## 구체적 Tool 예시

### FileTool

```python
class FileTool(BaseTool):
    name = "file_tool"
    description = "파일 읽기, 쓰기, 수정, 삭제"
    
    def _define_actions(self):
        return {
            "read": ActionSchema(
                name="read",
                description="파일 내용 읽기",
                params=[
                    ActionParam("path", "str", True, "파일 경로")
                ],
                output_type="str",
                output_description="파일 내용"
            ),
            "write": ActionSchema(
                name="write",
                description="파일에 내용 쓰기",
                params=[
                    ActionParam("path", "str", True, "파일 경로"),
                    ActionParam("content", "str", True, "작성할 내용"),
                    ActionParam("overwrite", "bool", False, "덮어쓰기 여부", False)
                ],
                output_type="str",
                output_description="작성된 파일 경로"
            ),
            "delete": ActionSchema(
                name="delete",
                description="파일 삭제",
                params=[
                    ActionParam("path", "str", True, "파일 경로")
                ],
                output_type="bool",
                output_description="삭제 성공 여부"
            )
        }
```

### WebTool

```python
class WebTool(BaseTool):
    name = "web_tool"
    description = "웹 검색, 페이지 가져오기, 요약"
    
    def _define_actions(self):
        return {
            "search": ActionSchema(
                name="search",
                description="키워드로 웹 검색",
                params=[
                    ActionParam("keyword", "str", True, "검색 키워드"),
                    ActionParam("max_results", "int", False, "최대 결과 수", 5)
                ],
                output_type="list",
                output_description="검색 결과 목록 (title, url, snippet)"
            ),
            "fetch": ActionSchema(
                name="fetch",
                description="URL에서 페이지 내용 가져오기",
                params=[
                    ActionParam("url", "str", True, "가져올 URL")
                ],
                output_type="str",
                output_description="페이지 텍스트 내용"
            )
        }
```

### LLMTool

```python
class LLMTool(BaseTool):
    name = "llm_tool"
    description = "LLM에 질문하고 응답받기"
    
    def _define_actions(self):
        return {
            "ask": ActionSchema(
                name="ask",
                description="LLM에 자유 질문",
                params=[
                    ActionParam("prompt", "str", True, "질문 내용")
                ],
                output_type="str",
                output_description="LLM 응답"
            ),
            "analyze": ActionSchema(
                name="analyze",
                description="텍스트 분석 요청",
                params=[
                    ActionParam("content", "str", True, "분석할 내용"),
                    ActionParam("instruction", "str", False, "분석 지시사항", "내용을 분석해주세요")
                ],
                output_type="dict",
                output_description="분석 결과 (구조는 내용에 따라 다름)"
            )
        }
```

## Orchestrator 연동

```python
# 1. Orchestrator가 available tools 스키마 수집
tools = [file_tool, web_tool, llm_tool]
tools_schema = [tool.get_schema() for tool in tools]

# 2. LLM에게 전달하여 다음 action 결정
next_action = llm.decide_next_action(
    user_query=user_query,
    available_tools=tools_schema,
    previous_results=storage.get_summary()
)
# → {"tool": "file_tool", "action": "read", "params": {"path": "wifi.log"}}

# 3. Tool 실행
tool = get_tool_by_name(next_action["tool"])
result = tool.execute(next_action["action"], next_action["params"])

# 4. SharedStorage에 저장
storage.add_result(
    executor=tool.name,
    executor_type="tool",
    action=next_action["action"],
    input_data=next_action["params"],
    output=result.output,
    status="success" if result.success else "error",
    error_message=result.error
)
```

## 파일 위치

```
multi_agent_chatbot/
├── core/
│   ├── __init__.py
│   ├── shared_storage.py
│   └── base_tool.py         ← BaseTool 구현
├── tools/
│   ├── __init__.py
│   ├── file_tool.py         ← FileTool 구현
│   ├── web_tool.py          ← WebTool 구현
│   └── llm_tool.py          ← LLMTool 구현
└── tests/
    ├── test_shared_storage.py
    └── test_tools.py
```

## 테스트 전략

각 Tool은 독립적으로 테스트 가능:

```python
def test_file_tool_read():
    tool = FileTool()
    result = tool.execute("read", {"path": "test.txt"})
    assert result.success
    assert isinstance(result.output, str)

def test_file_tool_invalid_action():
    tool = FileTool()
    result = tool.execute("invalid_action", {})
    assert not result.success
    assert "not found" in result.error

def test_file_tool_missing_param():
    tool = FileTool()
    result = tool.execute("read", {})  # path 누락
    assert not result.success
    assert "required" in result.error
```
