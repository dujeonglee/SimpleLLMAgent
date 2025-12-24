# SharedStorage 설계 문서

## 개요

SharedStorage는 Multi-Agent Chatbot의 **중앙 데이터 저장소**입니다.
Agent/Tool 간 데이터 공유 및 실행 결과 관리를 담당합니다.

## 핵심 원칙

```
Agent/Tool 간 데이터 전달은 오직 results[N].output을 통해서만!
```

```
results[N-1].output  →  results[N].input  →  results[N].output
```

## 데이터 구조

### 전체 구조

```
SharedStorage
├── context      # 현재 작업 컨텍스트
├── results      # step별 실행 결과 리스트  
├── history      # 완료된 세션들의 히스토리
├── save_to_file()
├── load_from_file()
└── get_summary()  # LLM에 전달할 요약
```

### 1. Context (현재 작업 상태)

```python
context = {
    "user_query": "wifi.log 파일 읽어서 분석해줘",
    "current_plan": ["파일 읽기", "분석하기", "결과 정리"],
    "current_step": 1,
    "session_id": "abc12345",
    "created_at": "2025-01-15T10:30:00"
}
```

- 한 세션(하나의 user query 처리) 동안 유지
- 세션 완료 시 초기화됨

### 2. Results (실행 결과 누적)

```python
results = [
    {
        "step": 1,
        "executor": "file_tool",
        "executor_type": "tool",      # "tool" 또는 "agent"
        "action": "read",
        "input": {"path": "/var/log/wifi.log"},
        "output": "로그 내용...",       # ← 다음 step이 사용
        "status": "success",
        "error_message": None,
        "timestamp": "2025-01-15T10:30:01"
    },
    {
        "step": 2,
        "executor": "llm_tool",
        "executor_type": "tool",
        "action": "analyze",
        "input": {"content": "로그 내용..."},  # ← 이전 output 참조
        "output": "분석 결과...",
        "status": "success",
        ...
    }
]
```

- Tool과 Agent 결과를 같이 저장 (`executor_type`으로 구분)
- 순서대로 누적되어 추적 가능

### 3. History (완료된 세션 보관)

```python
history = [
    {
        "session_id": "abc12345",
        "user_query": "wifi.log 파일 읽어서 분석해줘",
        "results": [...],
        "final_response": "분석 결과입니다...",
        "status": "completed",
        "created_at": "2025-01-15T10:30:00",
        "completed_at": "2025-01-15T10:35:00"
    }
]
```

**용도:**
- 디버깅: "어제 그 쿼리 왜 실패했지?" 추적
- 재실행: 같은 쿼리 다시 실행 시 참고
- 파일 저장/로드로 세션 간 지속성 확보

## 데이터 흐름

```
User Query
    ↓
┌─────────────────────────────────────────────┐
│  SharedStorage                              │
│                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   │
│  │ context │ → │ results │ → │ history │   │
│  │ (현재)  │   │ (누적)  │   │ (완료후)│   │
│  └─────────┘   └─────────┘   └─────────┘   │
│        ↓                                    │
│   get_summary() → LLM 전달                  │
└─────────────────────────────────────────────┘
```

## 주요 메서드

### 세션 관리

| 메서드 | 설명 |
|--------|------|
| `start_session(user_query, plan)` | 새 세션 시작, session_id 반환 |
| `update_plan(plan)` | 실행 계획 업데이트 |
| `advance_step()` | 다음 step으로 이동 |
| `complete_session(final_response)` | 세션 완료 → History로 이동 |

### 결과 관리

| 메서드 | 설명 |
|--------|------|
| `add_result(...)` | Tool/Agent 실행 결과 저장 |
| `get_results()` | 모든 실행 결과 반환 |
| `get_last_output()` | 마지막 실행의 output만 반환 |
| `get_output_by_step(step)` | 특정 step의 output 반환 |

### Summary (LLM용)

| 메서드 | 설명 |
|--------|------|
| `get_summary()` | 현재 상태 요약 (모든 결과 포함, truncate 적용) |

**설계 결정**: 모든 이전 결과를 요약에 포함하고, LLM이 필요한 것만 선택해서 사용하도록 함.

### 영속성

| 메서드 | 설명 |
|--------|------|
| `save_to_file(filepath)` | 현재 상태를 JSON 파일로 저장 |
| `load_from_file(filepath)` | JSON 파일에서 상태 복원 |

## 사용 예시

```python
from core.shared_storage import SharedStorage

# 1. 초기화
storage = SharedStorage(debug_enabled=True)

# 2. 세션 시작
session_id = storage.start_session(
    user_query="wifi.log 분석해줘",
    plan=["파일 읽기", "분석하기"]
)

# 3. 결과 추가
storage.add_result(
    executor="file_tool",
    executor_type="tool",
    action="read",
    input_data={"path": "wifi.log"},
    output="[ERROR] DMA timeout..."
)
storage.advance_step()

# 4. 이전 결과 사용
previous = storage.get_last_output()  # → "[ERROR] DMA timeout..."

storage.add_result(
    executor="llm_tool",
    executor_type="tool", 
    action="analyze",
    input_data={"content": previous},
    output="DMA timeout 에러 발견..."
)

# 5. LLM에 전달할 요약
summary = storage.get_summary()

# 6. 세션 완료
storage.complete_session("분석 완료: DMA timeout 에러...")

# 7. 파일 저장
storage.save_to_file("./data/session_backup.json")
```

## 파일 위치

```
multi_agent_chatbot/
├── core/
│   ├── __init__.py
│   └── shared_storage.py    ← 구현
└── tests/
    └── test_shared_storage.py
```
