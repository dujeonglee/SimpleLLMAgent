"""
Orchestrator Module (Updated)
=============================
LLMTool 연동 및 이전 결과 참조 기능 추가.

주요 변경사항:
1. LLMTool 초기화 시 llm_caller와 previous_result_getter 주입
2. [PREVIOUS_RESULT] 플레이스홀더 자동 대체
3. 이전 결과를 활용하는 프롬프트 개선
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

from core.shared_storage import SharedStorage
from core.debug_logger import DebugLogger
from core.base_tool import ToolRegistry, ToolResult, ActionSchema
from core.json_parser import parse_json_strict
from core.execution_events import (
    ExecutionEvent,
    PlanPromptEvent,
    PlanReadyEvent,
    ThinkingEvent,
    StepPromptEvent,
    ToolCallEvent,
    ToolResultEvent,
    FinalAnswerEvent,
    ErrorEvent
)


# =============================================================================
# Data Classes (변경 없음)
# =============================================================================

@dataclass
class LLMConfig:
    """LLM 동적 설정"""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2048
    num_ctx: int = 4096
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_steps: int = 10

    def to_dict(self) -> Dict:
        return asdict(self)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class ToolCall:
    """LLM이 요청한 Tool 호출 정보"""
    name: str
    arguments: Dict
    known_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if "." in self.name:
            parts = self.name.split(".", 1)
            self.name = parts[0]
            if "action" not in self.arguments:
                self.arguments["action"] = parts[1]

        if not self.arguments.get("action"):
            inferred_action = self._infer_action_from_prefix()
            if inferred_action:
                self.arguments["action"] = inferred_action

    def _infer_action_from_prefix(self) -> Optional[str]:
        for key in self.arguments.keys():
            if "_" in key:
                prefix = key.split("_")[0]
                if prefix in self.known_actions:
                    return prefix
        return None
    
    @property
    def action(self) -> str:
        return self.arguments.get("action", "")
    
    @property
    def params(self) -> Dict:
        return {k: v for k, v in self.arguments.items() if k != "action"}


@dataclass
class LLMResponse:
    """LLM 응답 파싱 결과"""
    thought: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    content: Optional[str] = None
    raw_response: str = ""
    
    @property
    def is_final_answer(self) -> bool:
        return self.tool_calls is None and self.content is not None


class StepType(Enum):
    PLANNING = "planning"
    PLAN_PROMPT = "plan_prompt"
    PLAN_READY = "plan_ready"
    THINKING = "thinking"
    STEP_PROMPT = "step_prompt"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class PlannedStep:
    """계획된 단계"""
    step: int
    tool_name: str
    action: str
    description: str
    params_hint: Dict = field(default_factory=dict)
    status: str = "pending"

@dataclass
class StepInfo:
    """Streaming 출력용 Step 정보"""
    type: StepType
    step: int
    content: Any
    tool_name: Optional[str] = None
    action: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "step": self.step,
            "content": self.content,
            "tool_name": self.tool_name,
            "action": self.action,
            "timestamp": self.timestamp
        }


# =============================================================================
# Orchestrator (Updated)
# =============================================================================

class Orchestrator:
    """
    Multi-Agent Chatbot Orchestrator
    
    변경사항:
    - LLMTool 자동 연동
    - [PREVIOUS_RESULT] 플레이스홀더 처리
    - 이전 결과 참조 프롬프트 개선
    """
    
    def __init__(
        self,
        tools: ToolRegistry,
        storage: SharedStorage,
        llm_config: LLMConfig = None,
        max_steps: int = 10,
        on_step_complete: Callable[[ExecutionEvent], None] = None,
        debug_enabled: bool = True
    ):
        self.tools = tools
        self.storage = storage
        self.llm_config = llm_config or LLMConfig()
        self.max_steps = max_steps
        self.on_step_complete = on_step_complete
        self.logger = DebugLogger("Orchestrator", enabled=debug_enabled)

        self._stopped = False
        self._current_step = 0
        self._plan: List[PlannedStep] = []
        self._needs_tools: bool = True  # 기본값: 툴 필요
        
        # LLMTool 연동
        self._setup_llm_tool()
        
        self.logger.info("Orchestrator 초기화", {
            "max_steps": max_steps,
            "llm_model": self.llm_config.model,
            "tools": self.tools.list_tools()
        })
    
    def _setup_llm_tool(self):
        """LLMTool에 필요한 함수 주입"""
        llm_tool = self.tools.get("llm_tool")
        if llm_tool is None:
            self.logger.debug("llm_tool이 등록되지 않음")
            return

        # LLMTool 타입 체크 (duck typing)
        if hasattr(llm_tool, 'set_llm_caller'):
            llm_tool.set_llm_caller(self._call_llm_api)
            self.logger.info("LLMTool에 llm_caller 주입 완료")

    def _get_known_actions(self) -> Dict[str, List[str]]:
        """등록된 tool들의 action 정보를 동적으로 가져오기"""
        known_actions = {}
        for tool in self.tools.get_all():
            known_actions[tool.name] = tool.get_action_names()
        self.logger.debug(f"Known actions 생성 완료: {known_actions}")
        return known_actions
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def update_llm_config(self, **kwargs):
        """LLM 설정 동적 변경"""
        old_values = {}
        for key in kwargs:
            if hasattr(self.llm_config, key):
                old_values[key] = getattr(self.llm_config, key)
        
        self.llm_config.update(**kwargs)
        
        self.logger.info("LLM 설정 변경", {
            "changes": {k: {"old": old_values.get(k), "new": v} 
                       for k, v in kwargs.items() if k in old_values}
        })
    
    def stop(self):
        """실행 중지"""
        self._stopped = True
        self.logger.info("실행 중지 요청됨")

    def _initialize_session(self, user_query: str, files_context: str = ""):
        """세션 초기화"""
        self._stopped = False
        self._current_step = 0
        self._plan = []
        self._needs_tools = True
        self._files_context = files_context
        self.storage.start_session(user_query)
        self.logger.info(f"실행 시작: {user_query[:50]}...")

    def run_stream(self, user_query: str, files_context: str = "") -> Generator[ExecutionEvent, None, None]:
        """Streaming 실행 - Plan & Execute 패턴"""
        self._initialize_session(user_query, files_context)

        try:
            # Phase 1: Planning
            for event in self._create_plan(user_query):
                yield event

                # Plan이 direct answer나 error로 끝나면 종료
                if isinstance(event, (FinalAnswerEvent, ErrorEvent)):
                    return

            # Phase 2: Execution
            for event in self._execute_plan(user_query):
                yield event

                # Execution이 error로 끝나면 종료
                if isinstance(event, ErrorEvent):
                    return

            # Phase 3: Final Answer
            for event in self._generate_final_answer_stream(user_query):
                yield event

        except Exception as e:
            error_msg = f"실행 중 오류 발생: {str(e)}"
            self.logger.error(error_msg, {"exception": type(e).__name__})
            self.storage.complete_session(final_response=error_msg, status="error")
            yield ErrorEvent(error_message=error_msg)
    
    # =========================================================================
    # Tool Execution (Updated)
    # =========================================================================

    def _execute_tool_with_retry(
        self,
        tool_call: ToolCall,
        planned_step: PlannedStep,
        max_retries: int = 2
    ) -> ToolResult:
        """
        Tool 실행 및 실패 시 자동 재시도

        Args:
            tool_call: 실행할 tool call
            planned_step: 계획된 step 정보
            max_retries: 최대 재시도 횟수 (기본 2회)

        Returns:
            ToolResult: 최종 실행 결과 (성공 또는 최종 실패)
                       metadata에 재시도 정보 포함:
                       - retry_count: 재시도 횟수
                       - retry_history: 각 시도의 에러 기록
                       - corrected_params: 수정된 파라미터 (재시도 성공 시)
        """
        retry_history = []

        # 첫 시도
        result = self._execute_tool(tool_call)

        if result.success:
            # 첫 시도 성공 - 재시도 정보 없음
            result.metadata["retry_count"] = 0
            return result

        # 첫 시도 실패 - 재시도 시작
        retry_history.append({
            "attempt": 0,
            "error": result.error,
            "params": tool_call.params.copy()
        })

        # 재시도 루프
        for retry_count in range(1, max_retries + 1):
            self.logger.info(f"재시도 {retry_count}/{max_retries}: {tool_call.name}.{tool_call.action}", {
                "previous_error": result.error
            })

            # LLM에 에러 전달하여 수정된 파라미터 요청
            corrected_call = self._request_parameter_correction(
                original_call=tool_call,
                error_message=result.error,
                planned_step=planned_step,
                retry_count=retry_count
            )

            if not corrected_call:
                # LLM이 수정 불가능하다고 판단
                self.logger.warn(f"재시도 중단: LLM이 파라미터 수정 불가 판단")
                break

            # 수정된 파라미터로 재시도
            result = self._execute_tool(corrected_call)

            retry_history.append({
                "attempt": retry_count,
                "error": result.error if not result.success else None,
                "params": corrected_call.params.copy(),
                "success": result.success
            })

            if result.success:
                # 재시도 성공!
                self.logger.info(f"재시도 성공: {retry_count}번째 시도에서 성공")
                result.metadata["retry_count"] = retry_count
                result.metadata["retry_history"] = retry_history
                result.metadata["corrected_params"] = corrected_call.params
                return result

        # 모든 재시도 실패
        result.metadata["retry_count"] = len(retry_history) - 1
        result.metadata["retry_history"] = retry_history
        result.metadata["retry_exhausted"] = True

        self.logger.error(f"재시도 모두 실패: {max_retries}회 시도 후 실패")

        return result

    def _request_parameter_correction(
        self,
        original_call: ToolCall,
        error_message: str,
        planned_step: PlannedStep,
        retry_count: int
    ) -> Optional[ToolCall]:
        """
        LLM에 에러 정보를 전달하여 수정된 파라미터 요청

        Args:
            original_call: 원본 tool call
            error_message: 발생한 에러 메시지
            planned_step: 계획된 step
            retry_count: 현재 재시도 횟수

        Returns:
            Optional[ToolCall]: 수정된 tool call (수정 불가능하면 None)
        """
        # 현재 tool 가져오기
        current_tool = self.tools.get(original_call.name)
        if not current_tool:
            return None

        # 현재 action의 schema만 가져오기
        action_schema = current_tool.get_action_schema(original_call.action)
        if not action_schema:
            return None

        # action schema를 JSON 형식으로 변환
        action_schema_dict = action_schema.to_dict()
        action_schema_json = json.dumps(
            {
                "action": original_call.action,
                **action_schema_dict
            },
            indent=2,
            ensure_ascii=False
        )

        # 사용 가능한 결과 목록
        available_results = self.storage.get_available_results_summary()

        system_prompt = f"""You are analyzing a tool execution error and providing corrected parameters.

## Current Step Context
Step {planned_step.step}: {planned_step.description}

## Current Action Schema
Tool: {original_call.name}
Action: {original_call.action}

{action_schema_json}

## Previous Execution Attempt
Parameters Used:
{json.dumps(original_call.params, indent=2, ensure_ascii=False)}

Error Occurred:
{error_message}

## Available Previous Results (for reference)
{available_results}

## Your Task
1. Analyze why the execution failed based on the error message
2. Determine if the error can be fixed by modifying parameters
3. If fixable: provide corrected parameters
4. If NOT fixable (e.g., file doesn't exist, permission denied): return empty tool_calls

## Response Format (JSON only)

If error is fixable with parameter changes:
{{
    "thought": "Analysis: why it failed and how to fix it",
    "tool_calls": [{{
        "name": "{original_call.name}",
        "arguments": {{
            "action": "{original_call.action}",
            ... corrected parameters ...
        }}
    }}]
}}

If error CANNOT be fixed by changing parameters:
{{
    "thought": "Analysis: why this error cannot be fixed with parameter changes",
    "tool_calls": []
}}

## Rules
- You MUST respond with valid JSON only
- Analyze the error carefully before deciding
- Common fixable errors: wrong path format, missing required params, invalid parameter values
- Common unfixable errors: file not found, permission denied, network unreachable
- Use [RESULT:session_id_step] to reference previous results if helpful (e.g., [RESULT:Session0_1])"""

        user_prompt = f"""## Retry Attempt {retry_count}

The tool execution failed. Analyze the error and provide corrected parameters if possible.

Respond with JSON only."""

        try:
            raw_response = self._call_llm_api(system_prompt, user_prompt)
            parsed = self._parse_tool_call_response(raw_response)

            if not parsed.tool_calls:
                self.logger.info(f"LLM 판단: 파라미터 수정으로 해결 불가능", {
                    "step": planned_step.step,
                    "description": planned_step.description,
                    "thought": parsed.thought
                })
                return None

            self.logger.info(f"LLM이 파라미터 수정 제안", {
                "step": planned_step.step,
                "description": planned_step.description,
                "thought": parsed.thought,
                "corrected_params": parsed.tool_calls[0].params
            })

            return parsed.tool_calls[0]

        except Exception as e:
            self.logger.error(f"파라미터 수정 요청 실패: {str(e)}", {
                "step": planned_step.step,
                "description": planned_step.description
            })
            return None

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Tool 실행 - REF 자동 치환

        지원하는 참조 형식:
        - [RESULT:session_id_step]: 특정 session_id_step의 output으로 치환 (예: [RESULT:Session0_1])

        Note: 이 메서드는 재시도 로직이 없는 단순 실행만 수행.
              재시도가 필요한 경우 _execute_tool_with_retry() 사용.
        """
        # 파라미터 복사 및 REF 치환
        params = tool_call.params.copy()
        params = self._substitute_all_refs(params)

        self.logger.info(f"Tool 실행: {tool_call.name}.{tool_call.action}", {
            "params_keys": list(params.keys())
        })

        result = self.tools.execute(
            tool_name=tool_call.name,
            action=tool_call.action,
            params=params,
            session_id=self.storage.get_current_context().session_id,
            step=self._current_step
        )

        return result
    
    # =========================================================================
    # Planning Methods
    # =========================================================================
    
    def _create_plan(self, user_query: str) -> Generator[ExecutionEvent, None, None]:
        """Phase 1: 전체 실행 계획 수립 및 초기 응답 생성"""
        # 1. Planning 시작
        yield ThinkingEvent(message="작업 계획 수립 중...")

        # 2. 중지 체크
        if self._stopped:
            self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
            yield ErrorEvent(error_message="사용자 요청으로 취소됨")
            return

        # 3. Query classification - 툴 필요 여부 판단
        self._needs_tools = self._classify_query(user_query)
        self.logger.debug("Query classification 완료", {"needs_tools": self._needs_tools})

        # 4. LLM 호출하여 plan 생성 (또는 direct answer)
        plan_response, plan_system_prompt, plan_user_prompt, plan_raw_response = self._call_plan_llm(user_query)

        # 5. Direct answer 처리
        if plan_response.get("direct_answer"):
            self.storage.complete_session(
                final_response=plan_response["direct_answer"],
                status="completed"
            )
            yield FinalAnswerEvent(answer=plan_response["direct_answer"])
            return

        yield PlanPromptEvent(
            system_prompt=plan_system_prompt,
            user_prompt=plan_user_prompt,
            raw_response=plan_raw_response
        )

        # 5. Plan 파싱
        self._plan = self._parse_plan(plan_response.get("plan", []))

        # 6. Plan 없음 처리
        if not self._plan:
            direct_response = plan_response.get("thought", "요청을 처리할 수 없습니다.")
            self.storage.complete_session(final_response=direct_response, status="completed")
            yield FinalAnswerEvent(answer=direct_response)
            return

        # 7. Plan 준비 완료
        plan_summary = self._format_plan_summary()
        yield PlanReadyEvent(plan_content=plan_summary)

    def _classify_query(self, user_query: str) -> bool:
        """Query가 툴 사용이 필요한지 판단"""
        tools_list = "\n".join([f"- {name}: {tool.get_schema().get('description', 'No description')}"
                                for name, tool in self.tools._tools.items()])

        system_prompt = f"""You are a query classifier. Determine if the user's query requires using tools or can be answered directly.

## Available Tools
{tools_list}

## Your Task
Analyze if the query needs tool usage (file operations, code analysis, web search, etc.) or can be answered with general knowledge.

## Response Format (JSON only)
{{
    "needs_tools": true/false,
    "reasoning": "Brief explanation"
}}"""

        user_prompt = f"""## User Query
{user_query}

Does this query require tool usage? Respond with JSON only."""

        self.logger.debug("Query classification 시작")

        raw_response = self._call_llm_api(system_prompt, user_prompt)

        try:
            parsed = parse_json_strict(raw_response)
            needs_tools = parsed.get("needs_tools", True)  # 기본값: True (안전)
            self.logger.debug("Classification 결과", {
                "needs_tools": needs_tools,
                "reasoning": parsed.get("reasoning", "")
            })
            return needs_tools
        except Exception as e:
            self.logger.warn(f"Classification 파싱 실패, 기본값(True) 사용: {e}")
            return True  # 파싱 실패 시 안전하게 True 반환

    def _call_plan_llm(self, user_query: str) -> tuple[Dict, str, str, str]:
        """LLM에 계획 수립 또는 direct answer 요청

        Returns:
            tuple: (parsed_response, system_prompt, user_prompt, raw_response)
        """
        if not self._needs_tools:
            # 툴이 필요 없는 경우: direct answer 생성
            system_prompt = """You are a helpful AI assistant. Answer the user's question directly and concisely.

## Response Format (JSON only)
{
    "direct_answer": "Your answer here"
}"""

            user_prompt = f"""## User Query
{user_query}

Provide a direct answer. Respond with JSON only."""

        else:
            # 툴이 필요한 경우: 기존 planning 로직
            tools_schema = json.dumps(
                self.tools.get_all_schemas(),
                indent=2,
                ensure_ascii=False
            )

            system_prompt = f"""You are a task planning expert. Analyze the user's request and create an execution plan.

## Available Tools
{tools_schema}

## Your Task
1. Analyze what the user wants
2. Create a step-by-step plan using available tools
3. If plan cannot be created, provide direct answer explaining why

## Reference System
You can use references in parameter values:

**[FILE:path]** - Reference file content directly (RECOMMENDED)
- Automatically replaced with file content from workspace/files/
- Example: {{"content": "[FILE:example.py]"}} → file content is loaded
- **PREFER this over file_tool.read** - it's more efficient and reduces plan steps
- Use when: You need file content in any parameter

**[RESULT:session_step]** - Reference previous step output
- Example: [RESULT:Session0_1] → output from Step 1 in Session0
- Use when: Reusing output from earlier steps in the current plan

## Plan Optimization
**Minimize steps by using references ([FILE:*], [RESULT:*]) instead of intermediate operations.**

Good (1 step with FILE reference):
{{
    "step": 1,
    "tool_name": "llm_tool",
    "action": "staticanalysis",
    "params_hint": {{"content": "[FILE:example.py]"}}
}}

Bad (2 steps - unnecessary read step):
{{
    "step": 1,
    "tool_name": "file_tool",
    "action": "read",
    "params_hint": {{"read_path": "example.py"}}
}},
{{
    "step": 2,
    "tool_name": "llm_tool",
    "action": "staticanalysis",
    "params_hint": {{"content": "[RESULT:Session0_1]"}}
}}

## Response Format (JSON only)

If tools are needed:
{{
    "thought": "Brief analysis",
    "plan": [
        {{
            "step": 1,
            "tool_name": "tool_name",
            "action": "action_name",
            "description": "What this step does",
            "params_hint": {{
                "param1": "value or [FILE:path] or [RESULT:Session0_X]"
            }}
        }}
    ]
}}

If no tools needed:
{{
    "thought": "This can be answered directly",
    "direct_answer": "Your answer (in Korean/한글)"
}}

## Rules
- Maximum {self.max_steps} steps
- Use references ([FILE:*], [RESULT:*]) to minimize steps
- Each step must have params_hint with concrete values or references
- Respond with valid JSON only"""

            # _needs_tools=True일 때만 files_context 추가
            if self._files_context:
                user_prompt = f"""## User Request
{user_query}

{self._files_context}

Create an execution plan. Respond with JSON only."""
            else:
                user_prompt = f"""## User Request
{user_query}

Create an execution plan. Respond with JSON only."""

        # 공통: LLM 호출 및 파싱
        self.logger.debug("Planning 호출", {
            "query": user_query[:50],
            "needs_tools": self._needs_tools
        })

        raw_response = self._call_llm_api(system_prompt, user_prompt)

        try:
            parsed = parse_json_strict(raw_response)
            self.logger.debug("Plan 생성 완료", {
                "has_plan": "plan" in parsed,
                "has_direct_answer": "direct_answer" in parsed,
                "needs_tools": self._needs_tools
            })
            return parsed, system_prompt, user_prompt, raw_response
        except Exception as e:
            self.logger.error(f"Plan 파싱 실패: {e}")
            return {"direct_answer": f"계획 수립 중 오류가 발생했습니다: {str(e)}"}, system_prompt, user_prompt, raw_response

    # =========================================================================
    # Execution Methods
    # =========================================================================

    def _execute_plan(self, user_query: str) -> Generator[ExecutionEvent, None, None]:
        """Phase 2: 계획된 단계들을 순차적으로 실행"""
        # Precondition: Plan이 존재해야 함
        if not self._plan:
            error_msg = "CRITICAL: _execute_plan called with empty plan"
            self.logger.error(error_msg)
            self.storage.complete_session(final_response=error_msg, status="error")
            yield ErrorEvent(error_message=error_msg)
            return

        for planned_step in self._plan:
            self._current_step = planned_step.step
            planned_step.status = "running"

            yield ThinkingEvent(message=f"Step {self._current_step}: {planned_step.description}")

            # LLM 호출 전 중지 체크
            if self._stopped:
                self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                yield ErrorEvent(error_message="사용자 요청으로 취소됨")
                return

            execution_response, prompt_info = self._get_execution_params(user_query, planned_step)

            # Step 프롬프트 정보 yield (ToolResultEvent에서 사용할 정보)
            step_prompt_event = StepPromptEvent(
                step=self._current_step,
                tool_name=planned_step.tool_name,
                action=planned_step.action,
                system_prompt=prompt_info["system_prompt"],
                user_prompt=prompt_info["user_prompt"],
                raw_response=prompt_info["raw_response"]
            )
            yield step_prompt_event

            # LLM 호출 후 중지 체크
            if self._stopped:
                self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                yield ErrorEvent(error_message="사용자 요청으로 취소됨")
                return

            if execution_response.thought:
                yield ThinkingEvent(message=execution_response.thought)

            # Check if tool_calls count is not exactly 1
            tool_calls_count = len(execution_response.tool_calls) if execution_response.tool_calls else 0
            if tool_calls_count != 1:
                error_msg = f"LLM failed to generate exactly one tool_call for Step {self._current_step} ({planned_step.tool_name}.{planned_step.action}). Expected 1, got {tool_calls_count}."
                self.logger.error(error_msg, {
                    "step": self._current_step,
                    "planned_tool": planned_step.tool_name,
                    "planned_action": planned_step.action,
                    "tool_calls_count": tool_calls_count,
                    "llm_thought": execution_response.thought
                })

                planned_step.status = "failed"

                yield ErrorEvent(
                    error_message=f"⚠️ Step {self._current_step} failed: {error_msg}\n\nLLM response: {execution_response.thought or 'No explanation provided'}"
                )
                # Continue to next step instead of stopping entirely
                continue

            # Execute the single tool call
            tool_call = execution_response.tool_calls[0]

            # Tool 실행 전 중지 체크
            if self._stopped:
                self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                yield ErrorEvent(error_message="사용자 요청으로 취소됨")
                return

            yield ToolCallEvent(
                step=self._current_step,
                tool_name=tool_call.name,
                action=tool_call.action,
                arguments=tool_call.arguments
            )

            # Tool 실행 (자동 재시도 포함)
            result = self._execute_tool_with_retry(tool_call, planned_step, max_retries=2)

            # Tool 실행 후 중지 체크
            if self._stopped:
                self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                yield ErrorEvent(error_message="사용자 요청으로 취소됨")
                return

            # ToolResult를 그대로 전달
            self.storage.add_result(result)

            planned_step.status = "completed" if result.success else "failed"

            yield ToolResultEvent(
                step=self._current_step,
                tool_name=tool_call.name,
                action=tool_call.action,
                result=result.output if result.success else result.error,
                success=result.success
            )

            if self.on_step_complete:
                self.on_step_complete(ToolResultEvent(
                    step=self._current_step,
                    tool_name=tool_call.name,
                    action=tool_call.action,
                    result="Step completed",
                    success=result.success
                ))

        # 중지된 경우 Final Answer 생성 스킵
        if self._stopped:
            self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
            yield ErrorEvent(error_message="사용자 요청으로 취소됨")
            return

    def _generate_final_answer_stream(self, user_query: str) -> Generator[ExecutionEvent, None, None]:
        """Phase 3: 최종 응답 생성"""
        # 1. Thinking 시작
        yield ThinkingEvent(message="최종 응답 생성 중...")

        # 2. 중지 체크
        if self._stopped:
            self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
            yield ErrorEvent(error_message="사용자 요청으로 취소됨")
            return

        # 3. 최종 응답 생성
        final_response = self._generate_final_answer(user_query)

        # 4. 세션 완료
        self.storage.complete_session(final_response=final_response, status="completed")

        # 5. 최종 응답 반환
        yield FinalAnswerEvent(answer=final_response)

    def _get_execution_params(self, user_query: str, planned_step: PlannedStep) -> tuple[LLMResponse, Dict]:
        """
        계획된 step 실행을 위한 정확한 파라미터 결정

        Returns:
            tuple: (LLMResponse, prompt_info dict with system_prompt, user_prompt, raw_response)
        """
        # 현재 step의 tool과 action 가져오기
        current_tool = self.tools.get(planned_step.tool_name)
        if not current_tool:
            raise ValueError(f"Tool not found: {planned_step.tool_name}")

        # 현재 action의 schema 가져오기
        action_schema = current_tool.get_action_schema(planned_step.action)
        if not action_schema:
            raise ValueError(f"Action not found: {planned_step.tool_name}.{planned_step.action}")

        # 특정 action의 스키마만 JSON으로 변환
        action_schema_dict = action_schema.to_dict()
        tool_schema = json.dumps(
            {
                "tool": planned_step.tool_name,
                "action": planned_step.action,
                "schema": action_schema_dict
            },
            indent=2,
            ensure_ascii=False
        )

        # 실행 컨텍스트 가져오기 (user_query + 결과 목록)
        execution_context = self.storage.get_available_results_summary(max_output_preview=300)

        # 현재 action의 schema를 기반으로 response format 생성
        response_format = self._generate_response_format(planned_step, action_schema)

        system_prompt = f"""You are executing a planned task. Provide exact parameters for the current step.

## Current Action Schema
{tool_schema}

## Reference System
You can use these references in parameter values:

**[FILE:path]** - Reference file content directly (RECOMMENDED)
- Example: {{"content": "[FILE:example.py]"}} → file content is loaded automatically
- **PREFER this over file_tool.read** - it's more efficient
- Use for: Loading file content in any parameter

**[RESULT:session_step]** - Reference previous step output
- Example: [RESULT:Session0_1] → output from Step 1 in Session0
- Use for: Reusing results from earlier steps in the current execution

## Response Format for Current Step
{response_format}

## Rules
- Execute ONLY the current step (Step {planned_step.step})
- You MUST generate exactly ONE tool_call for this step
- Use parameter names with action prefix (e.g., read_path, write_content)
- Use references ([FILE:*], [RESULT:*]) when appropriate
- All required parameters must be provided
- Respond with valid JSON only"""

        # params_hint를 포맷팅 (있는 경우)
        params_hint_section = ""
        if planned_step.params_hint:
            params_hint_section = f"""
## Suggested Parameters
{json.dumps(planned_step.params_hint, indent=2, ensure_ascii=False)}

Use these as hints - verify they match the schema and adjust if needed.
"""
        else:
            params_hint_section = ""

        user_prompt = f"""{execution_context}

## Current Step to Execute
Step {planned_step.step}: {planned_step.tool_name}.{planned_step.action}
Description: {planned_step.description}
{params_hint_section}
Provide exact parameters for this step. Respond with JSON only."""

        raw_response = self._call_llm_api(system_prompt, user_prompt)

        # 프롬프트 정보 수집
        prompt_info = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response
        }

        return self._parse_tool_call_response(raw_response), prompt_info
    
    # =========================================================================
    # Helper Methods
    # =========================================================================


    def _generate_response_format(self, planned_step: PlannedStep, action_schema: Optional[ActionSchema]) -> str:
        """
        Generate response format template based on the action schema

        Args:
            planned_step: The planned step to execute
            action_schema: The ActionSchema for the current action

        Returns:
            str: JSON template showing the expected response format
        """
        if not action_schema:
            # Fallback to generic format if schema not found
            return f"""{{
    "thought": "Brief explanation of parameter choices",
    "tool_calls": [
        {{
            "name": "{planned_step.tool_name}",
            "arguments": {{
                "action": "{planned_step.action}",
                ... parameters ...
            }}
        }}
    ]
}}"""

        # Build parameter examples from schema
        param_examples = {}
        param_examples["action"] = planned_step.action

        for param in action_schema.params:
            param_name = param.name
            param_desc = param.description

            # Generate example value based on description
            if "path" in param_name.lower() or "file" in param_name.lower():
                example_value = '"example.txt"'
            elif "content" in param_name.lower():
                example_value = '"[RESULT:Session0_1]" or "actual content"'
            elif param.param_type == "int":
                example_value = "100"
            elif param.param_type == "bool":
                example_value = "true"
            else:
                example_value = f'"<{param_desc[:50]}>"' if param_desc else '"example value"'

            # Mark required parameters
            if param.required:
                param_examples[param_name] = f"{example_value}  // REQUIRED"
            else:
                param_examples[param_name] = f"{example_value}  // optional"

        # Format as JSON with comments
        param_lines = []
        for key, value in param_examples.items():
            param_lines.append(f'                "{key}": {value}')

        params_str = ",\n".join(param_lines)

        return f"""{{
    "thought": "Brief explanation of why these specific parameters are needed for this step",
    "tool_calls": [
        {{
            "name": "{planned_step.tool_name}",
            "arguments": {{
{params_str}
            }}
        }}
    ]
}}"""

    def _substitute_result_refs(self, params: Dict) -> Dict:
        """파라미터의 [RESULT:*] 참조를 실제 값으로 치환"""
        import re

        def replace_ref(value):
            if not isinstance(value, str):
                return value

            # [RESULT:session_id_step] 패턴 찾기 (예: [RESULT:Session0_1])
            pattern = r'\[RESULT:([^\]]+)\]'
            matches = re.findall(pattern, value)

            for result_key in matches:
                output = self.storage.get_output_by_key(result_key)
                if output is not None:
                    # 전체 문자열이 REF만 있으면 output을 직접 사용, 아니면 문자열 치환
                    if value.strip() == f"[RESULT:{result_key}]":
                        return output
                    else:
                        value = value.replace(f"[RESULT:{result_key}]", str(output))

            return value

        # 모든 파라미터 값에 대해 치환 수행
        substituted = {}
        for key, value in params.items():
            if isinstance(value, str):
                substituted[key] = replace_ref(value)
            elif isinstance(value, dict):
                substituted[key] = self._substitute_result_refs(value)
            elif isinstance(value, list):
                substituted[key] = [replace_ref(v) if isinstance(v, str) else v for v in value]
            else:
                substituted[key] = value

        return substituted

    def _substitute_file_refs(self, params: Dict) -> Dict:
        """파라미터의 [FILE:*] 참조를 실제 파일 내용으로 치환"""
        import re
        import os
        from pathlib import Path

        BASE_PATH = os.path.join(".", "workspace")
        BASE_PATH = os.path.join(BASE_PATH, "files")

        def read_file_content(file_path: str) -> str:
            """파일 내용을 읽어서 반환"""
            # 상대 경로를 base_path 기준으로 해석
            path = BASE_PATH / file_path
            
            if not path.exists():
                raise FileNotFoundError(f"[FILE:{file_path}] - File not found: {path}")
            if not path.is_file():
                raise ValueError(f"[FILE:{file_path}] - Not a file: {path}")
            
            # 경로 탈출 방지 (보안)
            try:
                path.resolve().relative_to(BASE_PATH.resolve())
            except ValueError:
                raise ValueError(f"[FILE:{file_path}] - Path escape not allowed: {file_path}")
            
            try:
                return path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                raise ValueError(f"[FILE:{file_path}] - Cannot read binary file as text: {path}")

        def replace_ref(value):
            if not isinstance(value, str):
                return value

            # [FILE:path] 패턴 찾기 (예: [FILE:src/main.c])
            pattern = r'\[FILE:([^\]]+)\]'
            matches = re.findall(pattern, value)

            for file_path in matches:
                file_path_stripped = file_path.strip()
                content = read_file_content(file_path_stripped)
                
                # 전체 문자열이 FILE REF만 있으면 content를 직접 사용
                if value.strip() == f"[FILE:{file_path}]":
                    return content
                else:
                    value = value.replace(f"[FILE:{file_path}]", content)

            return value

        # 모든 파라미터 값에 대해 치환 수행
        substituted = {}
        for key, value in params.items():
            if isinstance(value, str):
                substituted[key] = replace_ref(value)
            elif isinstance(value, dict):
                substituted[key] = self._substitute_file_refs(value)
            elif isinstance(value, list):
                substituted[key] = [
                    replace_ref(v) if isinstance(v, str) 
                    else self._substitute_file_refs(v) if isinstance(v, dict)
                    else v 
                    for v in value
                ]
            else:
                substituted[key] = value

        return substituted


    def _substitute_all_refs(self, params: Dict) -> Dict:
        """모든 REF ([FILE:*], [RESULT:*])를 치환"""
        params = self._substitute_file_refs(params)
        params = self._substitute_result_refs(params)
        return params

    def _parse_plan(self, plan_data: List[Dict]) -> List[PlannedStep]:
        """계획 데이터를 PlannedStep 리스트로 변환"""
        planned_steps = []
        if not isinstance(plan_data, list):
            return planned_steps
        
        for item in plan_data:
            if not isinstance(item, dict):
                continue
            try:
                step = PlannedStep(
                    step=item.get("step", len(planned_steps) + 1),
                    tool_name=item.get("tool_name", item.get("tool", "")),
                    action=item.get("action", ""),
                    description=item.get("description", ""),
                    params_hint=item.get("params_hint", item.get("params", {})),
                    status="pending"
                )
                planned_steps.append(step)
            except Exception as e:
                self.logger.warn(f"Step 파싱 실패: {e}")
                continue
        
        return planned_steps
    
    def _format_plan_summary(self) -> str:
        """계획 요약 문자열"""
        if not self._plan:
            return "계획 없음"
        
        lines = ["📋 **실행 계획**\n"]
        for step in self._plan:
            status_icon = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}.get(step.status, "⏳")
            lines.append(f"{status_icon} Step {step.step}: {step.tool_name}.{step.action}")
            lines.append(f"   └─ {step.description}")
        
        return "\n".join(lines)
    
    def _format_plan_with_status(self) -> str:
        """상태 포함 계획"""
        if not self._plan:
            return "No plan"
        
        lines = []
        for step in self._plan:
            status = {"pending": "[ ]", "running": "[→]", "completed": "[✓]", "failed": "[✗]"}.get(step.status, "[ ]")
            lines.append(f"{status} Step {step.step}: {step.tool_name}.{step.action} - {step.description}")
        
        return "\n".join(lines)
    
    def _format_previous_results(self, results: List[Dict]) -> str:
        """이전 결과 포맷"""
        if not results:
            return "None yet."
        
        formatted = ""
        for r in results:
            step = r.get('step', '?')
            tool = r.get('tool', r.get('executor', 'unknown'))
            action = r.get('action', 'unknown')
            status = "success" if r.get('success', False) == True else "failure"
            output = str(r.get("output", ""))
            error = str(r.get("error", ""))
            
            status_icon = "✅" if status == "success" else "❌"
            
            formatted += f"""
[Step {step}] {status_icon} {tool}.{action} - {status.upper()}
"""
            if status == "success":
                formatted += f"""Output: {output}"""
            else:
                formatted += f"""Error: {error}"""

        return formatted
    
    def _generate_final_answer(self, user_query: str) -> str:
        """Phase 3: 최종 응답 생성"""
        results = self.storage.get_results()
        previous_results = self._format_previous_results(results)
        plan_status = self._format_plan_with_status()

        system_prompt = """You are generating the final response based on executed tasks.

## Your Task
1. Review what was done (plan execution results)
2. Summarize the findings
3. Provide a helpful, complete answer to the user

## Rules
- Be concise but thorough
- Reference specific results when relevant
- If any step failed, mention it and provide alternatives if possible
- Respond naturally, not in JSON format
- If the task was to save a file, confirm it was saved successfully
- IMPORTANT: Respond in Korean (한글로 답변하세요)"""

        user_prompt = f"""## Original User Query
{user_query}

## Execution Plan
{plan_status}

## Execution Results
{previous_results}

Based on the above results, provide a final answer to the user's query."""

        response = self._call_llm_api(system_prompt, user_prompt)
        return response.strip()
    
    # =========================================================================
    # LLM API & Parsing (변경 없음)
    # =========================================================================
    
    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """Ollama Chat API 호출 (with performance logging)"""
        try:
            import requests
            import time

            url = f"{self.llm_config.base_url}/api/chat"

            # Chat API 형식: messages 배열 사용
            payload = {
                "model": self.llm_config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": self.llm_config.temperature,
                    "top_p": self.llm_config.top_p,
                    "top_k": self.llm_config.top_k,
                    "num_predict": self.llm_config.max_tokens,
                    "repeat_penalty": self.llm_config.repeat_penalty,
                    "frequency_penalty": self.llm_config.frequency_penalty,
                    "presence_penalty": self.llm_config.presence_penalty,
                    "num_ctx": self.llm_config.num_ctx
                }
            }

            # Measure performance
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=self.llm_config.timeout)
            response.raise_for_status()
            elapsed_time = time.time() - start_time

            data = response.json()

            # Extract performance metrics from Ollama response
            # Ollama provides: eval_count (output tokens), prompt_eval_count (input tokens),
            # eval_duration (ns), prompt_eval_duration (ns)
            prompt_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)
            total_duration_ns = data.get("total_duration", 0)
            eval_duration_ns = data.get("eval_duration", 0)

            # Calculate metrics
            prompt_length = len(system_prompt) + len(user_prompt)  # Character count
            response_length = len(data.get("message", {}).get("content", ""))  # Character count

            # Tokens per second (based on eval_duration for output tokens)
            tokens_per_sec = 0
            if eval_duration_ns > 0:
                tokens_per_sec = (output_tokens / eval_duration_ns) * 1e9  # Convert ns to seconds

            # TTFT approximation (prompt eval duration)
            ttft_ms = data.get("prompt_eval_duration", 0) / 1e6  # Convert ns to ms

            # Log performance metrics
            self.logger.info("LLM Performance Metrics", {
                "elapsed_time_sec": round(elapsed_time, 2),
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
                "tokens_per_sec": round(tokens_per_sec, 2),
                "ttft_ms": round(ttft_ms, 2),
                "prompt_chars": prompt_length,
                "response_chars": response_length,
                "model": self.llm_config.model
            })

            # Chat API 응답 형식: message.content
            return data.get("message", {}).get("content", "")

        except ImportError:
            self.logger.warn("requests 패키지 없음 - Mock 응답 반환")
            return self._mock_llm_response()
        except Exception as e:
            self.logger.error(f"LLM API 호출 실패: {str(e)}")
            raise

    def _parse_tool_call_response(self, raw_response: str) -> LLMResponse:
        """LLM 응답에서 tool_calls를 추출하여 LLMResponse 객체로 변환"""
        try:
            data = parse_json_strict(raw_response)

            tool_calls = None
            if data.get("tool_calls"):
                all_known_actions = self._get_known_actions()
                tool_calls = []
                for tc in data["tool_calls"]:
                    tool_name = tc.get("name", "")
                    # tool_name에서 "." 이후 제거 (e.g., "file_tool.read" -> "file_tool")
                    base_tool_name = tool_name.split(".")[0] if "." in tool_name else tool_name
                    # 해당 tool의 known_actions만 가져오기
                    tool_known_actions = all_known_actions.get(base_tool_name, [])
                    tool_calls.append(
                        ToolCall(
                            name=tool_name,
                            arguments=tc.get("arguments", {}),
                            known_actions=tool_known_actions
                        )
                    )

            return LLMResponse(
                thought=data.get("thought"),
                tool_calls=tool_calls,
                content=data.get("content"),
                raw_response=raw_response
            )

        except (ValueError, json.JSONDecodeError) as e:
            self.logger.warn(f"JSON 파싱 실패: {str(e)}")
            return LLMResponse(content=raw_response, raw_response=raw_response)
    
    def _mock_llm_response(self) -> str:
        """테스트용 Mock 응답"""
        results = self.storage.get_results()
        
        if not results:
            return json.dumps({
                "thought": "파일을 먼저 읽어야 합니다 (Mock)",
                "tool_calls": [{
                    "name": "file_tool",
                    "arguments": {"action": "read", "read_path": "test.txt"}
                }]
            })
        else:
            return json.dumps({
                "thought": "충분한 정보를 얻었습니다 (Mock)",
                "tool_calls": None,
                "content": f"Mock 분석 완료. {len(results)}개의 step을 수행했습니다."
            })
