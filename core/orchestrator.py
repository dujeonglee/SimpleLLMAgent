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

from core.shared_storage import SharedStorage, DebugLogger
from core.base_tool import ToolRegistry, ToolResult


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
    repeat_penalty: float = 1.1
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
    PLAN_READY = "plan_ready"
    THINKING = "thinking"
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
    
    def to_dict(self) -> Dict:
        return asdict(self)


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
        on_step_complete: Callable[[StepInfo], None] = None,
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

        if hasattr(llm_tool, 'set_previous_result_getter'):
            llm_tool.set_previous_result_getter(self.storage.get_last_output)
            self.logger.info("LLMTool에 previous_result_getter 주입 완료")

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
    
    def run(self, user_query: str) -> str:
        """동기 실행"""
        final_response = ""
        for step_info in self.run_stream(user_query):
            if step_info.type == StepType.FINAL_ANSWER:
                final_response = step_info.content
        return final_response
    
    def run_stream(self, user_query: str) -> Generator[StepInfo, None, None]:
        """Streaming 실행 - Plan & Execute 패턴"""
        self._stopped = False
        self._current_step = 0
        self._plan = []
        
        session_id = self.storage.start_session(user_query)
        self.logger.info(f"실행 시작: {user_query[:50]}...")
        
        try:
            yield StepInfo(type=StepType.PLANNING, step=0, content="작업 계획 수립 중...")
            
            plan_response = self._create_plan(user_query)
            
            if self._stopped:
                self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                yield StepInfo(type=StepType.ERROR, step=self._current_step, content="사용자 요청으로 취소됨")
                return
            
            if plan_response.get("direct_answer"):
                self.storage.complete_session(
                    final_response=plan_response["direct_answer"],
                    status="completed"
                )
                yield StepInfo(type=StepType.FINAL_ANSWER, step=0, content=plan_response["direct_answer"])
                return
            
            self._plan = self._parse_plan(plan_response.get("plan", []))
            
            if not self._plan:
                direct_response = plan_response.get("thought", "요청을 처리할 수 없습니다.")
                self.storage.complete_session(final_response=direct_response, status="completed")
                yield StepInfo(type=StepType.FINAL_ANSWER, step=0, content=direct_response)
                return
            
            plan_summary = self._format_plan_summary()
            yield StepInfo(type=StepType.PLAN_READY, step=0, content=plan_summary)
            
            # Phase 2: Execution
            for planned_step in self._plan:
                self._current_step = planned_step.step
                planned_step.status = "running"
                
                yield StepInfo(
                    type=StepType.THINKING,
                    step=self._current_step,
                    content=f"Step {self._current_step}: {planned_step.description}"
                )
                
                # LLM 호출 전 중지 체크
                if self._stopped:
                    self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                    yield StepInfo(type=StepType.ERROR, step=self._current_step, content="사용자 요청으로 취소됨")
                    return
                
                execution_response = self._get_execution_params(user_query, planned_step)
                
                # LLM 호출 후 중지 체크
                if self._stopped:
                    self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                    yield StepInfo(type=StepType.ERROR, step=self._current_step, content="사용자 요청으로 취소됨")
                    return
                
                if execution_response.thought:
                    yield StepInfo(
                        type=StepType.THINKING,
                        step=self._current_step,
                        content=execution_response.thought
                    )
                
                if execution_response.tool_calls:
                    for tool_call in execution_response.tool_calls:
                        # Tool 실행 전 중지 체크
                        if self._stopped:
                            self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                            yield StepInfo(type=StepType.ERROR, step=self._current_step, content="사용자 요청으로 취소됨")
                            return
                        
                        yield StepInfo(
                            type=StepType.TOOL_CALL,
                            step=self._current_step,
                            content=tool_call.arguments,
                            tool_name=tool_call.name,
                            action=tool_call.action
                        )
                        
                        # Tool 실행
                        result = self._execute_tool(tool_call)

                        # ToolResult에 ExecutionResult 필드 설정
                        result.step = self._current_step
                        result.executor = tool_call.name
                        result.executor_type = "tool"
                        result.action = tool_call.action
                        result.input = tool_call.params

                        # Tool 실행 후 중지 체크
                        if self._stopped:
                            self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                            yield StepInfo(type=StepType.ERROR, step=self._current_step, content="사용자 요청으로 취소됨")
                            return

                        # ToolResult를 그대로 전달
                        self.storage.add_result(result)
                        
                        planned_step.status = "completed" if result.success else "failed"
                        
                        yield StepInfo(
                            type=StepType.TOOL_RESULT,
                            step=self._current_step,
                            content=result.output if result.success else result.error,
                            tool_name=tool_call.name,
                            action=tool_call.action
                        )
                    # inner loop에서 중지된 경우
                    if self._stopped:
                        self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                        yield StepInfo(type=StepType.ERROR, step=self._current_step, content="사용자 요청으로 취소됨")
                        return
                if self.on_step_complete:
                    self.on_step_complete(StepInfo(
                        type=StepType.TOOL_RESULT,
                        step=self._current_step,
                        content="Step completed"
                    ))
            
            # 중지된 경우 Final Answer 생성 스킵
            if self._stopped:
                self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                yield StepInfo(type=StepType.ERROR, step=self._current_step, content="사용자 요청으로 취소됨")
                return
            
            # Phase 3: Final Answer
            yield StepInfo(
                type=StepType.THINKING,
                step=self._current_step + 1,
                content="최종 응답 생성 중..."
            )
            
            if self._stopped:
                self.storage.complete_session(final_response="사용자 요청으로 취소됨", status="error")
                yield StepInfo(type=StepType.ERROR, step=self._current_step, content="사용자 요청으로 취소됨")
                return
            
            final_response = self._generate_final_answer(user_query)
            
            self.storage.complete_session(final_response=final_response, status="completed")
            
            yield StepInfo(type=StepType.FINAL_ANSWER, step=self._current_step + 1, content=final_response)
        
        except Exception as e:
            error_msg = f"실행 중 오류 발생: {str(e)}"
            self.logger.error(error_msg, {"exception": type(e).__name__})
            self.storage.complete_session(final_response=error_msg, status="error")
            yield StepInfo(type=StepType.ERROR, step=self._current_step, content=error_msg)
    
    # =========================================================================
    # Tool Execution (Updated)
    # =========================================================================
    
    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Tool 실행 - [PREVIOUS_RESULT] 플레이스홀더 자동 대체"""
        
        # 파라미터 복사 및 플레이스홀더 처리
        params = tool_call.params.copy()
        
        for key, value in params.items():
            if isinstance(value, str):
                # [PREVIOUS_RESULT] 플레이스홀더 처리
                if "[PREVIOUS_RESULT]" in value:
                    previous = self.storage.get_last_output()
                    if previous:
                        params[key] = value.replace("[PREVIOUS_RESULT]", str(previous))
                        self.logger.debug(f"플레이스홀더 대체: {key}")
                    else:
                        self.logger.warn(f"이전 결과가 없어 플레이스홀더 대체 실패: {key}")
        
        self.logger.info(f"Tool 실행: {tool_call.name}.{tool_call.action}", {
            "params_keys": list(params.keys())
        })
        
        result = self.tools.execute(
            tool_name=tool_call.name,
            action=tool_call.action,
            params=params
        )
        
        return result
    
    # =========================================================================
    # Planning Methods
    # =========================================================================
    
    def _create_plan(self, user_query: str) -> Dict:
        """Phase 1: 전체 실행 계획 수립"""
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
2. Decide if tools are needed
3. If yes, create a step-by-step plan
4. If no tools needed, provide direct answer

## IMPORTANT: Data Flow Between Steps
- Each step's output becomes available for the next step
- For llm_tool: It can automatically access the previous step's output
- For file_tool.write: Use [PREVIOUS_RESULT] placeholder for content from previous step

## Response Format (JSON only)

If tools are needed:
{{
    "thought": "Analysis of what needs to be done",
    "plan": [
        {{
            "step": 1,
            "tool_name": "file_tool",
            "action": "read",
            "description": "Read the source file"
        }},
        {{
            "step": 2,
            "tool_name": "llm_tool",
            "action": "analyze",
            "description": "Analyze the file content (uses previous output automatically)"
        }},
        {{
            "step": 3,
            "tool_name": "file_tool",
            "action": "write",
            "description": "Save analysis result to file (uses [PREVIOUS_RESULT])"
        }}
    ]
}}

If no tools needed:
{{
    "thought": "This is a simple question I can answer directly",
    "direct_answer": "Your answer here"
}}

## Rules
- Maximum {self.max_steps} steps
- Use only available tools and actions
- Each step should have clear purpose
- Order steps logically (data dependencies)
- Respond with valid JSON only"""

        user_prompt = f"""## User Request
{user_query}

Create an execution plan or provide direct answer. Respond with JSON only."""

        self.logger.debug("Planning 호출", {"query": user_query[:50]})
        
        raw_response = self._call_llm_api(system_prompt, user_prompt)

        try:
            parsed = self._extract_json(raw_response)
            self.logger.debug("Plan 생성 완료", {
                "has_plan": "plan" in parsed,
                "has_direct_answer": "direct_answer" in parsed
            })
            return parsed
        except Exception as e:
            self.logger.error(f"Plan 파싱 실패: {e}")
            return {"direct_answer": f"계획 수립 중 오류가 발생했습니다: {str(e)}"}
    
    def _get_execution_params(self, user_query: str, planned_step: PlannedStep) -> LLMResponse:
        """계획된 step 실행을 위한 정확한 파라미터 결정"""
        tools_schema = json.dumps(
            self.tools.get_all_schemas(),
            indent=2,
            ensure_ascii=False
        )
        
        results = self.storage.get_results()
        previous_results = self._format_previous_results(results)
        plan_status = self._format_plan_with_status()
        
        # 이전 결과 요약 (파라미터 결정에 활용)
        last_output_preview = ""
        if results:
            last_output = self.storage.get_last_output()
            if isinstance(last_output, str) and len(last_output) > 500:
                last_output_preview = last_output[:500] + "..."
            else:
                last_output_preview = str(last_output)[:500]
        
        system_prompt = f"""You are executing a planned task. Provide exact parameters for the current step.

## Available Tools
{tools_schema}

## Current Plan Status
{plan_status}

## IMPORTANT: Using Previous Step Results

### For llm_tool:
- You can OMIT the content parameter (e.g., analyze_content, summarize_content)
- The tool will automatically use the previous step's output
- Example: {{"action": "analyze", "analyze_type": "static"}} - content is omitted

### For file_tool.write:
- Use [PREVIOUS_RESULT] as a placeholder for write_content
- Example: {{"action": "write", "write_path": "out.md", "write_content": "[PREVIOUS_RESULT]"}}

### Last Step Output Preview:
{last_output_preview if last_output_preview else "(No previous output)"}

## Response Format
{{
    "thought": "Why these parameters",
    "tool_calls": [
        {{
            "name": "{planned_step.tool_name}",
            "arguments": {{
                "action": "{planned_step.action}",
                ... parameters ...
            }}
        }}
    ]
}}

## Rules
- Execute ONLY the current step (Step {planned_step.step})
- Use parameter names with action prefix (e.g., read_path, write_content)
- For llm_tool: omit content param to use previous result
- For file_tool.write: use [PREVIOUS_RESULT] for content from previous step
- Respond with valid JSON only"""

        user_prompt = f"""## Original User Query
{user_query}

## Previous Results
{previous_results}

## Current Step to Execute
Step {planned_step.step}: {planned_step.tool_name}.{planned_step.action}
Description: {planned_step.description}

Provide exact parameters for this step. Respond with JSON only."""

        raw_response = self._call_llm_api(system_prompt, user_prompt)
        return self._parse_llm_response(raw_response)
    
    # =========================================================================
    # Helper Methods (변경 없음 - 기존 코드 유지)
    # =========================================================================
    
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
            status = r.get('status', 'unknown')
            output = str(r.get("output", ""))
            
            # 출력 길이 제한
            if len(output) > 1000:
                output = output[:1000] + "... (truncated)"
            
            status_icon = "✅" if status == "success" else "❌"
            
            formatted += f"""
[Step {step}] {status_icon} {tool}.{action} - {status.upper()}
  Output: {output}
"""
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
- If the task was to save a file, confirm it was saved successfully"""

        user_prompt = f"""## Original User Query
{user_query}

## Execution Plan
{plan_status}

## Execution Results
{previous_results}

Based on the above results, provide a final answer to the user's query."""

        response = self._call_llm_api(system_prompt, user_prompt)
        return response.strip()
    
    def _is_complete(self) -> bool:
        if self._stopped:
            return True
        if self._current_step >= self.max_steps:
            return True
        return False
    
    # =========================================================================
    # LLM API & Parsing (변경 없음)
    # =========================================================================
    
    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """Ollama API 호출"""
        try:
            import requests
            
            url = f"{self.llm_config.base_url}/api/generate"
            
            payload = {
                "model": self.llm_config.model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": self.llm_config.temperature,
                    "top_p": self.llm_config.top_p,
                    "num_predict": self.llm_config.max_tokens,
                    "repeat_penalty": self.llm_config.repeat_penalty,
                    "num_ctx": self.llm_config.num_ctx
                }
            }

            response = requests.post(url, json=payload, timeout=self.llm_config.timeout)
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
        except ImportError:
            self.logger.warn("requests 패키지 없음 - Mock 응답 반환")
            return self._mock_llm_response()
        except Exception as e:
            self.logger.error(f"LLM API 호출 실패: {str(e)}")
            raise
    
    def _extract_json(self, text: str) -> Dict:
        """텍스트에서 JSON 추출"""
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json.loads(json_match.group(1))
        
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group(0))
        
        raise ValueError("JSON not found in response")
    
    def _extract_json_str(self, text: str) -> Optional[str]:
        """괄호 매칭으로 JSON 추출"""
        start = text.find('{')
        if start == -1:
            return None
        
        depth = 0
        in_string = False
        escape = False
        
        for i, char in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        
        return None
    
    def _fix_triple_quotes(self, text: str) -> str:
        """삼중 따옴표 처리"""
        def replace_triple_quotes(match):
            content = match.group(1)
            content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            content = content.replace('"', '\\"')
            return f'"{content}"'
        
        text = re.sub(r'"""(.*?)"""', replace_triple_quotes, text, flags=re.DOTALL)
        text = re.sub(r"'''(.*?)'''", replace_triple_quotes, text, flags=re.DOTALL)
        return text
    
    def _sanitize_json_string(self, text: str) -> str:
        """JSON 문자열 정리"""
        text = self._fix_triple_quotes(text)
        
        result = []
        in_string = False
        escape = False
        
        for char in text:
            if escape:
                result.append(char)
                escape = False
                continue
            if char == '\\':
                result.append(char)
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                result.append(char)
                continue
            
            if in_string:
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif ord(char) < 32:
                    result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    def _parse_llm_response(self, raw_response: str) -> LLMResponse:
        """LLM 응답 파싱"""
        try:
            sanitized = self._sanitize_json_string(raw_response)
            json_str = self._extract_json_str(sanitized)

            if not json_str:
                return LLMResponse(content=raw_response, raw_response=raw_response)

            data = json.loads(json_str)

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

        except json.JSONDecodeError as e:
            self.logger.warn(f"JSON 파싱 실패: {str(e)}")
            return LLMResponse(content=raw_response, raw_response=raw_response)
    
    def _generate_partial_response(self) -> str:
        """최대 step 도달 시 부분 응답"""
        results = self.storage.get_results()
        
        if not results:
            return "작업을 완료하지 못했습니다."
        
        last_result = results[-1]
        return f"""최대 실행 단계({self.max_steps})에 도달했습니다.

지금까지의 진행 상황:
- 총 {len(results)}개의 작업 수행
- 마지막 작업: {last_result['executor']}.{last_result['action']}

부분 결과:
{str(last_result.get('output', ''))[:500]}"""
    
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
