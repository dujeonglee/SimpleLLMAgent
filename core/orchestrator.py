"""
Orchestrator Module
===================
Multi-Agent Chatbot의 두뇌 역할.
ReAct 패턴으로 Tool들을 조율하고 최종 응답을 생성합니다.

설계 결정:
- ReAct (단계별 결정): 매 step마다 LLM이 다음 행동 결정
- Function Calling 스타일: tool_calls 형식으로 응답
- Streaming 지원: 실시간 결과 출력
- 동적 LLM 설정: 모델/파라미터 실시간 변경 가능
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

from .shared_storage import SharedStorage, DebugLogger
from .base_tool import ToolRegistry, ToolResult


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LLMConfig:
    """LLM 동적 설정 - UI에서 실시간 변경 가능"""
    
    # 모델 선택
    model: str = "llama3.2"
    
    # 생성 파라미터
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    
    # 연결 설정
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    
    # 고급 옵션
    repeat_penalty: float = 1.1
    num_ctx: int = 4096
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def update(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class ToolCall:
    """LLM이 요청한 Tool 호출 정보"""
    name: str           # tool 이름 (예: file_tool)
    arguments: Dict     # action + params
    
    # Tool별 action 추론 규칙
    ACTION_INFERENCE_RULES = {
        "file_tool": {
            # arguments에 특정 키가 있으면 해당 action으로 추론
            "content": "write",      # content가 있으면 write
            "path": "read",          # path만 있으면 read
            "pattern": "list_dir",   # pattern이 있으면 list_dir
        },
        "web_tool": {
            "url": "fetch",          # url이 있으면 fetch
            "keyword": "search",     # keyword가 있으면 search
            "query": "search",       # query가 있으면 search
        },
        "llm_tool": {
            "content": "summarize",  # content가 있으면 summarize
            "prompt": "ask",         # prompt가 있으면 ask
            "question": "ask",       # question이 있으면 ask
            "text": "analyze",       # text가 있으면 analyze
        }
    }
    
    def __post_init__(self):
        """name에 action이 포함된 경우 분리, action 없으면 추론"""
        # 1. name에 action이 포함된 경우 (예: llm_tool.ask -> llm_tool, ask)
        if "." in self.name:
            parts = self.name.split(".", 1)
            self.name = parts[0]
            if "action" not in self.arguments:
                self.arguments["action"] = parts[1]
        
        # 2. action이 여전히 없으면 arguments 기반으로 추론
        if not self.arguments.get("action"):
            inferred_action = self._infer_action()
            if inferred_action:
                self.arguments["action"] = inferred_action
    
    def _infer_action(self) -> Optional[str]:
        """arguments 기반으로 action 추론"""
        rules = self.ACTION_INFERENCE_RULES.get(self.name, {})
        
        # 우선순위: content > 기타 키
        for key, action in rules.items():
            if key in self.arguments:
                return action
        
        # 기본 action (fallback)
        defaults = {
            "file_tool": "read",
            "web_tool": "search",
            "llm_tool": "ask"
        }
        return defaults.get(self.name)
    
    @property
    def action(self) -> str:
        return self.arguments.get("action", "")
    
    @property
    def params(self) -> Dict:
        """action을 제외한 나머지 파라미터"""
        return {k: v for k, v in self.arguments.items() if k != "action"}


@dataclass
class LLMResponse:
    """LLM 응답 파싱 결과"""
    thought: Optional[str] = None      # LLM의 사고 과정
    tool_calls: Optional[List[ToolCall]] = None  # Tool 호출 요청
    content: Optional[str] = None      # 최종 답변 (tool_calls가 None일 때)
    raw_response: str = ""             # 원본 응답
    
    @property
    def is_final_answer(self) -> bool:
        """최종 답변인지 여부"""
        return self.tool_calls is None and self.content is not None


class StepType(Enum):
    """Step 유형"""
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


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
# Orchestrator
# =============================================================================

class Orchestrator:
    """
    Multi-Agent Chatbot Orchestrator
    
    ReAct 패턴으로 동작:
    1. LLM에게 다음 행동 질문
    2. Tool 실행
    3. 결과 저장 + Streaming 출력
    4. 완료 여부 판단 → 반복 또는 종료
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
        
        self.logger.info("Orchestrator 초기화", {
            "max_steps": max_steps,
            "llm_model": self.llm_config.model,
            "tools": self.tools.list_tools()
        })
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def update_llm_config(self, **kwargs):
        """LLM 설정 동적 변경 (UI에서 호출)"""
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
        """
        동기 실행 - 최종 응답만 반환
        
        Args:
            user_query: 사용자 쿼리
            
        Returns:
            str: 최종 응답
        """
        final_response = ""
        for step_info in self.run_stream(user_query):
            if step_info.type == StepType.FINAL_ANSWER:
                final_response = step_info.content
        
        return final_response
    
    def run_stream(self, user_query: str) -> Generator[StepInfo, None, None]:
        """
        Streaming 실행 - 각 step 결과를 yield
        
        Args:
            user_query: 사용자 쿼리
            
        Yields:
            StepInfo: 각 step의 정보
        """
        self._stopped = False
        self._current_step = 0
        
        # 세션 시작
        session_id = self.storage.start_session(user_query)
        self.logger.info(f"실행 시작: {user_query[:50]}...")
        
        try:
            while not self._is_complete():
                self._current_step += 1
                
                # 1. LLM에게 다음 행동 질문
                yield StepInfo(
                    type=StepType.THINKING,
                    step=self._current_step,
                    content="다음 행동 결정 중..."
                )
                
                llm_response = self._ask_llm(user_query)
                
                # 2. 사고 과정 출력
                if llm_response.thought:
                    yield StepInfo(
                        type=StepType.THINKING,
                        step=self._current_step,
                        content=llm_response.thought
                    )
                
                # 3. 최종 답변인 경우
                if llm_response.is_final_answer:
                    self.storage.complete_session(
                        final_response=llm_response.content,
                        status="completed"
                    )
                    
                    yield StepInfo(
                        type=StepType.FINAL_ANSWER,
                        step=self._current_step,
                        content=llm_response.content
                    )
                    break
                
                # 4. Tool 호출 실행
                if llm_response.tool_calls:
                    for tool_call in llm_response.tool_calls:
                        # Tool 호출 알림
                        yield StepInfo(
                            type=StepType.TOOL_CALL,
                            step=self._current_step,
                            content=tool_call.arguments,
                            tool_name=tool_call.name,
                            action=tool_call.action
                        )
                        
                        # Tool 실행
                        result = self._execute_tool(tool_call)
                        
                        # 결과 저장
                        self.storage.add_result(
                            executor=tool_call.name,
                            executor_type="tool",
                            action=tool_call.action,
                            input_data=tool_call.params,
                            output=result.output,
                            status="success" if result.success else "error",
                            error_message=result.error
                        )
                        
                        # 결과 출력
                        yield StepInfo(
                            type=StepType.TOOL_RESULT,
                            step=self._current_step,
                            content=result.output if result.success else result.error,
                            tool_name=tool_call.name,
                            action=tool_call.action
                        )
                
                # Callback 호출
                if self.on_step_complete:
                    self.on_step_complete(StepInfo(
                        type=StepType.TOOL_RESULT,
                        step=self._current_step,
                        content="Step completed"
                    ))
            
            # 최대 step 도달
            if self._current_step >= self.max_steps:
                final_msg = self._generate_partial_response()
                self.storage.complete_session(
                    final_response=final_msg,
                    status="max_steps_reached"
                )
                
                yield StepInfo(
                    type=StepType.FINAL_ANSWER,
                    step=self._current_step,
                    content=final_msg
                )
        
        except Exception as e:
            error_msg = f"실행 중 오류 발생: {str(e)}"
            self.logger.error(error_msg, {"exception": type(e).__name__})
            
            self.storage.complete_session(
                final_response=error_msg,
                status="error"
            )
            
            yield StepInfo(
                type=StepType.ERROR,
                step=self._current_step,
                content=error_msg
            )
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _is_complete(self) -> bool:
        """완료 여부 판단"""
        if self._stopped:
            self.logger.info("수동 중지로 종료")
            return True
        
        if self._current_step >= self.max_steps:
            self.logger.info(f"최대 step({self.max_steps}) 도달로 종료")
            return True
        
        return False
    
    def _ask_llm(self, user_query: str) -> LLMResponse:
        """LLM에게 다음 행동 질문"""
        
        # System prompt 생성
        system_prompt = self._build_system_prompt()
        
        # User prompt 생성 (현재 컨텍스트 포함)
        user_prompt = self._build_user_prompt(user_query)

        self.logger.debug("LLM 호출", {
            "model": self.llm_config.model,
            "user_prompt_length": len(user_prompt)
        })
        
        # LLM API 호출
        raw_response = self._call_llm_api(system_prompt, user_prompt)
        
        # 응답 파싱
        parsed = self._parse_llm_response(raw_response)
        
        self.logger.debug("LLM 응답 파싱 완료", {
            "has_tool_calls": parsed.tool_calls is not None,
            "is_final": parsed.is_final_answer
        })
        
        return parsed
    
    def _build_system_prompt(self) -> str:
        """System prompt 생성"""
        tools_schema = json.dumps(
            self.tools.get_all_schemas(),
            indent=2,
            ensure_ascii=False
        )
        
        return f"""You are an AI assistant that helps users by using available tools.

## Available Tools
{tools_schema}

## Response Format
You must respond in valid JSON format only. No other text before or after the JSON.

1. To use a tool:
{{
    "thought": "Why I'm taking this action",
    "tool_calls": [
        {{
            "name": "tool_name",
            "arguments": {{
                "action": "action_name",
                "param1": "value1"
            }}
        }}
    ]
}}

2. To give final answer (when you have enough information):
{{
    "thought": "I have enough information to answer",
    "tool_calls": null,
    "content": "Your final answer here"
}}

## Rules
- Always respond with valid JSON only
- Use tools when you need external information (read files, search web, etc.)
- Give final answer when you have enough information to respond
- Be concise and accurate
- One tool call at a time is recommended for clarity

## IMPORTANT: JSON String Format
- NEVER use triple quotes (\"\"\") in JSON - they are invalid
- Use \\n for newlines inside strings
- Escape double quotes as \\"
- Example for code content: "content": "def hello():\\n    print(\\"Hello\\")"
"""
    
    def _build_user_prompt(self, user_query: str) -> str:
        """User prompt 생성 (현재 컨텍스트 포함)"""
        
        results = self.storage.get_results()
        
        if not results:
            previous_results = "None yet."
        else:
            previous_results = ""
            for r in results:
                step = r.get('step', '?')
                tool = r.get('tool', r.get('executor', 'unknown'))
                action = r.get('action', 'unknown')
                status = r.get('status', 'unknown')
                output = str(r.get("output", ""))
                
                # 상태 아이콘
                status_icon = "✅" if status == "success" else "❌"
                
                output_preview = output
                
                previous_results += f"""
[Step {step}] {status_icon} {tool}.{action} - {status.upper()}
  Result: {output_preview}
"""
        
        # 중복 호출 방지 힌트
        hint = ""
        if results:
            last_result = results[-1]
            if last_result.get("status") == "success":
                hint = "\n⚠️ IMPORTANT: The previous step was SUCCESSFUL. Do NOT repeat the same action. Proceed to the next step or provide final answer.\n"
        
        return f"""## User Query
{user_query}

## Previous Results
{previous_results}
{hint}
## Your Task
Decide the next action. If the task is complete, provide final answer. Respond with JSON only."""
    
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
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.llm_config.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
        except ImportError:
            self.logger.warn("requests 패키지 없음 - Mock 응답 반환")
            return self._mock_llm_response()
        
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Ollama 서버 연결 실패: {self.llm_config.base_url}")
            raise ConnectionError(f"Ollama 서버에 연결할 수 없습니다: {self.llm_config.base_url}")
        
        except Exception as e:
            self.logger.error(f"LLM API 호출 실패: {str(e)}")
            raise
    
    def _extract_json(self, text: str) -> Optional[str]:
        """텍스트에서 유효한 JSON 추출 (괄호 매칭 방식)"""
        # { 위치 찾기
        start = text.find('{')
        if start == -1:
            return None
        
        # 괄호 매칭으로 끝 찾기
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
        """삼중 따옴표를 유효한 JSON 문자열로 변환
        
        LLM이 가끔 Python 스타일의 삼중 따옴표를 사용할 때 복구
        예: "content": \"\"\"code\"\"\" → "content": "code"
        """
        import re
        
        # 패턴: "key": """content""" 또는 "key": '''content'''
        # 삼중 따옴표 내용을 캡처하고 일반 문자열로 변환
        
        def replace_triple_quotes(match):
            content = match.group(1)
            # 내부 줄바꿈을 \n으로 변환
            content = content.replace('\n', '\\n')
            content = content.replace('\r', '\\r')
            content = content.replace('\t', '\\t')
            # 내부 따옴표 이스케이프
            content = content.replace('"', '\\"')
            return f'"{content}"'
        
        # """ ... """ 패턴 처리
        text = re.sub(r'"""(.*?)"""', replace_triple_quotes, text, flags=re.DOTALL)
        
        # ''' ... ''' 패턴 처리  
        text = re.sub(r"'''(.*?)'''", replace_triple_quotes, text, flags=re.DOTALL)
        
        return text
    
    def _sanitize_json_string(self, text: str) -> str:
        """JSON 문자열 정리 (삼중 따옴표 + 제어 문자)"""
        
        # 1. 삼중 따옴표 먼저 처리
        text = self._fix_triple_quotes(text)
        
        # 2. 제어 문자 이스케이프
        result = []
        in_string = False
        escape = False
        i = 0
        
        while i < len(text):
            char = text[i]
            
            if escape:
                result.append(char)
                escape = False
                i += 1
                continue
            
            if char == '\\':
                result.append(char)
                escape = True
                i += 1
                continue
            
            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue
            
            # 문자열 내부의 제어 문자 처리
            if in_string:
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif ord(char) < 32:  # 기타 제어 문자
                    result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    def _parse_llm_response(self, raw_response: str) -> LLMResponse:
        """LLM 응답 파싱"""
        
        try:
            # 1. 제어 문자 이스케이프 (먼저!)
            sanitized_response = self._sanitize_json_string(raw_response)
            
            # 2. JSON 추출 (괄호 매칭 방식)
            json_str = self._extract_json(sanitized_response)
            
            if not json_str:
                # JSON을 찾을 수 없으면 전체를 final answer로 처리
                return LLMResponse(
                    content=raw_response,
                    raw_response=raw_response
                )
            
            # 3. JSON 파싱
            data = json.loads(json_str)
            
            # tool_calls 파싱
            tool_calls = None
            if data.get("tool_calls"):
                tool_calls = []
                for tc in data["tool_calls"]:
                    tool_calls.append(ToolCall(
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", {})
                    ))
            
            return LLMResponse(
                thought=data.get("thought"),
                tool_calls=tool_calls,
                content=data.get("content"),
                raw_response=raw_response
            )
            
        except json.JSONDecodeError as e:
            self.logger.warn(f"JSON 파싱 실패: {str(e)}")
            # 파싱 실패 시 전체를 final answer로 처리
            return LLMResponse(
                content=raw_response,
                raw_response=raw_response
            )
    
    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Tool 실행"""
        
        self.logger.info(f"Tool 실행: {tool_call.name}.{tool_call.action}", {
            "params": tool_call.params
        })
        
        result = self.tools.execute(
            tool_name=tool_call.name,
            action=tool_call.action,
            params=tool_call.params
        )
        
        if result.success:
            self.logger.info(f"Tool 실행 성공: {tool_call.name}.{tool_call.action}")
        else:
            self.logger.error(f"Tool 실행 실패: {result.error}")
        
        return result
    
    def _generate_partial_response(self) -> str:
        """최대 step 도달 시 부분 응답 생성"""
        results = self.storage.get_results()
        
        if not results:
            return "작업을 완료하지 못했습니다. 다시 시도해주세요."
        
        # 마지막 결과 기반으로 응답 생성
        last_result = results[-1]
        return f"""최대 실행 단계({self.max_steps})에 도달했습니다.

지금까지의 진행 상황:
- 총 {len(results)}개의 작업 수행
- 마지막 작업: {last_result['executor']}.{last_result['action']}

부분 결과:
{str(last_result.get('output', ''))[:500]}

더 자세한 분석이 필요하면 질문을 나눠서 다시 시도해주세요."""
    
    def _mock_llm_response(self) -> str:
        """테스트용 Mock 응답"""
        results = self.storage.get_results()
        
        if not results:
            # 첫 번째 step - file_tool 호출
            return json.dumps({
                "thought": "파일을 먼저 읽어야 합니다 (Mock)",
                "tool_calls": [{
                    "name": "file_tool",
                    "arguments": {
                        "action": "read",
                        "path": "test.txt"
                    }
                }]
            })
        else:
            # 이후 step - final answer
            return json.dumps({
                "thought": "충분한 정보를 얻었습니다 (Mock)",
                "tool_calls": None,
                "content": f"Mock 분석 완료. {len(results)}개의 step을 수행했습니다."
            })