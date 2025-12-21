"""
LLMTool Module
==============
LLM에게 분석, 요약, 변환 등의 작업을 요청하는 Tool.

핵심 기능:
- 이전 Step 결과를 자동으로 컨텍스트에 포함
- Orchestrator와 동일한 LLM 설정 사용
- 다양한 분석 작업 지원 (정적 분석, 요약, 변환 등)
"""

import json
from typing import Dict, Optional, Callable

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_tool import BaseTool, ActionSchema, ActionParam, ToolResult


class LLMTool(BaseTool):
    """LLM 호출 Tool - 분석, 요약, 변환 등"""
    
    name = "llm_tool"
    description = "LLM에게 텍스트 분석, 요약, 코드 리뷰, 변환 등의 작업을 요청"
    
    def __init__(
        self, 
        debug_enabled: bool = True, 
        use_mock: bool = False,
        llm_caller: Optional[Callable[[str, str], str]] = None,
        get_previous_result: Optional[Callable[[], str]] = None
    ):
        """
        Args:
            debug_enabled: 디버그 로깅 활성화
            use_mock: True면 Mock 응답 사용
            llm_caller: LLM API 호출 함수 (system_prompt, user_prompt) -> response
            get_previous_result: 이전 Step 결과를 가져오는 함수
        """
        self.use_mock = use_mock
        self._llm_caller = llm_caller
        self._get_previous_result = get_previous_result
        super().__init__(debug_enabled=debug_enabled)
        
        if use_mock:
            self.logger.warn("Mock 모드로 실행됩니다.")
    
    def set_llm_caller(self, caller: Callable[[str, str], str]):
        """LLM 호출 함수 설정 (Orchestrator에서 주입)"""
        self._llm_caller = caller
        self.logger.debug("LLM caller 설정됨")
    
    def set_previous_result_getter(self, getter: Callable[[], str]):
        """이전 결과 getter 설정 (Orchestrator에서 주입)"""
        self._get_previous_result = getter
        self.logger.debug("Previous result getter 설정됨")
    
    def _define_actions(self) -> Dict[str, ActionSchema]:
        return {
            "ask": ActionSchema(
                name="ask",
                description="LLM에게 자유 형식 질문/요청. 이전 Step 결과를 자동으로 컨텍스트에 포함.",
                params=[
                    ActionParam("ask_prompt", "str", True, "질문 또는 요청 내용"),
                    ActionParam("ask_include_previous", "bool", False, 
                               "이전 Step 결과를 컨텍스트에 포함할지 여부", True),
                ],
                output_type="str",
                output_description="LLM 응답"
            ),
            "analyze": ActionSchema(
                name="analyze",
                description="코드/텍스트 분석 요청. 정적 분석, 버그 탐지, 코드 리뷰 등.",
                params=[
                    ActionParam("analyze_content", "str", False, 
                               "분석할 내용 (생략 시 이전 Step 결과 사용)", None),
                    ActionParam("analyze_type", "str", False, 
                               "분석 유형: static, security, performance, style, general", "general"),
                    ActionParam("analyze_instruction", "str", False, 
                               "추가 분석 지시사항", None),
                ],
                output_type="str",
                output_description="분석 결과 (마크다운 형식)"
            ),
            "summarize": ActionSchema(
                name="summarize",
                description="텍스트 요약",
                params=[
                    ActionParam("summarize_content", "str", False, 
                               "요약할 내용 (생략 시 이전 Step 결과 사용)", None),
                    ActionParam("summarize_max_length", "int", False, 
                               "최대 요약 길이 (단어 수)", 200),
                    ActionParam("summarize_style", "str", False,
                               "요약 스타일: brief, detailed, bullet", "detailed"),
                ],
                output_type="str",
                output_description="요약된 텍스트"
            ),
            "transform": ActionSchema(
                name="transform",
                description="텍스트/코드 변환 (포맷 변경, 언어 번역, 코드 변환 등)",
                params=[
                    ActionParam("transform_content", "str", False, 
                               "변환할 내용 (생략 시 이전 Step 결과 사용)", None),
                    ActionParam("transform_instruction", "str", True, 
                               "변환 지시사항 (예: 'Python으로 변환', 'JSON으로 변환')"),
                ],
                output_type="str",
                output_description="변환된 결과"
            ),
            "extract": ActionSchema(
                name="extract",
                description="텍스트에서 특정 정보 추출",
                params=[
                    ActionParam("extract_content", "str", False, 
                               "추출할 대상 (생략 시 이전 Step 결과 사용)", None),
                    ActionParam("extract_what", "str", True, 
                               "추출할 정보 (예: 'errors', 'function names', 'TODOs')"),
                    ActionParam("extract_format", "str", False,
                               "출력 형식: list, json, markdown", "markdown"),
                ],
                output_type="str",
                output_description="추출된 정보"
            ),
        }
    
    def _execute_action(self, action: str, params: Dict) -> ToolResult:
        """실제 action 실행"""
        def get_param(name: str, default=None):
            prefixed = f"{action}_{name}"
            return params.get(prefixed, default)
        
        if action == "ask":
            return self._ask(
                get_param("prompt"),
                get_param("include_previous", True)
            )
        elif action == "analyze":
            return self._analyze(
                get_param("content"),
                get_param("type", "general"),
                get_param("instruction")
            )
        elif action == "summarize":
            return self._summarize(
                get_param("content"),
                get_param("max_length", 200),
                get_param("style", "detailed")
            )
        elif action == "transform":
            return self._transform(
                get_param("content"),
                get_param("instruction")
            )
        elif action == "extract":
            return self._extract(
                get_param("content"),
                get_param("what"),
                get_param("format", "markdown")
            )
        else:
            return ToolResult.error_result(f"Unknown action: {action}")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _get_content(self, provided_content: Optional[str]) -> str:
        """컨텐츠 가져오기 - 제공되지 않으면 이전 Step 결과 사용"""
        if provided_content:
            return provided_content
        
        if self._get_previous_result:
            previous = self._get_previous_result()
            if previous:
                self.logger.debug("이전 Step 결과를 컨텐츠로 사용")
                return str(previous)
        
        return ""
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """LLM 호출"""
        if self.use_mock:
            return self._mock_response(user_prompt)
        
        if self._llm_caller is None:
            self.logger.warn("LLM caller가 설정되지 않음. Mock 응답 반환.")
            return self._mock_response(user_prompt)
        
        try:
            response = self._llm_caller(system_prompt, user_prompt)
            return response
        except Exception as e:
            self.logger.error(f"LLM 호출 실패: {e}")
            raise
    
    def _mock_response(self, prompt: str) -> str:
        """Mock 응답 생성"""
        return f"""## Mock LLM Response

**요청**: {prompt[:100]}...

이것은 테스트용 Mock 응답입니다. 실제 LLM 연결 시 실제 분석 결과가 제공됩니다.

### 샘플 분석 결과
- 항목 1: Mock 데이터
- 항목 2: Mock 데이터  
- 항목 3: Mock 데이터
"""
    
    # =========================================================================
    # Action Implementations
    # =========================================================================
    
    def _ask(self, prompt: str, include_previous: bool) -> ToolResult:
        """자유 형식 질문"""
        context = ""
        if include_previous and self._get_previous_result:
            previous = self._get_previous_result()
            if previous:
                context = f"\n\n## 이전 Step 결과 (컨텍스트)\n```\n{previous}\n```"
        
        system_prompt = """You are a helpful assistant. Answer the user's question clearly and concisely.
If context from a previous step is provided, use it to inform your answer."""

        user_prompt = f"{prompt}{context}"
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={"action": "ask", "prompt_length": len(prompt)}
            )
        except Exception as e:
            return ToolResult.error_result(f"LLM 호출 실패: {str(e)}")
    
    def _analyze(self, content: Optional[str], analysis_type: str, 
                 instruction: Optional[str]) -> ToolResult:
        """코드/텍스트 분석"""
        content = self._get_content(content)
        if not content:
            return ToolResult.error_result("분석할 내용이 없습니다. content를 제공하거나 이전 Step에서 결과가 있어야 합니다.")
        
        # 분석 유형별 시스템 프롬프트
        type_prompts = {
            "static": """You are a static code analyzer. Analyze the code for:
- Potential bugs and errors
- Memory issues (leaks, buffer overflows, use-after-free)
- Uninitialized variables
- Dead code
- Logic errors
Provide specific line numbers and severity levels.""",
            
            "security": """You are a security code reviewer. Analyze for:
- Buffer overflows
- SQL injection vulnerabilities  
- XSS vulnerabilities
- Authentication/authorization issues
- Sensitive data exposure
Rate each issue by severity (Critical, High, Medium, Low).""",
            
            "performance": """You are a performance analyst. Analyze for:
- Algorithm complexity issues
- Unnecessary memory allocations
- Inefficient loops
- Cache-unfriendly patterns
- I/O bottlenecks
Suggest specific optimizations.""",
            
            "style": """You are a code style reviewer. Check for:
- Naming conventions
- Code formatting
- Documentation/comments
- Code organization
- Best practices adherence
Provide actionable suggestions.""",
            
            "general": """You are a code reviewer. Provide a comprehensive analysis including:
- Code quality assessment
- Potential bugs or issues
- Suggestions for improvement
- Best practices recommendations
Be specific and provide examples."""
        }
        
        system_prompt = type_prompts.get(analysis_type, type_prompts["general"])
        
        if instruction:
            system_prompt += f"\n\nAdditional instructions: {instruction}"
        
        system_prompt += "\n\nFormat your response in Markdown with clear sections and code examples."
        
        user_prompt = f"Please analyze the following:\n\n```\n{content}\n```"
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "analyze",
                    "type": analysis_type,
                    "content_length": len(content)
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"분석 실패: {str(e)}")
    
    def _summarize(self, content: Optional[str], max_length: int, style: str) -> ToolResult:
        """텍스트 요약"""
        content = self._get_content(content)
        if not content:
            return ToolResult.error_result("요약할 내용이 없습니다.")
        
        style_instructions = {
            "brief": "Provide a very concise summary in 1-2 sentences.",
            "detailed": "Provide a comprehensive summary covering all key points.",
            "bullet": "Provide the summary as bullet points."
        }
        
        system_prompt = f"""You are a summarization expert.
{style_instructions.get(style, style_instructions['detailed'])}
Keep the summary under {max_length} words."""

        user_prompt = f"Please summarize:\n\n{content}"
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "summarize",
                    "style": style,
                    "original_length": len(content)
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"요약 실패: {str(e)}")
    
    def _transform(self, content: Optional[str], instruction: str) -> ToolResult:
        """텍스트/코드 변환"""
        content = self._get_content(content)
        if not content:
            return ToolResult.error_result("변환할 내용이 없습니다.")
        
        system_prompt = f"""You are a transformation assistant.
Follow the user's transformation instruction precisely.
Instruction: {instruction}

Return only the transformed result without explanations unless asked."""

        user_prompt = f"Transform the following:\n\n```\n{content}\n```"
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "transform",
                    "instruction": instruction
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"변환 실패: {str(e)}")
    
    def _extract(self, content: Optional[str], what: str, format: str) -> ToolResult:
        """정보 추출"""
        content = self._get_content(content)
        if not content:
            return ToolResult.error_result("추출할 대상이 없습니다.")
        
        format_instructions = {
            "list": "Return as a simple list, one item per line.",
            "json": "Return as valid JSON array or object.",
            "markdown": "Return as formatted Markdown with headers and lists."
        }
        
        system_prompt = f"""You are an information extraction specialist.
Extract: {what}
{format_instructions.get(format, format_instructions['markdown'])}
Be thorough and accurate."""

        user_prompt = f"Extract from the following:\n\n```\n{content}\n```"
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "extract",
                    "what": what,
                    "format": format
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"추출 실패: {str(e)}")
