"""
Prompt Builder Module
=====================
프롬프트 구성을 위한 Fluent Interface 빌더.

공통 컴포넌트를 재사용하여 일관성 있는 프롬프트 생성을 지원합니다.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class PromptSection:
    """프롬프트 섹션"""
    name: str
    content: str
    order: int = 0


class PromptBuilder:
    """
    Fluent Interface 프롬프트 빌더

    사용 예시:
        builder = PromptBuilder()
        system_prompt, user_prompt = (
            builder
            .add_persona("expert code reviewer", ["quality", "security"])
            .add_reference_system()
            .add_response_format("json", {"thought": "...", "result": "..."})
            .build_both(user_query="Review this code")
        )
    """

    def __init__(self):
        self.system_sections: List[PromptSection] = []
        self.user_sections: List[PromptSection] = []
        self.variables: Dict[str, Any] = {}

    # =========================================================================
    # Basic Component Methods (재사용 가능한 공통 컴포넌트)
    # =========================================================================

    def add_persona(self, role: str, expertise_areas: Optional[List[str]] = None) -> 'PromptBuilder':
        """
        Persona 정의 추가

        Args:
            role: 역할 (예: "expert code reviewer", "software architect")
            expertise_areas: 전문 분야 리스트 (선택)

        Returns:
            self (method chaining)
        """
        content = f"You are {self._add_article(role)}."

        if expertise_areas:
            content += "\n" + self._format_expertise_areas(expertise_areas)

        self.system_sections.append(PromptSection("persona", content, order=0))
        return self

    def add_task_description(self, task: str, details: Optional[str] = None) -> 'PromptBuilder':
        """
        Task 설명 추가

        Args:
            task: 수행할 작업 설명
            details: 추가 세부사항 (선택)

        Returns:
            self
        """
        content = task
        if details:
            content += f"\n\n{details}"

        self.system_sections.append(PromptSection("task", content, order=10))
        return self

    def add_reference_system(self) -> 'PromptBuilder':
        """
        Reference System 문서 추가 ([FILE:*], [RESULT:*])

        Returns:
            self
        """
        content = """## Reference System
You can use these references in parameter values:

**[FILE:path]** - Reference file content directly (RECOMMENDED)
- Automatically replaced with file content from workspace/files/
- Example: {"content": "[FILE:example.py]"} → file content is loaded
- **PREFER this over file_tool.read** - it's more efficient and reduces plan steps
- Use when: You need file content in any parameter
- Common use cases:
  - Code analysis: {"content": "[FILE:main.c]"}
  - Code modification: {"prompt": "Refactor [FILE:utils.py]"}
  - Documentation: {"content": "[FILE:api.ts]"}

**[RESULT:session_step]** - Reference previous step output
- Example: [RESULT:Session0_1] → output from Step 1 in Session0
- Use when: Reusing output from earlier steps in the current plan
- Common use cases:
  - Pass analysis results to next step
  - Chain multiple transformations
  - Reference intermediate results

**Important:** When modifying a file, reference the file directly in the instruction/prompt parameter, not in a separate content parameter.
- Good: {"prompt": "Fix bugs in [FILE:app.py]"}
- Bad: {"content": "[RESULT:Session0_1]", "prompt": "Fix bugs"}"""

        self.system_sections.append(PromptSection("reference_system", content, order=20))
        return self

    def add_tool_schemas(self, schemas: Dict[str, Dict]) -> 'PromptBuilder':
        """
        Tool schemas 추가

        Args:
            schemas: Tool name -> schema dict 매핑

        Returns:
            self
        """
        content = "## Available Tools\n\n"

        for tool_name, tool_schema in schemas.items():
            content += f"### {tool_name}\n"
            content += f"{tool_schema.get('description', '')}\n\n"

            actions = tool_schema.get('actions', {})
            for action_name, action_schema in actions.items():
                content += f"**{action_name}**\n"
                content += f"- Description: {action_schema.get('description', '')}\n"

                params = action_schema.get('parameters', {}).get('properties', {})
                required = action_schema.get('parameters', {}).get('required', [])

                if params:
                    content += "- Parameters:\n"
                    for param_name, param_info in params.items():
                        req = " (required)" if param_name in required else " (optional)"
                        content += f"  - `{param_name}`: {param_info.get('description', '')}{req}\n"

                output_info = action_schema.get('output', {})
                content += f"- Output: {output_info.get('description', '')}\n\n"

        self.system_sections.append(PromptSection("tool_schemas", content, order=15))
        return self

    def add_action_schema(self, tool_name: str, action_name: str, schema: Dict) -> 'PromptBuilder':
        """
        단일 Action schema 추가 (실행 단계용)

        Args:
            tool_name: Tool 이름
            action_name: Action 이름
            schema: Action schema dict

        Returns:
            self
        """
        content = f"## Current Action: {tool_name}.{action_name}\n\n"
        content += f"**Description:** {schema.get('description', '')}\n\n"

        params = schema.get('parameters', {}).get('properties', {})
        required = schema.get('parameters', {}).get('required', [])

        if params:
            content += "**Parameters:**\n"
            for param_name, param_info in params.items():
                req = " **(required)**" if param_name in required else " (optional)"
                content += f"- `{param_name}` ({param_info.get('type', 'string')}){req}: {param_info.get('description', '')}\n"

                if 'default' in param_info:
                    content += f"  - Default: `{param_info['default']}`\n"

        output_info = schema.get('output', {})
        content += f"\n**Output:** {output_info.get('description', '')}\n"

        self.system_sections.append(PromptSection("action_schema", content, order=15))
        return self

    def add_execution_context(self, context: str, max_preview: int = 300) -> 'PromptBuilder':
        """
        실행 컨텍스트 추가 (이전 결과 요약 등)

        Args:
            context: 컨텍스트 문자열
            max_preview: 최대 미리보기 길이

        Returns:
            self
        """
        if len(context) > max_preview:
            preview = context[:max_preview] + f"... ({len(context)} chars total)"
        else:
            preview = context

        content = f"## Execution Context\n\n{preview}"
        self.user_sections.append(PromptSection("execution_context", content, order=10))
        return self

    def add_response_format(self, format_type: str, template: Optional[Dict] = None,
                           description: Optional[str] = None) -> 'PromptBuilder':
        """
        Response 형식 지정 추가

        Args:
            format_type: "json" 또는 "natural_language"
            template: JSON 템플릿 (format_type="json"일 때)
            description: 추가 설명

        Returns:
            self
        """
        content = "## Response Format\n\n"

        if format_type == "json":
            content += "Respond with valid JSON only.\n\n"
            if template:
                import json
                content += "Format:\n"
                content += json.dumps(template, indent=2, ensure_ascii=False)
                content += "\n"
        elif format_type == "natural_language":
            content += "Respond in natural language (Korean/한글).\n"

        if description:
            content += f"\n{description}"

        self.system_sections.append(PromptSection("response_format", content, order=30))
        return self

    def add_rules(self, rules: List[str]) -> 'PromptBuilder':
        """
        Rules/Constraints 추가

        Args:
            rules: 규칙 리스트

        Returns:
            self
        """
        content = "## Rules\n"
        for rule in rules:
            content += f"- {rule}\n"

        self.system_sections.append(PromptSection("rules", content, order=40))
        return self

    def add_constraint(self, name: str, value: Any) -> 'PromptBuilder':
        """
        단일 constraint 추가

        Args:
            name: Constraint 이름
            value: Constraint 값

        Returns:
            self
        """
        self.variables[name] = value
        return self

    def add_custom_section(self, name: str, content: str, order: int = 50,
                          section_type: str = "system") -> 'PromptBuilder':
        """
        커스텀 섹션 추가

        Args:
            name: 섹션 이름
            content: 섹션 내용
            order: 정렬 순서
            section_type: "system" 또는 "user"

        Returns:
            self
        """
        section = PromptSection(name, content, order)

        if section_type == "system":
            self.system_sections.append(section)
        else:
            self.user_sections.append(section)

        return self

    # =========================================================================
    # Build Methods
    # =========================================================================

    def build_system_prompt(self) -> str:
        """System prompt 생성"""
        sorted_sections = sorted(self.system_sections, key=lambda s: s.order)
        parts = [section.content for section in sorted_sections]
        return "\n\n".join(parts)

    def build_user_prompt(self, **variables) -> str:
        """
        User prompt 생성

        Args:
            **variables: 템플릿 변수 (user_query 등)

        Returns:
            User prompt 문자열
        """
        sorted_sections = sorted(self.user_sections, key=lambda s: s.order)
        parts = [section.content for section in sorted_sections]

        # variables 추가
        if 'user_query' in variables:
            parts.insert(0, variables['user_query'])

        prompt = "\n\n".join(parts)

        # 템플릿 변수 치환
        for key, value in {**self.variables, **variables}.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))

        return prompt

    def build_both(self, **variables) -> Tuple[str, str]:
        """
        System prompt와 User prompt를 모두 생성

        Args:
            **variables: 템플릿 변수

        Returns:
            (system_prompt, user_prompt) 튜플
        """
        return self.build_system_prompt(), self.build_user_prompt(**variables)

    def build(self, **variables) -> Tuple[str, str]:
        """build_both의 별칭"""
        return self.build_both(**variables)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _add_article(self, role: str) -> str:
        """역할 앞에 관사 추가"""
        if role.startswith(('a ', 'an ', 'the ')):
            return role

        # 모음으로 시작하면 'an', 자음이면 'a'
        first_char = role[0].lower()
        if first_char in 'aeiou':
            return f"an {role}"
        return f"a {role}"

    def _format_expertise_areas(self, areas: List[str]) -> str:
        """전문 분야 리스트 포맷팅"""
        if not areas:
            return ""

        if len(areas) == 1:
            return f"Specializing in {areas[0]}."

        formatted = "Specializing in:\n"
        for area in areas:
            formatted += f"- {area}\n"

        return formatted.rstrip()

    # =========================================================================
    # Specialized Builders for Orchestrator
    # =========================================================================

    def for_query_classification(self, tools_list: str) -> 'PromptBuilder':
        """
        Query classification prompt 생성

        Args:
            tools_list: 사용 가능한 도구 목록 문자열

        Returns:
            self
        """
        self.add_persona("query classifier")
        self.add_task_description(
            "Determine if the user query requires tool usage or can be answered directly."
        )

        # Available tools section
        self.add_custom_section(
            "available_tools",
            f"## Available Tools\n{tools_list}",
            order=15
        )

        # Response format
        self.add_response_format(
            "json",
            {
                "needs_tools": True,
                "reasoning": "Brief explanation of why tools are/aren't needed"
            }
        )

        # Rules
        self.add_rules([
            "Analyze the query carefully",
            "If the query asks about code, files, or requires execution → needs_tools: true",
            "If the query is a simple question that can be answered directly → needs_tools: false",
            "Respond with valid JSON only"
        ])

        return self

    def for_planning(self, tool_schemas: Dict[str, Dict], max_steps: int,
                    needs_tools: bool = True, files_context: Optional[str] = None) -> 'PromptBuilder':
        """
        Planning prompt 생성

        Args:
            tool_schemas: 도구 스키마 딕셔너리
            max_steps: 최대 스텝 수
            needs_tools: 도구 사용 필요 여부
            files_context: 파일 컨텍스트 (선택)

        Returns:
            self
        """
        self.add_persona("task planner and problem solver")

        if needs_tools:
            # With tools path
            self.add_task_description(
                "Create a step-by-step execution plan to solve the user's query.",
                "Analyze the query and break it down into concrete, executable steps using available tools."
            )

            self.add_tool_schemas(tool_schemas)
            self.add_reference_system()

            # Files context (optional)
            if files_context:
                self.add_custom_section(
                    "files_context",
                    f"## Available Files\n{files_context}",
                    order=16
                )

            # Plan optimization
            self.add_custom_section(
                "plan_optimization",
                """## Plan Optimization
**Minimize steps by using references ([FILE:*], [RESULT:*]) instead of intermediate operations.**

Good (1 step with FILE reference):
{
    "step": 1,
    "tool_name": "llm_tool",
    "action": "staticanalysis",
    "params_hint": {"content": "[FILE:example.py]"}
}

Bad (2 steps - unnecessary read step):
{
    "step": 1,
    "tool_name": "file_tool",
    "action": "read",
    "params_hint": {"path": "example.py"}
},
{
    "step": 2,
    "tool_name": "llm_tool",
    "action": "staticanalysis",
    "params_hint": {"content": "[RESULT:Session0_1]"}
}""",
                order=25
            )

            # Response format
            self.add_response_format(
                "json",
                {
                    "plan": [
                        {
                            "step": 1,
                            "tool_name": "tool_name",
                            "action": "action_name",
                            "description": "What this step does",
                            "params_hint": {
                                "param1": "value or [FILE:path] or [RESULT:Session0_X]"
                            }
                        }
                    ]
                }
            )

            # Rules
            self.add_rules([
                f"Maximum {max_steps} steps",
                "Use references ([FILE:*], [RESULT:*]) to minimize steps",
                "Each step must have params_hint with concrete values or references",
                "Respond with valid JSON only"
            ])

        else:
            # No tools path - direct answer
            self.add_task_description(
                "Provide a direct answer to the user's query.",
                "The query doesn't require tool usage - answer it directly based on your knowledge."
            )

            self.add_response_format(
                "json",
                {
                    "direct_answer": "Your answer (in Korean/한글)"
                }
            )

            self.add_rules([
                "Provide clear, concise answer in Korean",
                "Respond with valid JSON only"
            ])

        self.add_constraint("max_steps", max_steps)
        return self

    def for_execution(self, tool_name: str, action: str, action_schema: Dict,
                     step_num: int, params_hint: Optional[Dict] = None,
                     execution_context: Optional[str] = None) -> 'PromptBuilder':
        """
        Step execution prompt 생성

        Args:
            tool_name: Tool 이름
            action: Action 이름
            action_schema: Action 스키마
            step_num: 현재 스텝 번호
            params_hint: 파라미터 힌트 (선택)
            execution_context: 실행 컨텍스트 (선택)

        Returns:
            self
        """
        self.add_persona("step executor")
        self.add_task_description(
            f"Execute Step {step_num} by determining exact parameter values.",
            f"You are executing: {tool_name}.{action}"
        )

        self.add_action_schema(tool_name, action, action_schema)
        self.add_reference_system()

        if execution_context:
            self.add_execution_context(execution_context)

        # Params hint section
        if params_hint:
            import json
            params_str = json.dumps(params_hint, indent=2, ensure_ascii=False)
            self.add_custom_section(
                "params_hint",
                f"## Suggested Parameters\n\n{params_str}\n",
                order=18,
                section_type="user"
            )

        # Response format
        self.add_response_format(
            "json",
            {
                "thought": "Brief reasoning about parameter values",
                "tool_calls": [
                    {
                        "name": tool_name,
                        "arguments": {
                            "action": action,
                            "param1": "concrete value or reference"
                        }
                    }
                ]
            }
        )

        # Rules
        self.add_rules([
            f"Execute ONLY Step {step_num}",
            "You MUST generate exactly ONE tool_call",
            f"Use parameter names with action prefix (e.g., {action}_param)",
            "Use references ([FILE:*], [RESULT:*]) in parameter values when appropriate",
            "For content/prompt parameters: prefer [FILE:path] over [RESULT:*] when modifying files",
            "All required parameters must be provided",
            "Respond with valid JSON only"
        ])

        self.add_constraint("step_num", step_num)
        return self

    def for_retry(self, tool_name: str, action: str, action_schema: Dict,
                 error_message: str, failed_params: Dict,
                 previous_results: Optional[str] = None) -> 'PromptBuilder':
        """
        Parameter correction (retry) prompt 생성

        Args:
            tool_name: Tool 이름
            action: Action 이름
            action_schema: Action 스키마
            error_message: 에러 메시지
            failed_params: 실패한 파라미터
            previous_results: 이전 결과 (선택)

        Returns:
            self
        """
        self.add_persona("parameter correction specialist")
        self.add_task_description(
            "Analyze the error and determine if parameters can be corrected.",
            "If the error is fixable, provide corrected parameters. If not, return empty tool_calls."
        )

        self.add_action_schema(tool_name, action, action_schema)

        # Error context
        import json
        failed_params_str = json.dumps(failed_params, indent=2, ensure_ascii=False)
        error_section = f"""## Error Context

**Error Message:**
{error_message}

**Failed Parameters:**
{failed_params_str}
"""

        self.add_custom_section("error_context", error_section, order=16)

        if previous_results:
            self.add_custom_section(
                "previous_results",
                f"## Available Previous Results\n{previous_results}",
                order=17
            )

        # Fixability decision
        self.add_custom_section(
            "fixability_decision",
            """## Fixability Decision

**Fixable errors (provide corrected parameters):**
- Wrong reference format → correct to [RESULT:SessionX_Y]
- Missing required parameter → add it
- Wrong parameter value → fix it
- Path not found → try alternative path

**Unfixable errors (return empty tool_calls):**
- File genuinely doesn't exist and no alternatives
- Required data not available in previous results
- Tool/action fundamentally cannot handle the task""",
            order=18
        )

        # Response format
        self.add_response_format(
            "json",
            {
                "thought": "Analysis of whether error is fixable",
                "tool_calls": [
                    {
                        "name": tool_name,
                        "arguments": {
                            "action": action,
                            "param1": "corrected value"
                        }
                    }
                ]
            },
            "Note: If unfixable, return empty array for tool_calls: []"
        )

        self.add_rules([
            "Analyze error message carefully",
            "Only provide corrected parameters if error is truly fixable",
            "If unfixable, return tool_calls: []",
            "Respond with valid JSON only"
        ])

        return self

    def for_final_answer(self, user_query: str, plan_summary: str,
                        results_summary: str) -> 'PromptBuilder':
        """
        Final answer generation prompt 생성

        Args:
            user_query: 원래 사용자 질의
            plan_summary: 실행 계획 요약
            results_summary: 실행 결과 요약

        Returns:
            self
        """
        self.add_persona("result synthesizer")
        self.add_task_description(
            "Synthesize the execution results into a comprehensive final answer.",
            "Review the execution plan and results, then provide a clear answer to the user's original query."
        )

        # User prompt sections
        self.user_sections.append(
            PromptSection("original_query", f"## Original Query\n{user_query}", order=0)
        )
        self.user_sections.append(
            PromptSection("plan_summary", f"## Execution Plan\n{plan_summary}", order=10)
        )
        self.user_sections.append(
            PromptSection("results_summary", f"## Execution Results\n{results_summary}", order=20)
        )

        # Response format
        self.add_response_format("natural_language")

        # Rules
        self.add_rules([
            "Summarize the results in clear, concise Korean (한글)",
            "If any step failed, mention it briefly but focus on what was accomplished",
            "Provide actionable information based on the results",
            "Be conversational and helpful"
        ])

        return self
