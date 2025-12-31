"""
LLMTool Module
==============
Comprehensive LLM-based tool for code analysis, generation, and general queries.

Core Features:
- Code Review: Comprehensive code quality and best practices review
- Architecture Analysis: System design and architectural pattern analysis
- Code Documentation: Generate documentation, docstrings, and comments
- Static Analysis: Bug detection, security vulnerabilities, code quality issues
- Code Writer: Generate new code or modify existing code
- General Queries: Free-form questions and tasks

Uses the same LLM configuration as the Orchestrator.
"""

from typing import Dict, Optional, Callable

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_tool import BaseTool, ActionSchema, ActionParam, ToolResult


class LLMTool(BaseTool):
    """LLM-based tool for code review, analysis, documentation, generation, and general queries"""

    name = "llm_tool"
    description = "Perform code review, architecture analysis, documentation generation, static analysis, code writing, and general LLM queries"

    def __init__(
        self,
        base_path: str,
        debug_enabled: bool = True,
        use_mock: bool = False,
        llm_caller: Optional[Callable[[str, str], str]] = None,
        get_previous_result: Optional[Callable[[], str]] = None
    ):
        """
        Initialize LLMTool

        Args:
            base_path: Base directory path for file operations (files can only be accessed within this path for security)
            debug_enabled: Enable debug logging
            use_mock: If True, use mock responses instead of actual LLM calls
            llm_caller: LLM API calling function with signature (system_prompt, user_prompt) -> response
            get_previous_result: Function to retrieve previous step results (currently unused)
        """
        self.use_mock = use_mock
        self._llm_caller = llm_caller
        self._get_previous_result = get_previous_result
        super().__init__(base_path=base_path, debug_enabled=debug_enabled)

        if use_mock:
            self.logger.warn("Running in MOCK mode")
    
    def set_llm_caller(self, caller: Callable[[str, str], str]):
        """Set the LLM caller function (injected by Orchestrator)"""
        self._llm_caller = caller
        self.logger.debug("LLM caller configured")
    
    def _define_actions(self) -> Dict[str, ActionSchema]:
        return {
            "codereview": ActionSchema(
                name="codereview",
                description="Perform comprehensive code review with detailed feedback on code quality, best practices, potential bugs, and improvement suggestions. Returns analysis as text output (accessible via [RESULT:SessionX_Y]) - does NOT save to file. Use file_tool.write with [RESULT:SessionX_Y] to save if needed.",
                params=[
                    ActionParam("codereview_instruction", "str", True,
                               "Code review instructions or focus areas. Can be: string, [FILE:path], [RESULT:SessionX_Y], or mix of these. Use [FILE:path] for any file references (e.g., 'Focus on error handling in [FILE:utils.c]')"),
                ],
                output_type="str",
                output_description="Detailed code review report in Markdown format with specific recommendations"
            ),
            "architectureanalysis": ActionSchema(
                name="architectureanalysis",
                description="Analyze software architecture, design patterns, system structure, component relationships, and architectural trade-offs. Returns analysis as text output (accessible via [RESULT:SessionX_Y]) - does NOT save to file. Use file_tool.write with [RESULT:SessionX_Y] to save if needed.",
                params=[
                    ActionParam("architectureanalysis_instruction", "str", True,
                               "Architecture analysis instructions. Can be: string, [FILE:path], [RESULT:SessionX_Y], or mix of these. Use [FILE:path] for any file references (e.g., 'Evaluate scalability of [FILE:server.js]')"),
                ],
                output_type="str",
                output_description="Architecture analysis report with patterns identified and recommendations in Markdown format"
            ),
            "codedoc": ActionSchema(
                name="codedoc",
                description="Generate comprehensive documentation for code including docstrings, comments, API documentation, and usage examples. Returns documentation as text output (accessible via [RESULT:SessionX_Y]) - does NOT save to file. Use file_tool.write with [RESULT:SessionX_Y] to save if needed.",
                params=[
                    ActionParam("codedoc_instruction", "str", True,
                               "Code documentation instructions. Can be: string, [FILE:path], [RESULT:SessionX_Y], or mix of these. Use [FILE:path] for any file references (e.g., 'Generate API docs for [FILE:api.py]')"),
                ],
                output_type="str",
                output_description="Generated documentation in appropriate format (Markdown, docstrings, or comments)"
            ),
            "staticanalysis": ActionSchema(
                name="staticanalysis",
                description="Perform static code analysis to detect bugs, security vulnerabilities, code smells, type errors, and potential runtime issues without executing code. Returns analysis as text output (accessible via [RESULT:SessionX_Y]) - does NOT save to file. Use file_tool.write with [RESULT:SessionX_Y] to save if needed.",
                params=[
                    ActionParam("staticanalysis_instruction", "str", True,
                               "Static analysis instructions. Can be: string, [FILE:path], [RESULT:SessionX_Y], or mix of these. Use [FILE:path] for any file references (e.g., 'Check [FILE:auth.c] for security issues')"),
                ],
                output_type="str",
                output_description="Static analysis report with categorized issues, severity levels, and fix suggestions in Markdown format"
            ),
            "codewriter": ActionSchema(
                name="codewriter",
                description="Generate new code from scratch or modify existing code based on specifications, requirements, or improvement guidelines. Returns code as text output (accessible via [RESULT:SessionX_Y]) - does NOT save to file. Use file_tool.write with [RESULT:SessionX_Y] to save if needed.",
                params=[
                    ActionParam("codewriter_instruction", "str", True,
                               "Code writing instructions and specifications. Can be: string, [FILE:path], [RESULT:SessionX_Y], or mix of these. Use [FILE:path] for any file references (e.g., 'Add error handling to [FILE:app.js]', 'Refactor [FILE:utils.py] based on [RESULT:Session0_1]')"),
                ],
                output_type="str",
                output_description="Generated or modified code with explanatory comments"
            ),
            "general": ActionSchema(
                name="general",
                description="Handle general queries, questions, and tasks that don't fit into specialized categories. Free-form interaction with LLM for any other purpose. Returns response as text output (accessible via [RESULT:SessionX_Y]) - does NOT save to file. Use file_tool.write with [RESULT:SessionX_Y] to save if needed.",
                params=[
                    ActionParam("general_prompt", "str", True,
                               "Your question, request, or instruction for the LLM. Can be: string, [FILE:path], [RESULT:SessionX_Y], or mix of these. Use [FILE:path] for any file references (e.g., 'Modify [FILE:sample.c] based on [RESULT:Session0_1]')"),
                ],
                output_type="str",
                output_description="LLM response to the general query"
            ),
        }
    
    def _execute_action(self, action: str, params: Dict) -> ToolResult:
        """실제 action 실행"""
        def get_param(name: str, default=None):
            prefixed = f"{action}_{name}"
            return params.get(prefixed, default)

        if action == "codereview":
            return self._codereview(
                get_param("instruction")
            )
        elif action == "architectureanalysis":
            return self._architectureanalysis(
                get_param("instruction")
            )
        elif action == "codedoc":
            return self._codedoc(
                get_param("instruction")
            )
        elif action == "staticanalysis":
            return self._staticanalysis(
                get_param("instruction")
            )
        elif action == "codewriter":
            return self._codewriter(
                get_param("instruction")
            )
        elif action == "general":
            return self._general(
                get_param("prompt")
            )
        else:
            return ToolResult.error_result(f"Unknown action: {action}")
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM with system and user prompts"""
        if self.use_mock:
            return self._mock_response(user_prompt)

        if self._llm_caller is None:
            self.logger.warn("LLM caller not configured. Returning mock response.")
            return self._mock_response(user_prompt)

        try:
            response = self._llm_caller(system_prompt, user_prompt)
            return response
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing"""
        return f"""## Mock LLM Response

**Request**: {prompt[:100]}...

This is a test mock response. Actual analysis results will be provided when connected to a real LLM.

### Sample Analysis Results
- Item 1: Mock data
- Item 2: Mock data
- Item 3: Mock data
"""
    
    # =========================================================================
    # Action Implementations
    # =========================================================================

    def _codereview(self, instruction: Optional[str]) -> ToolResult:
        """Perform comprehensive code review"""

        system_prompt = """You are an expert code reviewer with years of experience in software engineering.
Provide a comprehensive code review covering:
- Code quality and readability
- Best practices adherence
- Potential bugs and edge cases
- Performance considerations
- Security concerns
- Maintainability and extensibility

Format your response in Markdown with clear sections.
Be specific and provide code examples for your suggestions."""

        user_prompt = instruction

        try:
            response = self._call_llm(system_prompt, user_prompt)
            summary = f"Code review completed ({len(response)} chars)"
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "codereview",
                    "instruction": instruction
                },
                short_summary=summary
            )
        except Exception as e:
            return ToolResult.error_result(f"Code review failed: {str(e)}")

    def _architectureanalysis(self, instruction: Optional[str]) -> ToolResult:
        """Analyze software architecture"""

        system_prompt = """You are a software architect specializing in system design and architecture analysis.
Analyze the architecture and provide insights on:
- Design patterns used and their appropriateness
- Component structure and relationships
- Separation of concerns
- Scalability considerations
- Coupling and cohesion
- Architectural trade-offs
- Potential improvements

Format your response in Markdown with diagrams where helpful."""

        user_prompt = instruction

        try:
            response = self._call_llm(system_prompt, user_prompt)
            summary = f"Architecture analysis completed ({len(response)} chars)"
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "architectureanalysis",
                    "instruction": instruction
                },
                short_summary=summary
            )
        except Exception as e:
            return ToolResult.error_result(f"Architecture analysis failed: {str(e)}")

    def _codedoc(self, instruction: Optional[str]) -> ToolResult:
        """Generate code documentation"""

        system_prompt = """You are a technical documentation specialist.
Generate comprehensive documentation including:
- Clear descriptions of functionality
- Parameter and return value documentation
- Usage examples
- Edge cases and caveats
- Related functions or modules

Follow language-specific documentation conventions (docstrings, JSDoc, etc.).
Format appropriately for the code's language and context."""

        user_prompt = instruction

        try:
            response = self._call_llm(system_prompt, user_prompt)
            summary = f"Documentation generated ({len(response)} chars)"
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "codedoc",
                    "instruction": instruction
                },
                short_summary=summary
            )
        except Exception as e:
            return ToolResult.error_result(f"Documentation generation failed: {str(e)}")

    def _staticanalysis(self, instruction: Optional[str]) -> ToolResult:
        """Perform static code analysis"""

        system_prompt = """You are a static code analyzer specializing in bug detection and code quality.
Analyze the code for:
- Potential bugs (null pointers, off-by-one errors, race conditions)
- Security vulnerabilities (injection, XSS, buffer overflows)
- Code smells and anti-patterns
- Type errors and mismatches
- Resource leaks (memory, file handles, connections)
- Dead code and unreachable statements

Categorize issues by severity (Critical, High, Medium, Low).
Provide specific line references and fix suggestions.
Format your response in Markdown."""

        user_prompt = instruction

        try:
            response = self._call_llm(system_prompt, user_prompt)
            summary = f"Static analysis completed ({len(response)} chars)"
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "staticanalysis",
                    "instruction": instruction
                },
                short_summary=summary
            )
        except Exception as e:
            return ToolResult.error_result(f"Static analysis failed: {str(e)}")

    def _codewriter(self, instruction: Optional[str]) -> ToolResult:
        """Generate or modify code"""

        # For codewriter, content is optional (new code generation)
        system_prompt = """You are an expert software engineer.
Write new code from scratch according to the specifications.
- Follow best practices and conventions
- Add clear, helpful comments
- Consider error handling and edge cases
- Write clean, readable, maintainable code
- Include usage examples if helpful"""

        user_prompt = instruction

        try:
            response = self._call_llm(system_prompt, user_prompt)
            action_type = "generated"
            summary = f"Code {action_type} ({len(response)} chars)"
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "codewriter",
                    "instruction": instruction
                },
                short_summary=summary
            )
        except Exception as e:
            return ToolResult.error_result(f"Code generation failed: {str(e)}")

    def _general(self, prompt: Optional[str]) -> ToolResult:
        """Handle general queries"""

        system_prompt = """You are a helpful AI assistant.
Answer questions clearly and concisely.
Provide accurate, well-reasoned responses."""

        user_prompt = prompt

        try:
            response = self._call_llm(system_prompt, user_prompt)
            summary = f"General query completed ({len(response)} chars)"
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "general",
                    "prompt_length": len(prompt)
                },
                short_summary=summary
            )
        except Exception as e:
            return ToolResult.error_result(f"Query failed: {str(e)}")
