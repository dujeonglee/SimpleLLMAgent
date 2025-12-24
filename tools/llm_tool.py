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
                description="Perform comprehensive code review with detailed feedback on code quality, best practices, potential bugs, and improvement suggestions",
                params=[
                    ActionParam("codereview_content", "str", False,
                               "Code content to review (provide either content or file_path)", None),
                    ActionParam("codereview_file_path", "str", False,
                               f"File path to the code file to review, relative to base_path ({self.base_path}) (provide either content or file_path)", None),
                    ActionParam("codereview_instruction", "str", True,
                               "Specific review instructions or focus areas (e.g., 'Focus on error handling', 'Check for security issues', 'Review API design')"),
                ],
                output_type="str",
                output_description="Detailed code review report in Markdown format with specific recommendations"
            ),
            "architectureanalysis": ActionSchema(
                name="architectureanalysis",
                description="Analyze software architecture, design patterns, system structure, component relationships, and architectural trade-offs",
                params=[
                    ActionParam("architectureanalysis_content", "str", False,
                               "Code or architectural documentation to analyze (provide either content or file_path)", None),
                    ActionParam("architectureanalysis_file_path", "str", False,
                               f"File path to analyze architecture from, relative to base_path ({self.base_path}) (provide either content or file_path)", None),
                    ActionParam("architectureanalysis_instruction", "str", True,
                               "Specific analysis instructions (e.g., 'Evaluate scalability', 'Identify design patterns', 'Suggest architectural improvements')"),
                ],
                output_type="str",
                output_description="Architecture analysis report with patterns identified and recommendations in Markdown format"
            ),
            "codedoc": ActionSchema(
                name="codedoc",
                description="Generate comprehensive documentation for code including docstrings, comments, API documentation, and usage examples",
                params=[
                    ActionParam("codedoc_content", "str", False,
                               "Code to document (provide either content or file_path)", None),
                    ActionParam("codedoc_file_path", "str", False,
                               f"File path to the code file to document, relative to base_path ({self.base_path}) (provide either content or file_path)", None),
                    ActionParam("codedoc_instruction", "str", True,
                               "Documentation instructions (e.g., 'Generate API docs', 'Add inline comments', 'Create usage examples', 'Write README section')"),
                ],
                output_type="str",
                output_description="Generated documentation in appropriate format (Markdown, docstrings, or comments)"
            ),
            "staticanalysis": ActionSchema(
                name="staticanalysis",
                description="Perform static code analysis to detect bugs, security vulnerabilities, code smells, type errors, and potential runtime issues without executing code",
                params=[
                    ActionParam("staticanalysis_content", "str", False,
                               "Code to analyze statically (provide either content or file_path)", None),
                    ActionParam("staticanalysis_file_path", "str", False,
                               f"File path to the code file for static analysis, relative to base_path ({self.base_path}) (provide either content or file_path)", None),
                    ActionParam("staticanalysis_instruction", "str", True,
                               "Analysis focus instructions (e.g., 'Check for null pointer dereferences', 'Find memory leaks', 'Detect SQL injection risks')"),
                ],
                output_type="str",
                output_description="Static analysis report with categorized issues, severity levels, and fix suggestions in Markdown format"
            ),
            "codewriter": ActionSchema(
                name="codewriter",
                description="Generate new code from scratch or modify existing code based on specifications, requirements, or improvement guidelines",
                params=[
                    ActionParam("codewriter_content", "str", False,
                               "Existing code to modify or extend (optional for new code generation)", None),
                    ActionParam("codewriter_file_path", "str", False,
                               f"File path to existing code for modification, relative to base_path ({self.base_path}) (optional for new code generation)", None),
                    ActionParam("codewriter_instruction", "str", True,
                               "Code writing instructions and specifications (e.g., 'Write a REST API handler', 'Add error handling to this function', 'Refactor using design pattern X')"),
                ],
                output_type="str",
                output_description="Generated or modified code with explanatory comments"
            ),
            "general": ActionSchema(
                name="general",
                description="Handle general queries, questions, and tasks that don't fit into specialized categories. Free-form interaction with LLM for any other purpose.",
                params=[
                    ActionParam("general_prompt", "str", True,
                               "Your question, request, or instruction for the LLM"),
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
                get_param("content"),
                get_param("file_path"),
                get_param("instruction")
            )
        elif action == "architectureanalysis":
            return self._architectureanalysis(
                get_param("content"),
                get_param("file_path"),
                get_param("instruction")
            )
        elif action == "codedoc":
            return self._codedoc(
                get_param("content"),
                get_param("file_path"),
                get_param("instruction")
            )
        elif action == "staticanalysis":
            return self._staticanalysis(
                get_param("content"),
                get_param("file_path"),
                get_param("instruction")
            )
        elif action == "codewriter":
            return self._codewriter(
                get_param("content"),
                get_param("file_path"),
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
    # Note: _resolve_path() and _validate_path() methods are inherited from BaseTool

    def _get_content(self, content: Optional[str], file_path: Optional[str]) -> tuple[str, str]:
        """
        Get content from either direct content or file path.
        Returns (content, error_message). If error_message is not empty, content retrieval failed.
        """
        if content:
            return content, ""

        if file_path:
            # Validate and resolve file path
            is_valid, resolved_or_error = self._validate_path(file_path)
            if not is_valid:
                return "", resolved_or_error

            resolved = resolved_or_error

            # Read from file
            try:
                if not os.path.exists(resolved):
                    return "", f"File not found: {file_path}"

                if not os.path.isfile(resolved):
                    return "", f"Not a file: {file_path}"

                with open(resolved, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                self.logger.debug(f"Read file: {resolved} ({len(file_content)} bytes)")
                return file_content, ""
            except UnicodeDecodeError as e:
                return "", f"Encoding error reading {file_path}: {str(e)}"
            except Exception as e:
                return "", f"Failed to read file {file_path}: {str(e)}"

        return "", "Either 'content' or 'file_path' must be provided"

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

    def _codereview(self, content: Optional[str], file_path: Optional[str], instruction: str) -> ToolResult:
        """Perform comprehensive code review"""
        code_content, error = self._get_content(content, file_path)
        if error:
            return ToolResult.error_result(error)

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

        if instruction:
            system_prompt += f"\n\nSpecific focus areas: {instruction}"

        user_prompt = f"Please review the following code:\n\n```\n{code_content}\n```"

        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "codereview",
                    "content_length": len(code_content),
                    "instruction": instruction
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Code review failed: {str(e)}")

    def _architectureanalysis(self, content: Optional[str], file_path: Optional[str], instruction: str) -> ToolResult:
        """Analyze software architecture"""
        arch_content, error = self._get_content(content, file_path)
        if error:
            return ToolResult.error_result(error)

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

        if instruction:
            system_prompt += f"\n\nSpecific analysis focus: {instruction}"

        user_prompt = f"Please analyze the architecture of:\n\n```\n{arch_content}\n```"

        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "architectureanalysis",
                    "content_length": len(arch_content),
                    "instruction": instruction
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Architecture analysis failed: {str(e)}")

    def _codedoc(self, content: Optional[str], file_path: Optional[str], instruction: str) -> ToolResult:
        """Generate code documentation"""
        code_content, error = self._get_content(content, file_path)
        if error:
            return ToolResult.error_result(error)

        system_prompt = """You are a technical documentation specialist.
Generate comprehensive documentation including:
- Clear descriptions of functionality
- Parameter and return value documentation
- Usage examples
- Edge cases and caveats
- Related functions or modules

Follow language-specific documentation conventions (docstrings, JSDoc, etc.).
Format appropriately for the code's language and context."""

        if instruction:
            system_prompt += f"\n\nDocumentation requirements: {instruction}"

        user_prompt = f"Please document the following code:\n\n```\n{code_content}\n```"

        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "codedoc",
                    "content_length": len(code_content),
                    "instruction": instruction
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Documentation generation failed: {str(e)}")

    def _staticanalysis(self, content: Optional[str], file_path: Optional[str], instruction: str) -> ToolResult:
        """Perform static code analysis"""
        code_content, error = self._get_content(content, file_path)
        if error:
            return ToolResult.error_result(error)

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

        if instruction:
            system_prompt += f"\n\nFocus areas: {instruction}"

        user_prompt = f"Please perform static analysis on:\n\n```\n{code_content}\n```"

        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "staticanalysis",
                    "content_length": len(code_content),
                    "instruction": instruction
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Static analysis failed: {str(e)}")

    def _codewriter(self, content: Optional[str], file_path: Optional[str], instruction: str) -> ToolResult:
        """Generate or modify code"""
        # For codewriter, content/file_path is optional (new code generation)
        existing_code = ""
        if content or file_path:
            existing_code, error = self._get_content(content, file_path)
            if error:
                return ToolResult.error_result(error)

        if existing_code:
            system_prompt = """You are an expert software engineer.
Modify or extend the provided code according to the specifications.
- Maintain existing code style and conventions
- Add clear comments for new functionality
- Ensure backward compatibility unless instructed otherwise
- Follow best practices and design patterns
- Write clean, readable, maintainable code"""

            user_prompt = f"""Existing code:
```
{existing_code}
```

Instructions: {instruction}

Provide the modified code with explanatory comments."""
        else:
            system_prompt = """You are an expert software engineer.
Write new code from scratch according to the specifications.
- Follow best practices and conventions
- Add clear, helpful comments
- Consider error handling and edge cases
- Write clean, readable, maintainable code
- Include usage examples if helpful"""

            user_prompt = f"""Instructions: {instruction}

Provide the complete code with explanatory comments."""

        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "codewriter",
                    "instruction": instruction,
                    "modification": bool(existing_code)
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Code generation failed: {str(e)}")

    def _general(self, prompt: str) -> ToolResult:
        """Handle general queries"""
        system_prompt = """You are a helpful AI assistant.
Answer questions clearly and concisely.
Provide accurate, well-reasoned responses."""

        user_prompt = prompt

        try:
            response = self._call_llm(system_prompt, user_prompt)
            return ToolResult.success_result(
                output=response,
                metadata={
                    "action": "general",
                    "prompt_length": len(prompt)
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Query failed: {str(e)}")
