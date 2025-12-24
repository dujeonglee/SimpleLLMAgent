#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프롬프트 테스트 스크립트

orchestrator에서 생성되는 모든 프롬프트를 출력하여
프롬프트가 어떻게 구성되는지 확인합니다.
"""

import sys
import os
import json
from typing import Dict, List

# Windows 콘솔 UTF-8 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트를 path에 추가 (tests 폴더에서 한 단계 위로)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.shared_storage import SharedStorage
from core.base_tool import ToolRegistry
from core.orchestrator import Orchestrator, PlannedStep
from core.workspace_manager import WorkspaceManager, ConfigManager
from tools.file_tool import FileTool
from tools.web_tool import WebTool
from tools.llm_tool import LLMTool


def save_to_file(filename: str, content: str):
    """파일에 프롬프트 저장"""
    output_dir = "tests"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Saved: {filepath}")


def test_planning_prompt():
    """1. Planning 단계 프롬프트 테스트"""
    print("\n" + "=" * 80)
    print("  TEST 1: PLANNING PROMPT")
    print("=" * 80)

    # 기본 설정
    storage = SharedStorage()
    tools = ToolRegistry()
    config_manager = ConfigManager()

    # 도구 등록
    tools.register(FileTool())
    tools.register(WebTool())
    tools.register(LLMTool())

    # Orchestrator 생성
    orchestrator = Orchestrator(
        tools=tools,
        storage=storage,
        llm_config=config_manager.load_llm_config(),
        max_steps=10
    )

    # 테스트 쿼리
    test_query = "workspace/example.txt 파일을 읽고 요약해서 workspace/summary.md로 저장해줘"

    # tools schema 준비
    tools_schema = json.dumps(
        tools.get_all_schemas(),
        indent=2,
        ensure_ascii=False
    )

    # Planning 프롬프트 생성
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
- Maximum {orchestrator.max_steps} steps
- Use only available tools and actions
- Each step should have clear purpose
- Order steps logically (data dependencies)
- Respond with valid JSON only"""

    user_prompt = f"""## User Request
{test_query}

Create an execution plan or provide direct answer. Respond with JSON only."""

    # 파일로 저장
    content = f"""# Planning Prompt

## SYSTEM PROMPT
{system_prompt}

## USER PROMPT
{user_prompt}
"""
    save_to_file("01_planning_prompt.txt", content)


def test_execution_params_prompt():
    """2. Execution Parameters 단계 프롬프트 테스트"""
    print("\n" + "=" * 80)
    print("  TEST 2: EXECUTION PARAMS PROMPT")
    print("=" * 80)

    # 기본 설정
    storage = SharedStorage()
    tools = ToolRegistry()
    config_manager = ConfigManager()

    # 도구 등록
    tools.register(FileTool())
    tools.register(WebTool())
    tools.register(LLMTool())

    # Orchestrator 생성
    orchestrator = Orchestrator(
        tools=tools,
        storage=storage,
        llm_config=config_manager.load_llm_config(),
        max_steps=10
    )

    # 테스트용 세션 시작 및 더미 결과 추가
    test_query = "workspace/example.txt 파일을 읽고 요약해서 workspace/summary.md로 저장해줘"
    storage.start_session(test_query)
    storage.update_plan([
        "Step 1: file_tool.read - Read example.txt file",
        "Step 2: llm_tool.summarize - Summarize the content",
        "Step 3: file_tool.write - Save summary to summary.md"
    ])

    # 현재 실행할 스텝 (Step 2: llm_tool.summarize)
    planned_step = PlannedStep(
        step=2,
        tool_name="llm_tool",
        action="summarize",
        description="Summarize the content",
        params_hint={},
        status="pending"
    )

    # tools schema 준비
    tools_schema = json.dumps(
        tools.get_all_schemas(),
        indent=2,
        ensure_ascii=False
    )

    # Execution summary (시뮬레이션)
    execution_summary = """## User Request
workspace/example.txt 파일을 읽고 요약해서 workspace/summary.md로 저장해줘

## Execution Plan
  ✓ 1. Step 1: file_tool.read - Read example.txt file
  → 2. Step 2: llm_tool.summarize - Summarize the content (current)
    3. Step 3: file_tool.write - Save summary to summary.md

## Execution Results
[Step 1] ✓ file_tool.read
  Parameters: {"action": "read", "read_path": "workspace/example.txt"}
  Output: This is example content from the file.
It contains multiple lines of text.
We will summarize this content.
  Ref: [RESULT:result_001]"""

    # 마지막 출력 미리보기
    last_output_preview = "This is example content from the file.\nIt contains multiple lines of text.\nWe will summarize this content."

    system_prompt = f"""You are executing a planned task. Provide exact parameters for the current step.

## Available Tools
{tools_schema}

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

    user_prompt = f"""## Execution Context
{execution_summary}

## Current Step to Execute
Step {planned_step.step}: {planned_step.tool_name}.{planned_step.action}
Description: {planned_step.description}

Provide exact parameters for this step. Respond with JSON only."""

    # 파일로 저장
    content = f"""# Execution Parameters Prompt

## SYSTEM PROMPT
{system_prompt}

## USER PROMPT
{user_prompt}
"""
    save_to_file("02_execution_params_prompt.txt", content)


def test_final_answer_prompt():
    """3. Final Answer 단계 프롬프트 테스트"""
    print("\n" + "=" * 80)
    print("  TEST 3: FINAL ANSWER PROMPT")
    print("=" * 80)

    # 기본 설정
    storage = SharedStorage()
    tools = ToolRegistry()
    config_manager = ConfigManager()
    workspace_manager = WorkspaceManager()

    # Orchestrator 생성
    orchestrator = Orchestrator(
        tools=tools,
        storage=storage,
        llm_config=config_manager.load_llm_config(),
        max_steps=10
    )

    # 테스트용 쿼리
    test_query = "workspace/example.txt 파일을 읽고 요약해서 workspace/summary.md로 저장해줘"

    # Plan status (시뮬레이션)
    plan_status = """[✓] Step 1: file_tool.read - Read example.txt file
[✓] Step 2: llm_tool.summarize - Summarize the content
[✓] Step 3: file_tool.write - Save summary to summary.md"""

    # Previous results (시뮬레이션)
    previous_results = """[Step 1] ✅ file_tool.read - SUCCESS
Output: This is example content from the file.
It contains multiple lines of text.

[Step 2] ✅ llm_tool.summarize - SUCCESS
Output: Summary: The file contains example text with multiple lines describing various content.

[Step 3] ✅ file_tool.write - SUCCESS
Output: Successfully wrote to workspace/summary.md"""

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
{test_query}

## Execution Plan
{plan_status}

## Execution Results
{previous_results}

Based on the above results, provide a final answer to the user's query."""

    # 파일로 저장
    content = f"""# Final Answer Prompt

## SYSTEM PROMPT
{system_prompt}

## USER PROMPT
{user_prompt}
"""
    save_to_file("03_final_answer_prompt.txt", content)


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("  프롬프트 생성 테스트")
    print("  - Orchestrator에서 사용하는 모든 프롬프트를 개별 파일로 저장합니다")
    print("=" * 80)

    # 1. Planning 프롬프트
    test_planning_prompt()

    # 2. Execution Params 프롬프트
    test_execution_params_prompt()

    # 3. Final Answer 프롬프트
    test_final_answer_prompt()

    print("\n" + "=" * 80)
    print("  테스트 완료")
    print("=" * 80)
    print("모든 프롬프트가 prompts_output/ 폴더에 저장되었습니다.\n")


if __name__ == "__main__":
    main()
