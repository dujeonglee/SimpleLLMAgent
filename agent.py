#!/usr/bin/env python3
"""
LLM Agent with Ollama - V2 (JSON Mode)

개선 사항:
1. JSON Mode를 기본으로 사용하여 파싱 안정성 향상
2. 유연한 응답 구조 (tool_call, response, clarification)
3. 향상된 에러 처리 및 재시도 로직
4. 기존 $key.field 참조 시스템 유지
"""

import os
import json
import sys
import threading
import urllib.parse
import urllib.request
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from tkinter import messagebox


try:
    import tkinter as tk
    from tkinter import scrolledtext, ttk
except ImportError:
    print("Error: tkinter not installed")
    print("Run: sudo apt-get install python3-tk")
    sys.exit(1)


# ============================================================================
# 전역 변수 - 저장소
# ============================================================================

TOOL_RESULT_STORAGE: Dict[str, Any] = {}

# LLM 클라이언트 참조 (ask_llm에서 사용)
_OLLAMA_CLIENT: Optional['OllamaClient'] = None
_CURRENT_MODEL: Optional[str] = None

# ============================================================================
# Tool 결과 저장소 관리 함수
# ============================================================================

def store_tool_result(key: str, data: Any) -> None:
    """Tool 실행 결과를 저장"""
    global TOOL_RESULT_STORAGE
    TOOL_RESULT_STORAGE[key] = data


def resolve_references(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    인자에서 $key 참조를 실제 값으로 치환

    지원하는 참조 형식:
    1. 값 전체가 참조인 경우:
       - {"context": "$source_code.content"} → {"context": "파일 내용..."}
    
    2. 문자열 내부에 참조가 포함된 경우:
       - {"query": "분석해줘: $source_code.content"} → {"query": "분석해줘: 파일 내용..."}
    
    참조 문법:
    - $key: 전체 값
    - $key.field: 특정 필드
    - $key.field[0]: 배열 인덱스
    - $key.field[0].subfield: 중첩 접근
    """
    resolved = {}

    for param_name, param_value in arguments.items():
        if isinstance(param_value, str):
            resolved[param_name] = _resolve_string_references(param_value)
        else:
            resolved[param_name] = param_value

    return resolved

def _resolve_string_references(text: str) -> Any:
    """
    문자열 내의 모든 $key.field 참조를 치환
    
    - 문자열 전체가 단일 참조면 해당 타입 그대로 반환 (dict, list 등)
    - 문자열 내에 참조가 포함되어 있으면 문자열로 치환
    """
    import re
    
    # $key 또는 $key.field 또는 $key.field[0] 패턴 매칭
    # 단어 경계나 문자열 끝에서 끝나도록 함
    pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*|\[\d+\])*)'

    matches = list(re.finditer(pattern, text))

    if not matches:
        # 참조 없음 - 원본 반환
        return text
    
    # 문자열 전체가 단일 참조인 경우 (앞뒤 공백 허용)
    if len(matches) == 1:
        match = matches[0]
        if text.strip() == match.group(0):
            # 전체가 참조 → 원래 타입 유지 (dict, list 등)
            ref = match.group(1)
            return _resolve_single_reference(ref, match.group(0))
    
    # 문자열 내에 참조가 포함된 경우 → 문자열로 치환
    result = text
    # 뒤에서부터 치환해야 인덱스가 밀리지 않음
    for match in reversed(matches):
        ref = match.group(1)
        full_match = match.group(0)
        
        try:
            resolved_value = _resolve_single_reference(ref, full_match)
            # 치환할 값을 문자열로 변환
            if isinstance(resolved_value, str):
                replacement = resolved_value
            elif isinstance(resolved_value, (dict, list)):
                import json
                replacement = json.dumps(resolved_value, ensure_ascii=False)
            else:
                replacement = str(resolved_value)
            
            result = result[:match.start()] + replacement + result[match.end():]
        except ValueError:
            # 참조 해석 실패 시 원본 유지하지 않고 에러 발생
            raise
    
    return result

def _resolve_single_reference(ref: str, original: str) -> Any:
    """
    단일 참조를 해석
    
    ref: "files_to_delete.files[0]" 형태
    original: "$files_to_delete.files[0]" (에러 메시지용)
    """
    import re
    
    # 첫 번째 부분 (storage key) 추출
    if "." in ref:
        key, rest = ref.split(".", 1)
    else:
        key = ref
        rest = None
    
    # Storage에서 데이터 가져오기
    data = TOOL_RESULT_STORAGE.get(key)
    if data is None:
        raise ValueError(f"Reference '{original}' not found. Available keys: {list(TOOL_RESULT_STORAGE.keys())}")
    
    # 추가 경로가 없으면 전체 반환
    if rest is None:
        return data
    
    # 경로 파싱: "files[0].name" -> ["files", "[0]", "name"]
    # 정규식으로 필드명과 인덱스를 분리
    tokens = re.findall(r'(\w+)|\[(\d+)\]', rest)
    
    current = data
    path_so_far = f"${key}"
    
    for token in tokens:
        field_name, index = token
        
        if field_name:
            # 딕셔너리 필드 접근
            path_so_far += f".{field_name}"
            
            if not isinstance(current, dict):
                raise ValueError(f"Cannot access field '{field_name}' on non-dict type at '{path_so_far}'")
            
            if field_name not in current:
                available = list(current.keys()) if isinstance(current, dict) else 'N/A'
                raise ValueError(f"Field '{field_name}' not found at '{path_so_far}'. Available fields: {available}")
            
            current = current[field_name]
        
        elif index:
            # 배열 인덱스 접근
            idx = int(index)
            path_so_far += f"[{idx}]"
            
            if not isinstance(current, (list, tuple)):
                raise ValueError(f"Cannot use index [{idx}] on non-list type at '{path_so_far}'")
            
            if idx < 0 or idx >= len(current):
                raise ValueError(f"Index [{idx}] out of range at '{path_so_far}'. List has {len(current)} items (0-{len(current)-1})")
            
            current = current[idx]
    
    return current


def make_return_object(data: Dict[str, Any]) -> Dict[str, Any]:
    """표준화된 반환 객체 생성"""
    base = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": sys._getframe(1).f_code.co_name
    }
    return {**base, **data}


def get_storage_summary() -> str:
    """현재 저장소 상태 요약"""
    if not TOOL_RESULT_STORAGE:
        return "저장된 데이터가 없습니다."
    
    summary = []
    for key, value in TOOL_RESULT_STORAGE.items():
        if isinstance(value, dict):
            fields = list(value.keys())
            summary.append(f"${key}: {fields}")
        else:
            summary.append(f"${key}: {type(value).__name__}")
    return ", ".join(summary)


# ============================================================================
# 도구 정의
# ============================================================================

def get_file(base_dir: str = ".", pattern: str = "*") -> Dict[str, Any]:
    """현재 디렉토리를 기준으로 재귀적으로 모든 파일을 상대 경로로 가져오는 함수."""
    try:
        base_path = Path(base_dir).resolve()

        if not base_path.exists():
            return make_return_object({
                "result": "failure",
                "base_dir": base_dir,
                "error": f"Directory '{base_dir}' does not exist"
            })

        if pattern == "*":
            all_files = [str(f.relative_to(base_path)) for f in base_path.rglob("*") if f.is_file()]
        else:
            all_files = [str(f.relative_to(base_path)) for f in base_path.rglob(pattern) if f.is_file()]

        return make_return_object({
            "result": "success",
            "base_dir": str(base_path),
            "files": sorted(all_files),
            "count": len(all_files)
        })

    except Exception as e:
        return make_return_object({
            "result": "failure",
            "base_dir": base_dir,
            "error": str(e)
        })


def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """파일의 내용을 읽어오는 함수."""
    try:
        if not os.path.exists(file_path):
            return make_return_object({
                "result": "failure",
                "filename": file_path,
                "error": f"File '{file_path}' does not exist"
            })

        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()

        file_size = os.path.getsize(file_path)

        return make_return_object({
            "result": "success",
            "filename": file_path,
            "content": content,
            "size": file_size
        })

    except Exception as e:
        return make_return_object({
            "result": "failure",
            "filename": file_path,
            "error": str(e)
        })


def write_file(file_path: str, content: object) -> Dict[str, Any]:
    """파일을 저장하는 함수."""
    try:
        dir_path = os.path.dirname(file_path)

        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(content))

        file_size = os.path.getsize(file_path)

        return make_return_object({
            "result": "success",
            "filename": file_path,
            "size": file_size
        })

    except Exception as e:
        return make_return_object({
            "result": "failure",
            "filename": file_path,
            "error": str(e)
        })


def delete_file(file_path: str) -> Dict[str, Any]:
    """파일을 삭제하는 함수."""
    try:
        if not os.path.exists(file_path):
            return make_return_object({
                "result": "failure",
                "filename": file_path,
                "error": f"File '{file_path}' does not exist"
            })

        os.remove(file_path)
        return make_return_object({
            "result": "success",
            "filename": file_path
        })

    except Exception as e:
        return make_return_object({
            "result": "failure",
            "filename": file_path,
            "error": str(e)
        })


def ask_llm(query: str, context: str = "") -> Dict[str, Any]:
    """LLM에 쿼리를 보내고 결과를 반환하는 함수."""
    global _OLLAMA_CLIENT, _CURRENT_MODEL

    try:
        if _OLLAMA_CLIENT is None or _CURRENT_MODEL is None:
            return make_return_object({
                "result": "failure",
                "error": "LLM client not initialized. Please connect first."
            })

        if context:
            full_prompt = f"""Context:
{context}

Request:
{query}

Please provide a detailed and helpful response."""
        else:
            full_prompt = query

        messages = [{"role": "user", "content": full_prompt}]

        # Non-streaming, no JSON mode (자유 형식 응답)
        response = _OLLAMA_CLIENT.chat_simple(
            model=_CURRENT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4000
        )

        return make_return_object({
            "result": "success",
            "response": response
        })

    except Exception as e:
        return make_return_object({
            "result": "failure",
            "error": str(e)
        })


# Tool 레지스트리
TOOLS = {
    "get_file": {
        "function": get_file,
        "description": "Recursively get all files in a directory with relative paths",
        "parameters": {
            "base_dir": {
                "type": "string",
                "required": False,
                "default": ".",
                "description": "Base directory to search from (default: current directory)"
            },
            "pattern": {
                "type": "string",
                "required": False,
                "default": "*",
                "description": "File pattern to match (e.g., '*.c', '*.py', default: '*' for all files)"
            }
        }
    },
    "read_file": {
        "function": read_file,
        "description": "Read the contents of a file",
        "parameters": {
            "file_path": {
                "type": "string",
                "required": True,
                "description": "Path to the file to read"
            },
            "encoding": {
                "type": "string",
                "required": False,
                "default": "utf-8",
                "description": "File encoding (default: utf-8)"
            }
        }
    },
    "write_file": {
        "function": write_file,
        "description": "Write content to a file",
        "parameters": {
            "file_path": {
                "type": "string",
                "required": True,
                "description": "Path where the file should be written"
            },
            "content": {
                "type": "string",
                "required": True,
                "description": "Content to write to the file. Use $key.field reference for stored data"
            }
        }
    },
    "delete_file": {
        "function": delete_file,
        "description": "Delete a file",
        "parameters": {
            "file_path": {
                "type": "string",
                "required": True,
                "description": "Path to the file to delete"
            }
        }
    },
    "ask_llm": {
        "function": ask_llm,
        "description": "Send a query to LLM for analysis. Result is stored and accessible as $key.response",
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": "The question or request to send to LLM"
            },
            "context": {
                "type": "string",
                "required": False,
                "default": "",
                "description": "Additional context like file content (use $key.content reference)"
            }
        }
    }
}


# ============================================================================
# Ollama 클라이언트
# ============================================================================

class OllamaClient:
    """Ollama API 클라이언트 - JSON Mode 지원"""

    def __init__(self, base_url: str = "http://192.168.0.30:11434"):
        self.base_url = base_url

    def _request(self, endpoint: str, payload: dict, timeout: int = 1800) -> dict:
        """HTTP 요청 공통 로직"""
        headers = {'Content-Type': 'application/json'}
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(f"{self.base_url}{endpoint}", data=data, headers=headers)
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.getcode() != 200:
                raise Exception(f"Ollama API error: {response.getcode()}")
            return json.loads(response.read().decode('utf-8'))

    def chat_simple(self, model: str, messages: List[Dict], 
                    temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """단순 채팅 (스트리밍 없음, JSON 모드 없음)"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        result = self._request("/api/chat", payload)
        return result["message"]["content"]

    def chat_json_mode(self, model: str, messages: List[Dict],
                       temperature: float = 0.7, max_tokens: int = 4000) -> dict:
        """
        ⭐ JSON Mode 채팅 - 항상 유효한 JSON 반환
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",  # ⭐ JSON 출력 강제
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        result = self._request("/api/chat", payload)
        content = result["message"]["content"]
        
        # JSON 파싱 (format: json이므로 항상 유효해야 함)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # 만약의 경우 에러 처리
            raise Exception(f"JSON parsing failed despite format:json - {e}\nContent: {content[:500]}")

# ============================================================================
# Agent - JSON Mode (기본)
# ============================================================================

class OllamaAgentJsonMode:
    """JSON Mode를 사용하는 Agent (권장)"""

    def __init__(self, ollama_url: str, model: str):
        global _OLLAMA_CLIENT, _CURRENT_MODEL

        self.ollama = OllamaClient(ollama_url)
        self.model = model
        self.conversation_history = []

        _OLLAMA_CLIENT = self.ollama
        _CURRENT_MODEL = self.model

    def _create_system_prompt(self) -> str:
        """JSON Mode용 시스템 프롬프트"""
        
        # Tool 설명 생성
        tools_desc = []
        for name, info in TOOLS.items():
            params_desc = []
            for param_name, param_info in info['parameters'].items():
                req = "required" if param_info['required'] else "optional"
                p_str = f"{param_name} ({param_info['type']}, {req})"
                if 'default' in param_info:
                    p_str += f" default={param_info['default']}"
                params_desc.append(p_str)
            
            tools_desc.append(f"- {name}: {info['description']}\n  Parameters: {', '.join(params_desc) if params_desc else 'none'}")

        tools_text = "\n".join(tools_desc)
        
        # 현재 저장소 상태
        storage_info = get_storage_summary()

        return f"""You are a WiFi driver development assistant.

CRITICAL: Always respond with valid JSON only. No other text.

RESPONSE TYPES (choose one):

1. When you need to use a tool:
{{
    "type": "tool_call",
    "tool": "tool_name",
    "arguments": {{"param": "value"}},
    "store_as": "key_name",
    "reasoning": "brief explanation why this tool is needed"
}}

2. When you have a final answer:
{{
    "type": "response",
    "content": "your detailed response here in Korean."
}}

3. When you need clarification:
{{
    "type": "clarification",
    "question": "what information do you need?"
}}

AVAILABLE TOOLS:
{tools_text}

REFERENCE SYSTEM:
- Results are stored with the key in "store_as"
- Use $key or $key.field in arguments to reference stored data
- Use $key.field[index] to access array elements (0-based index)
- Example: {{"file_path": "$file_list.files[0]"}} references first file from stored key "file_list"

CURRENT STORAGE: {storage_info}

IMPORTANT RULES:
- All JSON keys and string values must use double quotes
- Use $key.field references instead of embedding large content
- Use $key.field[0], $key.field[1], etc. to access individual array items
- Always provide "store_as" for tool calls to enable chaining
- Respond in the same language as the user

Example of field references:
- $key.content: File content from read_file
- $key.files: File list from get_file (array)
- $key.files[0]: First file path from get_file
- $key.files[1]: Second file path from get_file
- $key.response: LLM response from ask_llm
- $key.result: Success/failure status
- $key.count: Number of items (from get_file)"""

    def _validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Tool call 유효성 검사. 에러 시 에러 메시지 반환, 성공 시 None"""
        if tool_name not in TOOLS:
            return f"Unknown tool: {tool_name}. Available tools: {list(TOOLS.keys())}"
        
        tool_info = TOOLS[tool_name]
        params = tool_info["parameters"]
        
        # Required 파라미터 검증
        for param_name, param_info in params.items():
            if param_info.get("required", False) and param_name not in arguments:
                return f"Missing required parameter: '{param_name}' for tool '{tool_name}'"
        
        # Unknown 파라미터 검증
        valid_params = set(params.keys())
        provided_params = set(arguments.keys())
        unknown = provided_params - valid_params
        if unknown:
            return f"Unknown parameters: {unknown}. Valid parameters: {valid_params}"
        
        return None

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Tool 실행"""
        # 유효성 검사
        error = self._validate_tool_call(tool_name, arguments)
        if error:
            return {"result": "failure", "error": error}
        
        try:
            # $key 참조 해결
            resolved_args = resolve_references(arguments)
            
            # 기본값 적용
            for param_name, param_info in TOOLS[tool_name]["parameters"].items():
                if param_name not in resolved_args and "default" in param_info:
                    resolved_args[param_name] = param_info["default"]
            
            return TOOLS[tool_name]["function"](**resolved_args)
        except ValueError as e:
            return {"result": "failure", "error": f"Reference error: {str(e)}"}
        except Exception as e:
            return {"result": "failure", "error": str(e)}

    def _summarize_result(self, result: Dict[str, Any], store_as: Optional[str]) -> str:
        """Tool 결과 요약"""
        if not isinstance(result, dict):
            return str(result)[:200]
        
        summary_parts = []
        
        if "result" in result:
            summary_parts.append(f"status: {result['result']}")
        
        if "error" in result:
            summary_parts.append(f"error: {result['error']}")
            return "{" + ", ".join(summary_parts) + "}"
        
        for key, value in result.items():
            if key in ["result", "error", "created_at", "created_by"]:
                continue
            
            if isinstance(value, str):
                if len(value) > 100:
                    summary_parts.append(f"{key}: <{len(value)} chars>")
                else:
                    display = value[:50] + "..." if len(value) > 50 else value
                    summary_parts.append(f'{key}: "{display}"')
            elif isinstance(value, list):
                summary_parts.append(f"{key}: [{len(value)} items]")
            elif isinstance(value, dict):
                summary_parts.append(f"{key}: {{...}}")
            else:
                summary_parts.append(f"{key}: {value}")
        
        summary = "{" + ", ".join(summary_parts) + "}"
        
        if store_as:
            fields = [k for k in result.keys() if k not in ["result", "created_at", "created_by"]]
            summary += f"\n→ Stored as ${store_as}"
            if fields:
                summary += f" (fields: {', '.join(fields[:5])})"
        
        return summary

    def chat(self, user_message: str, 
             stream_callback: Callable[[str], None] = None,
             status_callback: Callable[[str], None] = None,
             confirm_callback: Callable[[str, Dict], bool] = None,
             max_iterations: int = 10,
             max_tokens: int = 4000) -> str:
        """
        ⭐ JSON Mode Agent 메인 루프
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        for iteration in range(max_iterations):
            if status_callback:
                status_callback(f"🔄 Iteration {iteration + 1}")

            messages = [
                {"role": "system", "content": self._create_system_prompt()}
            ] + self.conversation_history

            try:
                # ⭐ JSON Mode 호출
                response = self.ollama.chat_json_mode(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                
                # 스트리밍 콜백으로 응답 표시
                if stream_callback:
                    stream_callback(json.dumps(response, indent=2, ensure_ascii=False))

                response_type = response.get("type", "unknown")

                # ===== 최종 응답 =====
                if response_type == "response":
                    content = response.get("content", "")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps(response, ensure_ascii=False)
                    })
                    if status_callback:
                        status_callback("✅ Complete")
                    return content

                # ===== 명확화 요청 =====
                elif response_type == "clarification":
                    question = response.get("question", "무엇을 도와드릴까요?")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps(response, ensure_ascii=False)
                    })
                    if status_callback:
                        status_callback("❓ Clarification needed")
                    return f"질문: {question}"

                # ===== Tool 호출 =====
                elif response_type == "tool_call":
                    tool_name = response.get("tool", "")
                    arguments = response.get("arguments", {})
                    store_as = response.get("store_as")
                    reasoning = response.get("reasoning", "")

                    if reasoning and stream_callback:
                        stream_callback(f"\n\n💭 Reasoning: {reasoning}")

                    # 사용자 확인
                    if confirm_callback:
                        if status_callback:
                            status_callback("⏸️ Waiting for confirmation...")
                        
                        if not confirm_callback(tool_name, arguments):
                            if status_callback:
                                status_callback("❌ Tool execution cancelled")
                            return "Tool execution was cancelled by user. How would you like to proceed?"

                    if status_callback:
                        status_callback(f"🔧 Executing: {tool_name}")

                    # Tool 실행
                    tool_result = self._execute_tool(tool_name, arguments)

                    # 결과 저장
                    if store_as and tool_result.get("result") == "success":
                        store_tool_result(store_as, tool_result)
                        if status_callback:
                            status_callback(f"💾 Stored as: ${store_as}")

                    if status_callback:
                        status_callback("📊 Tool completed")

                    # 대화 기록에 추가
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": json.dumps(response, ensure_ascii=False)
                    })

                    # Tool 결과를 JSON 형식으로
                    result_summary = self._summarize_result(tool_result, store_as)
                    tool_result_json = {
                        "type": "tool_result",
                        "tool": tool_name,
                        "success": tool_result.get("result") == "success",
                        "summary": result_summary,
                        "stored_as": store_as,
                        "available_storage": get_storage_summary()
                    }

                    self.conversation_history.append({
                        "role": "user",
                        "content": json.dumps(tool_result_json, ensure_ascii=False)
                    })

                    if stream_callback:
                        stream_callback(f"\n\n📊 Result:\n{result_summary}\n\n")

                else:
                    # Unknown type
                    if status_callback:
                        status_callback(f"⚠️ Unknown response type: {response_type}")
                    return f"Unexpected response type: {response_type}"

            except Exception as e:
                if status_callback:
                    status_callback(f"❌ Error: {str(e)}")
                
                # JSON 파싱 실패 시 재시도 메시지
                if "JSON" in str(e):
                    self.conversation_history.append({
                        "role": "user",
                        "content": json.dumps({
                            "type": "system_error",
                            "error": "Invalid JSON response. Please respond with valid JSON only.",
                            "hint": "Ensure all keys and string values use double quotes"
                        })
                    })
                    continue
                
                return f"Error: {str(e)}"

        return "Max iterations reached"

    def reset(self):
        """대화 및 저장소 초기화"""
        global TOOL_RESULT_STORAGE
        self.conversation_history = []
        TOOL_RESULT_STORAGE = {}


# ============================================================================
# GUI
# ============================================================================

class AgentGUI:
    """GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Agent V2 🤖 (JSON Mode)")
        self.root.geometry("1100x850")

        self.agent = None
        self.processing = False
        self.confirm_tool_execution = tk.BooleanVar(value=True)

        self.setup_ui()

    def setup_ui(self):
        """UI 구성"""

        # ===== 설정 프레임 =====
        config_frame = ttk.LabelFrame(self.root, text="⚙️ Ollama 설정", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        # Row 1: URL, Model, Connect
        ttk.Label(config_frame, text="URL:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.url_entry = ttk.Entry(config_frame, width=30)
        self.url_entry.insert(0, "http://192.168.0.30:11434")
        self.url_entry.grid(row=0, column=1, padx=5)

        ttk.Label(config_frame, text="Model:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.model_var = tk.StringVar(value="llama3.1")
        self.model_entry = ttk.Combobox(config_frame, textvariable=self.model_var, 
                                         state="readonly", width=25)
        self.model_entry.bind("<<ComboboxSelected>>", self._on_model_change)
        self.model_entry.grid(row=0, column=3, padx=5)

        self.refresh_btn = ttk.Button(config_frame, text="🔄 Connect", command=self.connect)
        self.refresh_btn.grid(row=0, column=4, padx=10)

        self.status_label = ttk.Label(config_frame, text="● Not connected", foreground="red")
        self.status_label.grid(row=0, column=5, padx=10)

        # Row 2: Max tokens, Confirm
        ttk.Label(config_frame, text="Max tokens:").grid(row=1, column=0, padx=5, sticky=tk.W, pady=5)
        self.max_tokens_var = tk.IntVar(value=4000)
        max_tokens_spinbox = ttk.Spinbox(config_frame, from_=1000, to=32000, 
                                          increment=1000, textvariable=self.max_tokens_var, width=8)
        max_tokens_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        confirm_check = ttk.Checkbutton(config_frame, text="Confirm tool execution",
                                        variable=self.confirm_tool_execution)
        confirm_check.grid(row=1, column=2, padx=10, pady=5)

        # ===== 채팅 영역 =====
        chat_frame = ttk.LabelFrame(self.root, text="💬 Conversation", padding=10)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        paned = ttk.PanedWindow(chat_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # 왼쪽: 채팅 디스플레이
        chat_container = ttk.Frame(paned)
        paned.add(chat_container, weight=3)

        self.chat_display = scrolledtext.ScrolledText(
            chat_container, wrap=tk.WORD, width=70, height=30,
            font=("Consolas", 10), state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # 오른쪽: Storage TreeView
        storage_container = ttk.LabelFrame(paned, text="📦 Storage ($key)", padding=5)
        paned.add(storage_container, weight=1)

        self.storage_tree = ttk.Treeview(storage_container, show="tree headings", columns=("value",))
        self.storage_tree.heading("#0", text="Key", anchor=tk.W)
        self.storage_tree.heading("value", text="Value", anchor=tk.W)
        self.storage_tree.column("#0", width=120, minwidth=80)
        self.storage_tree.column("value", width=200, minwidth=100)

        tree_scroll_y = ttk.Scrollbar(storage_container, orient=tk.VERTICAL, 
                                       command=self.storage_tree.yview)
        tree_scroll_x = ttk.Scrollbar(storage_container, orient=tk.HORIZONTAL, 
                                       command=self.storage_tree.xview)
        self.storage_tree.configure(yscrollcommand=tree_scroll_y.set, 
                                     xscrollcommand=tree_scroll_x.set)

        self.storage_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        storage_container.grid_rowconfigure(0, weight=1)
        storage_container.grid_columnconfigure(0, weight=1)

        refresh_storage_btn = ttk.Button(storage_container, text="🔄 Refresh", 
                                          command=self.refresh_storage_tree)
        refresh_storage_btn.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        # 태그 설정
        self.chat_display.tag_config("user", foreground="#2196F3", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("assistant", foreground="#4CAF50", font=("Consolas", 10))
        self.chat_display.tag_config("system", foreground="#FF9800", font=("Consolas", 9, "italic"))
        self.chat_display.tag_config("json", foreground="#9C27B0", font=("Consolas", 9))

        # ===== 입력 영역 =====
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X)

        self.input_text = tk.Text(input_frame, height=3, font=("Consolas", 10))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_text.bind("<Control-Return>", lambda e: self.send_message())

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.send_btn = ttk.Button(btn_frame, text="Send\n(Ctrl+Enter)",
                                   command=self.send_message, state=tk.DISABLED)
        self.send_btn.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.reset_btn = ttk.Button(btn_frame, text="Reset\nChat",
                                    command=self.reset_chat, state=tk.DISABLED)
        self.reset_btn.pack(fill=tk.BOTH, expand=True)

        # ===== 도구 정보 =====
        tools_frame = ttk.LabelFrame(self.root, text="🔧 Available Tools", padding=10)
        tools_frame.pack(fill=tk.X, padx=10, pady=5)

        tools_text = "  •  ".join([f"{name}" for name in TOOLS.keys()])
        ttk.Label(tools_frame, text=tools_text, font=("Consolas", 9)).pack()

    def _on_model_change(self, event=None):
        """모델 변경 시 Agent 재생성"""
        self._create_agent()

    def _create_agent(self):
        """현재 설정으로 Agent 생성"""
        if not self.model_var.get():
            return
        
        self.agent = OllamaAgentJsonMode(self.url_entry.get(), self.model_var.get())
        self.append_text(f"[System] Agent created (JSON Mode)\n", "system")

    def refresh_storage_tree(self):
        """Storage TreeView 갱신"""
        for item in self.storage_tree.get_children():
            self.storage_tree.delete(item)

        for key, value in TOOL_RESULT_STORAGE.items():
            parent_id = self.storage_tree.insert("", tk.END, text=f"${key}", open=True)

            if isinstance(value, dict):
                for field, field_value in value.items():
                    display_value = self._format_tree_value(field_value)
                    self.storage_tree.insert(parent_id, tk.END, text=field, values=(display_value,))
            else:
                display_value = self._format_tree_value(value)
                self.storage_tree.insert(parent_id, tk.END, text="(value)", values=(display_value,))

    def _format_tree_value(self, value: Any) -> str:
        """TreeView에 표시할 값 포맷팅"""
        if isinstance(value, str):
            return f"<{len(value)} chars>" if len(value) > 50 else value
        elif isinstance(value, list):
            return f"[{len(value)} items]"
        elif isinstance(value, dict):
            return f"{{{len(value)} fields}}"
        return str(value)

    def append_text(self, text: str, tag: str = None):
        """텍스트 추가 (thread-safe)"""
        def _append():
            self.chat_display.config(state=tk.NORMAL)
            if tag:
                self.chat_display.insert(tk.END, text, tag)
            else:
                self.chat_display.insert(tk.END, text)
            self.chat_display.see(tk.END)
            self.chat_display.config(state=tk.DISABLED)
        self.root.after(0, _append)

    def set_status(self, text: str, color: str = "black"):
        """상태 업데이트"""
        def _update():
            self.status_label.config(text=text, foreground=color)
        self.root.after(0, _update)

    def confirm_tool_execution_dialog(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """도구 실행 확인 다이얼로그"""
        if not self.confirm_tool_execution.get():
            return True

        args_formatted = json.dumps(arguments, indent=2, ensure_ascii=False)
        tool_desc = TOOLS.get(tool_name, {}).get('description', 'No description')

        message = f"""The agent wants to execute a tool:

Tool: {tool_name}
Description: {tool_desc}

Arguments:
{args_formatted}

Do you want to proceed?"""

        return messagebox.askyesno("Confirm Tool Execution", message, icon='question')

    def connect(self):
        """Ollama 연결"""
        def _connect():
            try:
                # 모델 목록 가져오기
                url = urllib.parse.urljoin(self.url_entry.get(), "/api/tags")
                with urllib.request.urlopen(url, timeout=10) as response:
                    data = json.load(response)
                    models = [model["name"] for model in data["models"]]

                def _update_ui():
                    self.model_entry["values"] = models
                    if models:
                        self.model_entry.set(models[0])
                        self._create_agent()
                    self.set_status("● Connected", "green")
                    self.append_text(f"[System] Connected to {self.url_entry.get()}\n", "system")
                    self.append_text(f"[System] Available models: {', '.join(models)}\n", "system")
                    self.send_btn.config(state=tk.NORMAL)
                    self.reset_btn.config(state=tk.NORMAL)

                self.root.after(0, _update_ui)

            except Exception as e:
                def _show_error():
                    self.set_status("● Connection failed", "red")
                    self.append_text(f"[System] ❌ Connection failed: {str(e)}\n", "system")
                self.root.after(0, _show_error)

        threading.Thread(target=_connect, daemon=True).start()

    def send_message(self):
        """메시지 전송"""
        if self.processing or not self.agent:
            return

        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            return

        self.input_text.delete("1.0", tk.END)
        self.append_text(f"👤 You:\n", "user")
        self.append_text(f"{user_input}\n\n")

        self.processing = True
        self.send_btn.config(state=tk.DISABLED)
        self.input_text.config(state=tk.DISABLED)

        self.append_text(f"🤖 Assistant:\n", "assistant")

        def _process():
            try:
                def stream_cb(token):
                    self.append_text(token)

                def status_cb(status):
                    self.append_text(f"\n[{status}]\n", "system")
                    self.root.after(0, self.refresh_storage_tree)

                def confirm_cb(tool_name, arguments):
                    result_container = [None]
                    event = threading.Event()

                    def _ask():
                        result_container[0] = self.confirm_tool_execution_dialog(tool_name, arguments)
                        event.set()

                    self.root.after(0, _ask)
                    event.wait()
                    return result_container[0]

                self.agent.chat(
                    user_input,
                    stream_callback=stream_cb,
                    status_callback=status_cb,
                    confirm_callback=confirm_cb,
                    max_tokens=self.max_tokens_var.get()
                )

                self.append_text("\n\n" + "=" * 80 + "\n\n")

            except Exception as e:
                self.append_text(f"\n\n❌ Error: {str(e)}\n\n", "system")

            finally:
                self.processing = False
                self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.input_text.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.input_text.focus())

        threading.Thread(target=_process, daemon=True).start()

    def reset_chat(self):
        """채팅 초기화"""
        if self.agent:
            self.agent.reset()

        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)

        self.refresh_storage_tree()
        self.append_text("[System] Chat reset! 🔄\n\n", "system")


# ============================================================================
# Main
# ============================================================================

def main():
    root = tk.Tk()
    app = AgentGUI(root)
    app.reset_chat()
    root.mainloop()


if __name__ == "__main__":
    main()
