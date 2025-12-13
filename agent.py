#!/usr/bin/env python3
"""
LLM Agent with Ollama - V2 (JSON Mode)
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

# 모델 설정 (Agent용, ask_llm용 분리)
_AGENT_MODEL: Optional[str] = None
_AGENT_MAX_TOKENS: int = 4000
_ASK_LLM_MODEL: Optional[str] = None
_ASK_LLM_MAX_TOKENS: int = 4000


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
    
    pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*|\[\d+\])*)'
    matches = list(re.finditer(pattern, text))

    if not matches:
        return text
    
    if len(matches) == 1:
        match = matches[0]
        if text.strip() == match.group(0):
            ref = match.group(1)
            return _resolve_single_reference(ref, match.group(0))
    
    result = text
    for match in reversed(matches):
        ref = match.group(1)
        full_match = match.group(0)
        
        try:
            resolved_value = _resolve_single_reference(ref, full_match)
            if isinstance(resolved_value, str):
                replacement = resolved_value
            elif isinstance(resolved_value, (dict, list)):
                replacement = json.dumps(resolved_value, ensure_ascii=False)
            else:
                replacement = str(resolved_value)
            
            result = result[:match.start()] + replacement + result[match.end():]
        except ValueError:
            raise
    
    return result

def _resolve_single_reference(ref: str, original: str) -> Any:
    """단일 참조를 해석"""
    import re
    
    if "." in ref:
        key, rest = ref.split(".", 1)
    else:
        key = ref
        rest = None
    
    data = TOOL_RESULT_STORAGE.get(key)
    if data is None:
        raise ValueError(f"Reference '{original}' not found. Available keys: {list(TOOL_RESULT_STORAGE.keys())}")
    
    if rest is None:
        return data
    
    tokens = re.findall(r'(\w+)|\[(\d+)\]', rest)
    current = data
    path_so_far = f"${key}"
    
    for token in tokens:
        field_name, index = token
        
        if field_name:
            path_so_far += f".{field_name}"
            if not isinstance(current, dict):
                raise ValueError(f"Cannot access field '{field_name}' on non-dict type at '{path_so_far}'")
            if field_name not in current:
                available = list(current.keys()) if isinstance(current, dict) else 'N/A'
                raise ValueError(f"Field '{field_name}' not found at '{path_so_far}'. Available fields: {available}")
            current = current[field_name]
        elif index:
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
    """LLM에 쿼리를 보내고 결과를 반환하는 함수. (chat_simple 사용)"""
    global _OLLAMA_CLIENT, _ASK_LLM_MODEL, _ASK_LLM_MAX_TOKENS

    try:
        if _OLLAMA_CLIENT is None or _ASK_LLM_MODEL is None:
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

        # ⭐ chat_simple 사용 - 별도 모델/토큰 설정
        response = _OLLAMA_CLIENT.chat_simple(
            model=_ASK_LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=_ASK_LLM_MAX_TOKENS
        )

        return make_return_object({
            "result": "success",
            "response": response,
            "model_used": _ASK_LLM_MODEL,
            "max_tokens_used": _ASK_LLM_MAX_TOKENS
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
            "base_dir": {"type": "string", "required": False, "default": ".", "description": "Base directory to search from"},
            "pattern": {"type": "string", "required": False, "default": "*", "description": "File pattern to match (e.g., '*.c', '*.py')"}
        }
    },
    "read_file": {
        "function": read_file,
        "description": "Read the contents of a file",
        "parameters": {
            "file_path": {"type": "string", "required": True, "description": "Path to the file to read"},
            "encoding": {"type": "string", "required": False, "default": "utf-8", "description": "File encoding"}
        }
    },
    "write_file": {
        "function": write_file,
        "description": "Write content to a file",
        "parameters": {
            "file_path": {"type": "string", "required": True, "description": "Path where the file should be written"},
            "content": {"type": "string", "required": True, "description": "Content to write. Use $key.field reference for stored data"}
        }
    },
    "delete_file": {
        "function": delete_file,
        "description": "Delete a file",
        "parameters": {
            "file_path": {"type": "string", "required": True, "description": "Path to the file to delete"}
        }
    },
    "ask_llm": {
        "function": ask_llm,
        "description": "Send a query to LLM for analysis (uses separate model settings). Result accessible as $key.response",
        "parameters": {
            "query": {"type": "string", "required": True, "description": "The question or request to send to LLM"},
            "context": {"type": "string", "required": False, "default": "", "description": "Additional context (use $key.content reference)"}
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
        """단순 채팅 (스트리밍 없음, JSON 모드 없음) - ask_llm에서 사용"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        result = self._request("/api/chat", payload)
        return result["message"]["content"]

    def chat_json_mode(self, model: str, messages: List[Dict],
                       temperature: float = 0.7, max_tokens: int = 4000) -> dict:
        """⭐ JSON Mode 채팅 - Agent에서 사용"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        result = self._request("/api/chat", payload)
        content = result["message"]["content"]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parsing failed: {e}\nContent: {content[:500]}")


# ============================================================================
# Agent - JSON Mode
# ============================================================================

class OllamaAgentJsonMode:
    """JSON Mode를 사용하는 Agent"""

    def __init__(self, ollama_url: str, agent_model: str, agent_max_tokens: int,
                 ask_llm_model: str, ask_llm_max_tokens: int):
        global _OLLAMA_CLIENT, _AGENT_MODEL, _AGENT_MAX_TOKENS
        global _ASK_LLM_MODEL, _ASK_LLM_MAX_TOKENS

        self.ollama = OllamaClient(ollama_url)
        self.model = agent_model
        self.max_tokens = agent_max_tokens
        self.conversation_history = []

        _OLLAMA_CLIENT = self.ollama
        _AGENT_MODEL = agent_model
        _AGENT_MAX_TOKENS = agent_max_tokens
        _ASK_LLM_MODEL = ask_llm_model
        _ASK_LLM_MAX_TOKENS = ask_llm_max_tokens

    def update_settings(self, agent_model: str = None, agent_max_tokens: int = None,
                        ask_llm_model: str = None, ask_llm_max_tokens: int = None):
        """설정 업데이트"""
        global _AGENT_MODEL, _AGENT_MAX_TOKENS, _ASK_LLM_MODEL, _ASK_LLM_MAX_TOKENS
        
        if agent_model:
            self.model = agent_model
            _AGENT_MODEL = agent_model
        if agent_max_tokens:
            self.max_tokens = agent_max_tokens
            _AGENT_MAX_TOKENS = agent_max_tokens
        if ask_llm_model:
            _ASK_LLM_MODEL = ask_llm_model
        if ask_llm_max_tokens:
            _ASK_LLM_MAX_TOKENS = ask_llm_max_tokens

    def _create_system_prompt(self) -> str:
        """JSON Mode용 시스템 프롬프트"""
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
    "reasoning": "brief explanation"
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
- Example: {{"file_path": "$file_list.files[0]"}}

CURRENT STORAGE: {storage_info}

IMPORTANT RULES:
- All JSON keys and string values must use double quotes
- Use $key.field references instead of embedding large content
- Always provide "store_as" for tool calls to enable chaining
- Respond in the same language as the user

Example references:
- $key.content: File content from read_file
- $key.files[0]: First file path from get_file
- $key.response: LLM response from ask_llm"""

    def _validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Tool call 유효성 검사"""
        if tool_name not in TOOLS:
            return f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}"
        
        tool_info = TOOLS[tool_name]
        params = tool_info["parameters"]
        
        for param_name, param_info in params.items():
            if param_info.get("required", False) and param_name not in arguments:
                return f"Missing required parameter: '{param_name}' for tool '{tool_name}'"
        
        valid_params = set(params.keys())
        provided_params = set(arguments.keys())
        unknown = provided_params - valid_params
        if unknown:
            return f"Unknown parameters: {unknown}. Valid: {valid_params}"
        
        return None

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Tool 실행"""
        error = self._validate_tool_call(tool_name, arguments)
        if error:
            return {"result": "failure", "error": error}
        
        try:
            resolved_args = resolve_references(arguments)
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
             max_iterations: int = 10) -> str:
        """⭐ JSON Mode Agent 메인 루프"""
        self.conversation_history.append({"role": "user", "content": user_message})

        for iteration in range(max_iterations):
            if status_callback:
                status_callback(f"🔄 Iteration {iteration + 1}")

            messages = [{"role": "system", "content": self._create_system_prompt()}] + self.conversation_history

            try:
                response = self.ollama.chat_json_mode(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=self.max_tokens
                )
                
                if stream_callback:
                    stream_callback(json.dumps(response, indent=2, ensure_ascii=False))

                response_type = response.get("type", "unknown")

                if response_type == "response":
                    content = response.get("content", "")
                    self.conversation_history.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
                    if status_callback:
                        status_callback("✅ Complete")
                    return content

                elif response_type == "clarification":
                    question = response.get("question", "무엇을 도와드릴까요?")
                    self.conversation_history.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
                    if status_callback:
                        status_callback("❓ Clarification needed")
                    return f"질문: {question}"

                elif response_type == "tool_call":
                    tool_name = response.get("tool", "")
                    arguments = response.get("arguments", {})
                    store_as = response.get("store_as")
                    reasoning = response.get("reasoning", "")

                    if reasoning and stream_callback:
                        stream_callback(f"\n\n💭 Reasoning: {reasoning}")

                    if confirm_callback:
                        if status_callback:
                            status_callback("⏸️ Waiting for confirmation...")
                        if not confirm_callback(tool_name, arguments):
                            if status_callback:
                                status_callback("❌ Tool execution cancelled")
                            return "Tool execution was cancelled by user."

                    if status_callback:
                        status_callback(f"🔧 Executing: {tool_name}")

                    tool_result = self._execute_tool(tool_name, arguments)

                    if store_as and tool_result.get("result") == "success":
                        store_tool_result(store_as, tool_result)
                        if status_callback:
                            status_callback(f"💾 Stored as: ${store_as}")

                    if status_callback:
                        status_callback("📊 Tool completed")

                    self.conversation_history.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})

                    result_summary = self._summarize_result(tool_result, store_as)
                    tool_result_json = {
                        "type": "tool_result",
                        "tool": tool_name,
                        "success": tool_result.get("result") == "success",
                        "summary": result_summary,
                        "stored_as": store_as,
                        "available_storage": get_storage_summary()
                    }
                    self.conversation_history.append({"role": "user", "content": json.dumps(tool_result_json, ensure_ascii=False)})

                    if stream_callback:
                        stream_callback(f"\n\n📊 Result:\n{result_summary}\n\n")

                else:
                    if status_callback:
                        status_callback(f"⚠️ Unknown response type: {response_type}")
                    return f"Unexpected response type: {response_type}"

            except Exception as e:
                if status_callback:
                    status_callback(f"❌ Error: {str(e)}")
                
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
        self.root.geometry("1200x900")

        self.agent = None
        self.processing = False
        self.confirm_tool_execution = tk.BooleanVar(value=True)
        self.available_models = []

        self.setup_ui()

    def setup_ui(self):
        """UI 구성"""

        # ===== 연결 설정 프레임 =====
        connect_frame = ttk.LabelFrame(self.root, text="🔗 Ollama 연결", padding=10)
        connect_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(connect_frame, text="URL:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.url_entry = ttk.Entry(connect_frame, width=35)
        self.url_entry.insert(0, "http://192.168.0.30:11434")
        self.url_entry.grid(row=0, column=1, padx=5)

        self.refresh_btn = ttk.Button(connect_frame, text="🔄 Connect", command=self.connect)
        self.refresh_btn.grid(row=0, column=2, padx=10)

        self.status_label = ttk.Label(connect_frame, text="● Not connected", foreground="red")
        self.status_label.grid(row=0, column=3, padx=10)

        # ===== 모델 설정 프레임 =====
        model_frame = ttk.LabelFrame(self.root, text="⚙️ 모델 설정", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        # Agent 설정 (chat_json_mode)
        ttk.Label(model_frame, text="🤖 Agent (JSON Mode):", font=("", 9, "bold")).grid(row=0, column=0, padx=5, sticky=tk.W)
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=1, padx=5, sticky=tk.E)
        self.agent_model_var = tk.StringVar()
        self.agent_model_combo = ttk.Combobox(model_frame, textvariable=self.agent_model_var, state="readonly", width=25)
        self.agent_model_combo.grid(row=0, column=2, padx=5)
        self.agent_model_combo.bind("<<ComboboxSelected>>", self._on_agent_model_change)

        ttk.Label(model_frame, text="Max Tokens:").grid(row=0, column=3, padx=5, sticky=tk.E)
        self.agent_tokens_var = tk.IntVar(value=4000)
        agent_tokens_spinbox = ttk.Spinbox(model_frame, from_=1000, to=32000, increment=1000, textvariable=self.agent_tokens_var, width=8)
        agent_tokens_spinbox.grid(row=0, column=4, padx=5, sticky=tk.W)
        agent_tokens_spinbox.bind("<FocusOut>", self._on_agent_tokens_change)
        agent_tokens_spinbox.bind("<Return>", self._on_agent_tokens_change)

        # ask_llm 설정 (chat_simple)
        ttk.Label(model_frame, text="💬 ask_llm (Chat Mode):", font=("", 9, "bold")).grid(row=1, column=0, padx=5, sticky=tk.W, pady=(10,0))
        
        ttk.Label(model_frame, text="Model:").grid(row=1, column=1, padx=5, sticky=tk.E, pady=(10,0))
        self.ask_llm_model_var = tk.StringVar()
        self.ask_llm_model_combo = ttk.Combobox(model_frame, textvariable=self.ask_llm_model_var, state="readonly", width=25)
        self.ask_llm_model_combo.grid(row=1, column=2, padx=5, pady=(10,0))
        self.ask_llm_model_combo.bind("<<ComboboxSelected>>", self._on_ask_llm_model_change)

        ttk.Label(model_frame, text="Max Tokens:").grid(row=1, column=3, padx=5, sticky=tk.E, pady=(10,0))
        self.ask_llm_tokens_var = tk.IntVar(value=4000)
        ask_llm_tokens_spinbox = ttk.Spinbox(model_frame, from_=1000, to=32000, increment=1000, textvariable=self.ask_llm_tokens_var, width=8)
        ask_llm_tokens_spinbox.grid(row=1, column=4, padx=5, sticky=tk.W, pady=(10,0))
        ask_llm_tokens_spinbox.bind("<FocusOut>", self._on_ask_llm_tokens_change)
        ask_llm_tokens_spinbox.bind("<Return>", self._on_ask_llm_tokens_change)

        # 현재 설정 표시
        self.settings_label = ttk.Label(model_frame, text="", font=("Consolas", 8), foreground="gray")
        self.settings_label.grid(row=2, column=0, columnspan=4, padx=5, pady=(10,0), sticky=tk.W)

        # 옵션
        confirm_check = ttk.Checkbutton(model_frame, text="Tool 실행 전 확인", variable=self.confirm_tool_execution)
        confirm_check.grid(row=2, column=4, padx=10, pady=(10,0), sticky=tk.E)

        # ===== 채팅 영역 =====
        chat_frame = ttk.LabelFrame(self.root, text="💬 Conversation", padding=10)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        paned = ttk.PanedWindow(chat_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # 왼쪽: 채팅 디스플레이
        chat_container = ttk.Frame(paned)
        paned.add(chat_container, weight=3)

        self.chat_display = scrolledtext.ScrolledText(chat_container, wrap=tk.WORD, width=70, height=30, font=("Consolas", 10), state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # 오른쪽: Storage TreeView
        storage_container = ttk.LabelFrame(paned, text="📦 Storage ($key)", padding=5)
        paned.add(storage_container, weight=1)

        self.storage_tree = ttk.Treeview(storage_container, show="tree headings", columns=("value",))
        self.storage_tree.heading("#0", text="Key", anchor=tk.W)
        self.storage_tree.heading("value", text="Value", anchor=tk.W)
        self.storage_tree.column("#0", width=120, minwidth=80)
        self.storage_tree.column("value", width=200, minwidth=100)

        tree_scroll_y = ttk.Scrollbar(storage_container, orient=tk.VERTICAL, command=self.storage_tree.yview)
        tree_scroll_x = ttk.Scrollbar(storage_container, orient=tk.HORIZONTAL, command=self.storage_tree.xview)
        self.storage_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

        self.storage_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        storage_container.grid_rowconfigure(0, weight=1)
        storage_container.grid_columnconfigure(0, weight=1)

        refresh_storage_btn = ttk.Button(storage_container, text="🔄 Refresh", command=self.refresh_storage_tree)
        refresh_storage_btn.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        # 태그 설정
        self.chat_display.tag_config("user", foreground="#2196F3", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("assistant", foreground="#4CAF50", font=("Consolas", 10))
        self.chat_display.tag_config("system", foreground="#FF9800", font=("Consolas", 9, "italic"))

        # ===== 입력 영역 =====
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X)

        self.input_text = tk.Text(input_frame, height=3, font=("Consolas", 10))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_text.bind("<Control-Return>", lambda e: self.send_message())

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.send_btn = ttk.Button(btn_frame, text="Send\n(Ctrl+Enter)", command=self.send_message, state=tk.DISABLED)
        self.send_btn.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.reset_btn = ttk.Button(btn_frame, text="Reset\nChat", command=self.reset_chat, state=tk.DISABLED)
        self.reset_btn.pack(fill=tk.BOTH, expand=True)

        # ===== 도구 정보 =====
        tools_frame = ttk.LabelFrame(self.root, text="🔧 Available Tools", padding=10)
        tools_frame.pack(fill=tk.X, padx=10, pady=5)

        tools_text = "  •  ".join([f"{name}" for name in TOOLS.keys()])
        ttk.Label(tools_frame, text=tools_text, font=("Consolas", 9)).pack()

    def _update_settings_label(self):
        """현재 설정 레이블 업데이트"""
        agent_model = self.agent_model_var.get() or "None"
        agent_tokens = self.agent_tokens_var.get()
        ask_llm_model = self.ask_llm_model_var.get() or "None"
        ask_llm_tokens = self.ask_llm_tokens_var.get()
        self.settings_label.config(text=f"Agent: {agent_model} ({agent_tokens} tokens) | ask_llm: {ask_llm_model} ({ask_llm_tokens} tokens)")

    def _on_agent_model_change(self, event=None):
        if self.agent:
            self.agent.update_settings(agent_model=self.agent_model_var.get())
        self._update_settings_label()
        self.append_text(f"[System] Agent model → {self.agent_model_var.get()}\n", "system")

    def _on_agent_tokens_change(self, event=None):
        if self.agent:
            self.agent.update_settings(agent_max_tokens=self.agent_tokens_var.get())
        self._update_settings_label()

    def _on_ask_llm_model_change(self, event=None):
        if self.agent:
            self.agent.update_settings(ask_llm_model=self.ask_llm_model_var.get())
        self._update_settings_label()
        self.append_text(f"[System] ask_llm model → {self.ask_llm_model_var.get()}\n", "system")

    def _on_ask_llm_tokens_change(self, event=None):
        if self.agent:
            self.agent.update_settings(ask_llm_max_tokens=self.ask_llm_tokens_var.get())
        self._update_settings_label()

    def _create_agent(self):
        if not self.agent_model_var.get():
            return
        self.agent = OllamaAgentJsonMode(
            ollama_url=self.url_entry.get(),
            agent_model=self.agent_model_var.get(),
            agent_max_tokens=self.agent_tokens_var.get(),
            ask_llm_model=self.ask_llm_model_var.get(),
            ask_llm_max_tokens=self.ask_llm_tokens_var.get()
        )
        self._update_settings_label()
        self.append_text(f"[System] Agent created\n", "system")

    def refresh_storage_tree(self):
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
        if isinstance(value, str):
            return f"<{len(value)} chars>" if len(value) > 50 else value
        elif isinstance(value, list):
            return f"[{len(value)} items]"
        elif isinstance(value, dict):
            return f"{{{len(value)} fields}}"
        return str(value)

    def append_text(self, text: str, tag: str = None):
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
        def _update():
            self.status_label.config(text=text, foreground=color)
        self.root.after(0, _update)

    def confirm_tool_execution_dialog(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        if not self.confirm_tool_execution.get():
            return True
        args_formatted = json.dumps(arguments, indent=2, ensure_ascii=False)
        tool_desc = TOOLS.get(tool_name, {}).get('description', 'No description')
        message = f"""Tool 실행 요청:

Tool: {tool_name}
설명: {tool_desc}

Arguments:
{args_formatted}

실행하시겠습니까?"""
        return messagebox.askyesno("Tool 실행 확인", message, icon='question')

    def connect(self):
        def _connect():
            try:
                url = urllib.parse.urljoin(self.url_entry.get(), "/api/tags")
                with urllib.request.urlopen(url, timeout=10) as response:
                    data = json.load(response)
                    models = [model["name"] for model in data["models"]]

                def _update_ui():
                    self.available_models = models
                    self.agent_model_combo["values"] = models
                    self.ask_llm_model_combo["values"] = models
                    if models:
                        self.agent_model_combo.set(models[0])
                        self.ask_llm_model_combo.set(models[0])
                    self._create_agent()
                    self.set_status("● Connected", "green")
                    self.append_text(f"[System] Connected to {self.url_entry.get()}\n", "system")
                    self.append_text(f"[System] Models: {', '.join(models)}\n", "system")
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
        if self.processing or not self.agent:
            return
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            return

        self.input_text.delete("1.0", tk.END)
        self.append_text(f"👤 You:\n{user_input}\n\n", "user")
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

                self.agent.chat(user_input, stream_callback=stream_cb, status_callback=status_cb, confirm_callback=confirm_cb)
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
