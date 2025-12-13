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
# 통계 관리
# ============================================================================

class Statistics:
    """Agent 사용 통계 수집"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """통계 초기화"""
        self.start_time = datetime.now()
        self.total_messages = 0
        self.user_messages = 0
        self.assistant_messages = 0
        self.tool_calls = {}  # {tool_name: count}
        self.tool_successes = 0
        self.tool_failures = 0
        self.total_iterations = 0
        self.storage_keys_created = 0
    
    def record_message(self, role: str):
        """메시지 기록"""
        self.total_messages += 1
        if role == "user":
            self.user_messages += 1
        elif role == "assistant":
            self.assistant_messages += 1
    
    def record_tool_call(self, tool_name: str, success: bool):
        """도구 호출 기록"""
        if tool_name not in self.tool_calls:
            self.tool_calls[tool_name] = 0
        self.tool_calls[tool_name] += 1
        
        if success:
            self.tool_successes += 1
        else:
            self.tool_failures += 1
    
    def record_iteration(self):
        """반복 횟수 기록"""
        self.total_iterations += 1
    
    def record_storage_key(self):
        """저장소 키 생성 기록"""
        self.storage_keys_created += 1
    
    def get_uptime(self) -> str:
        """가동 시간 반환"""
        delta = datetime.now() - self.start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_summary(self) -> Dict[str, Any]:
        """통계 요약 반환"""
        return {
            "uptime": self.get_uptime(),
            "total_messages": self.total_messages,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "tool_calls": dict(self.tool_calls),
            "tool_successes": self.tool_successes,
            "tool_failures": self.tool_failures,
            "total_iterations": self.total_iterations,
            "storage_keys": self.storage_keys_created
        }


# ============================================================================
# Config 관리
# ============================================================================

class ConfigManager:
    """설정 저장/불러오기 관리"""
    
    def __init__(self, config_file: str = "agent_config.json"):
        self.config_file = config_file
        self.default_config = {
            "ollama_url": "http://192.168.0.30:11434",
            "agent_model": "",
            "agent_max_tokens": 4000,
            "ask_llm_model": "",
            "ask_llm_max_tokens": 4000,
            "confirm_tool_execution": True,
            "window_geometry": "1200x900"
        }
    
    def load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 기본값과 병합 (누락된 키 대비)
                    return {**self.default_config, **config}
            else:
                return self.default_config.copy()
        except Exception as e:
            print(f"Config load error: {e}")
            return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """설정 파일 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Config save error: {e}")
            return False


# ============================================================================
# 전역 변수 - 저장소
# ============================================================================
# 전역 변수로서, 도구 실행 결과를 저장하는 딕셔너리입니다.
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

        _OLLAMA_CLIENT = OllamaClient(ollama_url)
        _AGENT_MODEL = agent_model
        _AGENT_MAX_TOKENS = agent_max_tokens
        _ASK_LLM_MODEL = ask_llm_model
        _ASK_LLM_MAX_TOKENS = ask_llm_max_tokens

        self.conversation_history = []

    def update_settings(self, agent_model: str = None, agent_max_tokens: int = None,
                        ask_llm_model: str = None, ask_llm_max_tokens: int = None):
        """설정 업데이트"""
        global _AGENT_MODEL, _AGENT_MAX_TOKENS, _ASK_LLM_MODEL, _ASK_LLM_MAX_TOKENS
        
        if agent_model:
            _AGENT_MODEL = agent_model
        if agent_max_tokens:
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
             max_iterations: int = 10,
             stats_callback: Callable[[str, str, bool], None] = None) -> str:
        """⭐ JSON Mode Agent 메인 루프"""
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # 통계 기록
        if stats_callback:
            stats_callback("message", "user", True)

        for iteration in range(max_iterations):
            if status_callback:
                status_callback(f"🔄 Iteration {iteration + 1}")
            
            # 통계 기록
            if stats_callback:
                stats_callback("iteration", "", True)

            messages = [{"role": "system", "content": self._create_system_prompt()}] + self.conversation_history

            try:
                response = _OLLAMA_CLIENT.chat_json_mode(
                    model= _AGENT_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=_AGENT_MAX_TOKENS
                )
                
                if stream_callback:
                    stream_callback(json.dumps(response, indent=2, ensure_ascii=False))

                response_type = response.get("type", "unknown")

                if response_type == "response":
                    content = response.get("content", "")
                    self.conversation_history.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
                    
                    # 통계 기록
                    if stats_callback:
                        stats_callback("message", "assistant", True)
                    
                    if status_callback:
                        status_callback("✅ Complete")
                    return content

                elif response_type == "clarification":
                    question = response.get("question", "무엇을 도와드릴까요?")
                    self.conversation_history.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
                    
                    # 통계 기록
                    if stats_callback:
                        stats_callback("message", "assistant", True)
                    
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
                            
                            # 통계 기록 (실패)
                            if stats_callback:
                                stats_callback("tool", tool_name, False)
                            
                            return "Tool execution was cancelled by user."

                    if status_callback:
                        status_callback(f"🔧 Executing: {tool_name}")

                    tool_result = self._execute_tool(tool_name, arguments)
                    
                    # 통계 기록
                    success = tool_result.get("result") == "success"
                    if stats_callback:
                        stats_callback("tool", tool_name, success)

                    if store_as and success:
                        store_tool_result(store_as, tool_result)
                        
                        # 통계 기록
                        if stats_callback:
                            stats_callback("storage", store_as, True)
                        
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
        self.root.title("Agent 🤖")
        
        self.agent = None
        self.processing = False
        self.confirm_tool_execution = tk.BooleanVar(value=True)
        self.available_models = []
        
        # Config Manager
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # Statistics
        self.stats = Statistics()
        
        # 윈도우 크기 복원
        self.root.geometry(self.config.get("window_geometry", "1200x900"))

        self.setup_ui()
        
        # 설정 적용
        self.apply_config()

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

        # 옵션 및 설정 버튼
        options_frame = ttk.Frame(model_frame)
        options_frame.grid(row=2, column=0, columnspan=5, pady=(10,0), sticky=tk.EW)
        
        confirm_check = ttk.Checkbutton(options_frame, text="Tool 실행 전 확인", variable=self.confirm_tool_execution)
        confirm_check.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(options_frame, text="💾 Save Config", command=self.save_config).pack(side=tk.RIGHT, padx=2)
        ttk.Button(options_frame, text="📂 Load Config", command=self.load_config).pack(side=tk.RIGHT, padx=2)

        # ===== 메인 영역 - Notebook 탭 =====
        main_notebook = ttk.Notebook(self.root)
        main_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ===== Tab 1: Chat =====
        chat_tab = ttk.Frame(main_notebook)
        main_notebook.add(chat_tab, text="💬 Chat")

        self.chat_display = scrolledtext.ScrolledText(chat_tab, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # ===== Tab 2: History =====
        history_tab = ttk.Frame(main_notebook)
        main_notebook.add(history_tab, text="📜 History")

        # History 툴바
        history_toolbar = ttk.Frame(history_tab)
        history_toolbar.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(history_toolbar, text="Conversation History:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.history_count_label = ttk.Label(history_toolbar, text="(0 messages)", foreground="gray")
        self.history_count_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(history_toolbar, text="🔄 Refresh", command=self.refresh_history).pack(side=tk.RIGHT, padx=5)
        ttk.Button(history_toolbar, text="📋 Copy JSON", command=self.copy_history_json).pack(side=tk.RIGHT, padx=5)

        # History TreeView
        history_paned = ttk.PanedWindow(history_tab, orient=tk.HORIZONTAL)
        history_paned.pack(fill=tk.BOTH, expand=True)

        # 왼쪽: 메시지 리스트
        history_list_frame = ttk.Frame(history_paned)
        history_paned.add(history_list_frame, weight=1)

        self.history_tree = ttk.Treeview(history_list_frame, columns=("role", "preview"), show="tree headings")
        self.history_tree.heading("#0", text="#", anchor=tk.W)
        self.history_tree.heading("role", text="Role", anchor=tk.W)
        self.history_tree.heading("preview", text="Preview", anchor=tk.W)
        self.history_tree.column("#0", width=50, minwidth=30)
        self.history_tree.column("role", width=100, minwidth=80)
        self.history_tree.column("preview", width=400, minwidth=200)

        history_scroll_y = ttk.Scrollbar(history_list_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll_y.set)

        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_tree.bind("<<TreeviewSelect>>", self.on_history_select)

        # 오른쪽: 메시지 상세 내용
        history_detail_frame = ttk.LabelFrame(history_paned, text="📄 Message Detail", padding=5)
        history_paned.add(history_detail_frame, weight=2)

        self.history_detail = scrolledtext.ScrolledText(history_detail_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.history_detail.pack(fill=tk.BOTH, expand=True)

        # System Prompt 섹션
        system_prompt_frame = ttk.LabelFrame(history_tab, text="🔧 Current System Prompt", padding=5)
        system_prompt_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.system_prompt_display = scrolledtext.ScrolledText(system_prompt_frame, wrap=tk.WORD, height=10, font=("Consolas", 9), state=tk.DISABLED)
        self.system_prompt_display.pack(fill=tk.BOTH, expand=True)

        # ===== Tab 3: Storage =====
        storage_tab = ttk.Frame(main_notebook)
        main_notebook.add(storage_tab, text="📦 Storage")

        # ===== Tab 4: Debug Log =====
        debug_tab = ttk.Frame(main_notebook)
        main_notebook.add(debug_tab, text="🐛 Debug Log")

        # ===== Tab 5: Statistics =====
        stats_tab = ttk.Frame(main_notebook)
        main_notebook.add(stats_tab, text="📊 Statistics")

        # Statistics 툴바
        stats_toolbar = ttk.Frame(stats_tab)
        stats_toolbar.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(stats_toolbar, text="Usage Statistics", font=("", 12, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Button(stats_toolbar, text="🔄 Refresh", command=self.refresh_statistics).pack(side=tk.RIGHT, padx=5)
        ttk.Button(stats_toolbar, text="🗑️ Reset Stats", command=self.reset_statistics).pack(side=tk.RIGHT, padx=5)

        # Statistics 표시 영역 (2열 레이아웃)
        stats_container = ttk.Frame(stats_tab)
        stats_container.pack(fill=tk.BOTH, expand=True, padx=10)

        # 왼쪽 열: 기본 통계
        left_frame = ttk.LabelFrame(stats_container, text="📈 General Stats", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.stats_general = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, height=20, font=("Consolas", 10), state=tk.DISABLED)
        self.stats_general.pack(fill=tk.BOTH, expand=True)

        # 오른쪽 열: 도구 사용 통계
        right_frame = ttk.LabelFrame(stats_container, text="🔧 Tool Usage", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.stats_tools = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=20, font=("Consolas", 10), state=tk.DISABLED)
        self.stats_tools.pack(fill=tk.BOTH, expand=True)

        # Debug 툴바
        debug_toolbar = ttk.Frame(debug_tab)
        debug_toolbar.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(debug_toolbar, text="Execution Log:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.debug_count_label = ttk.Label(debug_toolbar, text="(0 entries)", foreground="gray")
        self.debug_count_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(debug_toolbar, text="🗑️ Clear Log", command=self.clear_debug_log).pack(side=tk.RIGHT, padx=5)
        ttk.Button(debug_toolbar, text="💾 Export Log", command=self.export_debug_log).pack(side=tk.RIGHT, padx=5)

        # Debug Log 표시 영역
        self.debug_log = scrolledtext.ScrolledText(debug_tab, wrap=tk.WORD, font=("Consolas", 9))
        self.debug_log.pack(fill=tk.BOTH, expand=True)

        # Debug 로그 태그 설정
        self.debug_log.tag_config("timestamp", foreground="#666666", font=("Consolas", 8))
        self.debug_log.tag_config("info", foreground="#2196F3")
        self.debug_log.tag_config("success", foreground="#4CAF50")
        self.debug_log.tag_config("warning", foreground="#FF9800")
        self.debug_log.tag_config("error", foreground="#F44336")
        self.debug_log.tag_config("tool", foreground="#9C27B0", font=("Consolas", 9, "bold"))

        # Debug log 엔트리 카운터
        self.debug_log_count = 0

        # Storage TreeView + 상세정보 (기존 코드 이동)
        storage_paned = ttk.PanedWindow(storage_tab, orient=tk.HORIZONTAL)
        storage_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 왼쪽: Storage TreeView
        storage_tree_container = ttk.LabelFrame(storage_paned, text="🗂️ Storage Tree", padding=5)
        storage_paned.add(storage_tree_container, weight=1)

        # TreeView와 상세정보를 좌우로 나눔
        storage_tree_paned = ttk.PanedWindow(storage_tree_container, orient=tk.VERTICAL)
        storage_tree_paned.pack(fill=tk.BOTH, expand=True)

        # 위쪽: TreeView
        tree_frame = ttk.Frame(storage_tree_paned)
        storage_tree_paned.add(tree_frame, weight=1)

        self.storage_tree = ttk.Treeview(tree_frame, show="tree headings", columns=("value",))
        self.storage_tree.heading("#0", text="Key", anchor=tk.W)
        self.storage_tree.heading("value", text="Value", anchor=tk.W)
        self.storage_tree.column("#0", width=150, minwidth=80)
        self.storage_tree.column("value", width=200, minwidth=100)

        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.storage_tree.yview)
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.storage_tree.xview)
        self.storage_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

        self.storage_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        # TreeView 클릭 이벤트
        self.storage_tree.bind("<<TreeviewSelect>>", self.on_storage_item_select)

        # 아래쪽: 상세정보 패널
        detail_frame = ttk.LabelFrame(storage_tree_paned, text="📝 Detail View", padding=5)
        storage_tree_paned.add(detail_frame, weight=2)

        # 경로 표시
        self.detail_path_label = ttk.Label(detail_frame, text="Select an item to view details", font=("Consolas", 9), foreground="gray")
        self.detail_path_label.pack(anchor=tk.W, pady=(0, 5))

        # 값 표시 및 편집 영역
        value_frame = ttk.Frame(detail_frame)
        value_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(value_frame, text="Value:", font=("", 9, "bold")).pack(anchor=tk.W)
        
        self.detail_text = scrolledtext.ScrolledText(value_frame, wrap=tk.WORD, height=10, font=("Consolas", 9))
        self.detail_text.pack(fill=tk.BOTH, expand=True, pady=(2, 5))

        # 버튼 프레임
        btn_frame = ttk.Frame(detail_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.save_value_btn = ttk.Button(btn_frame, text="💾 Save", command=self.save_storage_value, state=tk.DISABLED)
        self.save_value_btn.pack(side=tk.LEFT, padx=2)

        self.delete_value_btn = ttk.Button(btn_frame, text="🗑️ Delete", command=self.delete_storage_value, state=tk.DISABLED)
        self.delete_value_btn.pack(side=tk.LEFT, padx=2)

        self.cancel_btn = ttk.Button(btn_frame, text="↩️ Cancel", command=self.cancel_edit, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=2)

        # 전체 새로고침 버튼
        refresh_storage_btn = ttk.Button(storage_tree_container, text="🔄 Refresh All", command=self.refresh_storage_tree)
        refresh_storage_btn.pack(fill=tk.X, pady=5)

        # 현재 선택된 항목 추적
        self.selected_storage_path = None
        self.original_value = None

        # ===== 채팅 입력 영역 (Chat 탭에 추가) =====
        input_frame = ttk.Frame(chat_tab)
        input_frame.pack(fill=tk.X, pady=(5, 0))

        self.input_text = tk.Text(input_frame, height=3, font=("Consolas", 10))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_text.bind("<Control-Return>", lambda e: self.send_message())

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.send_btn = ttk.Button(btn_frame, text="Send\n(Ctrl+Enter)", command=self.send_message, state=tk.DISABLED)
        self.send_btn.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.reset_btn = ttk.Button(btn_frame, text="Reset\nChat", command=self.reset_chat, state=tk.DISABLED)
        self.reset_btn.pack(fill=tk.BOTH, expand=True)

        # 채팅 디스플레이 태그 설정
        self.chat_display.tag_config("user", foreground="#2196F3", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("assistant", foreground="#4CAF50", font=("Consolas", 10))
        self.chat_display.tag_config("system", foreground="#FF9800", font=("Consolas", 9, "italic"))

        # History 디스플레이 태그 설정
        self.history_detail.tag_config("role", foreground="#1976D2", font=("Consolas", 9, "bold"))
        self.history_detail.tag_config("content", foreground="#333333", font=("Consolas", 9))
        self.history_detail.tag_config("json", foreground="#00796B", font=("Consolas", 9))

        # ===== 도구 정보 =====
        tools_frame = ttk.LabelFrame(self.root, text="🔧 Available Tools", padding=10)
        tools_frame.pack(fill=tk.X, padx=10, pady=5)

        tools_text = "  •  ".join([f"{name}" for name in TOOLS.keys()])
        ttk.Label(tools_frame, text=tools_text, font=("Consolas", 9)).pack()

    def refresh_statistics(self):
        """통계 표시 갱신"""
        summary = self.stats.get_summary()
        
        # 일반 통계
        self.stats_general.config(state=tk.NORMAL)
        self.stats_general.delete("1.0", tk.END)
        
        general_text = f"""
⏱️  Uptime: {summary['uptime']}

💬 Messages:
   Total: {summary['total_messages']}
   User: {summary['user_messages']}
   Assistant: {summary['assistant_messages']}

🔄 Iterations: {summary['total_iterations']}

💾 Storage Keys Created: {summary['storage_keys']}

🔧 Tool Execution:
   Successes: {summary['tool_successes']}
   Failures: {summary['tool_failures']}
   Total: {summary['tool_successes'] + summary['tool_failures']}
"""
        
        if summary['tool_successes'] + summary['tool_failures'] > 0:
            success_rate = (summary['tool_successes'] / (summary['tool_successes'] + summary['tool_failures'])) * 100
            general_text += f"   Success Rate: {success_rate:.1f}%\n"
        
        self.stats_general.insert("1.0", general_text)
        self.stats_general.config(state=tk.DISABLED)
        
        # 도구 사용 통계
        self.stats_tools.config(state=tk.NORMAL)
        self.stats_tools.delete("1.0", tk.END)
        
        if summary['tool_calls']:
            tools_text = "Tool Call Count:\n\n"
            sorted_tools = sorted(summary['tool_calls'].items(), key=lambda x: x[1], reverse=True)
            
            max_count = max(summary['tool_calls'].values())
            for tool_name, count in sorted_tools:
                # 간단한 바 차트
                bar_length = int((count / max_count) * 30)
                bar = "█" * bar_length
                tools_text += f"{tool_name:20} {count:3}  {bar}\n"
            
            self.stats_tools.insert("1.0", tools_text)
        else:
            self.stats_tools.insert("1.0", "No tool calls recorded yet")
        
        self.stats_tools.config(state=tk.DISABLED)

    def reset_statistics(self):
        """통계 초기화"""
        if messagebox.askyesno("Confirm Reset", "Reset all statistics?"):
            self.stats.reset()
            self.refresh_statistics()
            self.log_debug("Statistics reset", "info")

    def log_debug(self, message: str, level: str = "info"):
        """
        Debug 로그에 메시지 추가
        level: info, success, warning, error, tool
        """
        def _log():
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            self.debug_log.config(state=tk.NORMAL)
            self.debug_log.insert(tk.END, f"[{timestamp}] ", "timestamp")
            
            # Level 표시
            level_map = {
                "info": ("ℹ️ INFO", "info"),
                "success": ("✅ SUCCESS", "success"),
                "warning": ("⚠️ WARNING", "warning"),
                "error": ("❌ ERROR", "error"),
                "tool": ("🔧 TOOL", "tool")
            }
            
            level_text, level_tag = level_map.get(level, ("INFO", "info"))
            self.debug_log.insert(tk.END, f"{level_text}: ", level_tag)
            self.debug_log.insert(tk.END, f"{message}\n")
            
            self.debug_log.see(tk.END)
            self.debug_log.config(state=tk.DISABLED)
            
            self.debug_log_count += 1
            self.debug_count_label.config(text=f"({self.debug_log_count} entries)")
        
        self.root.after(0, _log)

    def clear_debug_log(self):
        """Debug 로그 초기화"""
        self.debug_log.config(state=tk.NORMAL)
        self.debug_log.delete("1.0", tk.END)
        self.debug_log.config(state=tk.DISABLED)
        self.debug_log_count = 0
        self.debug_count_label.config(text="(0 entries)")
        self.log_debug("Debug log cleared", "info")

    def export_debug_log(self):
        """Debug 로그를 파일로 내보내기"""
        try:
            log_content = self.debug_log.get("1.0", tk.END)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_log_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(log_content)
            
            messagebox.showinfo("Success", f"Debug log exported to:\n{filename}")
            self.log_debug(f"Log exported to {filename}", "success")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export log:\n{str(e)}")
            self.log_debug(f"Export failed: {str(e)}", "error")

    def apply_config(self):
        """저장된 설정 적용"""
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, self.config.get("ollama_url", "http://192.168.0.30:11434"))
        
        self.agent_tokens_var.set(self.config.get("agent_max_tokens", 4000))
        self.ask_llm_tokens_var.set(self.config.get("ask_llm_max_tokens", 4000))
        self.confirm_tool_execution.set(self.config.get("confirm_tool_execution", True))
        
        # 모델은 연결 후 설정됨
        self.append_text("[System] Config loaded from agent_config.json\n", "system")

    def save_config(self):
        """현재 설정을 config.json에 저장"""
        try:
            # 현재 윈도우 크기 저장
            geometry = self.root.geometry()
            
            config = {
                "ollama_url": self.url_entry.get(),
                "agent_model": self.agent_model_var.get(),
                "agent_max_tokens": self.agent_tokens_var.get(),
                "ask_llm_model": self.ask_llm_model_var.get(),
                "ask_llm_max_tokens": self.ask_llm_tokens_var.get(),
                "confirm_tool_execution": self.confirm_tool_execution.get(),
                "window_geometry": geometry
            }
            
            if self.config_manager.save_config(config):
                self.config = config
                self.append_text("[System] 💾 Config saved to agent_config.json\n", "system")
                messagebox.showinfo("Success", "Configuration saved successfully!")
            else:
                messagebox.showerror("Error", "Failed to save configuration")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{str(e)}")

    def load_config(self):
        """config.json에서 설정 불러오기"""
        try:
            self.config = self.config_manager.load_config()
            self.apply_config()
            self._update_settings_label()
            messagebox.showinfo("Success", "Configuration loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{str(e)}")

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

    def on_storage_item_select(self, event):
        """TreeView 항목 선택 시 호출"""
        selection = self.storage_tree.selection()
        if not selection:
            self._clear_detail_view()
            return

        item_id = selection[0]
        
        # 경로 구성 (부모부터 추적)
        path_parts = []
        current_id = item_id
        
        while current_id:
            item_text = self.storage_tree.item(current_id, "text")
            path_parts.insert(0, item_text)
            current_id = self.storage_tree.parent(current_id)
        
        if not path_parts:
            self._clear_detail_view()
            return
        
        # 첫 번째는 $key 형태
        key = path_parts[0].lstrip('$')
        
        # Storage에서 값 가져오기
        try:
            value = TOOL_RESULT_STORAGE.get(key)
            if value is None:
                self._clear_detail_view()
                return
            
            # 하위 필드 접근
            for field in path_parts[1:]:
                if field == "(value)":  # 리프 노드
                    continue
                if isinstance(value, dict):
                    value = value.get(field)
                elif isinstance(value, list):
                    # 리스트 인덱스 추출 (예: "item[0]")
                    import re
                    match = re.match(r'.*\[(\d+)\]', field)
                    if match:
                        idx = int(match.group(1))
                        value = value[idx]
                else:
                    break
            
            # 경로 표시
            full_path = ".".join(path_parts) if len(path_parts) > 1 else path_parts[0]
            self.detail_path_label.config(text=f"Path: {full_path}", foreground="blue")
            
            # 값 표시
            self._display_value(value, full_path)
            
            # 버튼 활성화
            self.save_value_btn.config(state=tk.NORMAL)
            self.delete_value_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.NORMAL)
            
            # 현재 선택 정보 저장
            self.selected_storage_path = (key, path_parts[1:] if len(path_parts) > 1 else [])
            self.original_value = value
            
        except Exception as e:
            self._clear_detail_view()
            self.detail_path_label.config(text=f"Error: {str(e)}", foreground="red")

    def _display_value(self, value, path):
        """값을 상세 패널에 표시"""
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        
        if isinstance(value, (dict, list)):
            # JSON 형태로 표시
            formatted = json.dumps(value, indent=2, ensure_ascii=False)
            self.detail_text.insert("1.0", formatted)
        else:
            # 문자열이나 기타 타입
            self.detail_text.insert("1.0", str(value))
        
        self.detail_text.config(state=tk.NORMAL)  # 편집 가능하게 유지

    def _clear_detail_view(self):
        """상세 패널 초기화"""
        self.detail_path_label.config(text="Select an item to view details", foreground="gray")
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.config(state=tk.DISABLED)
        
        self.save_value_btn.config(state=tk.DISABLED)
        self.delete_value_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.DISABLED)
        
        self.selected_storage_path = None
        self.original_value = None

    def save_storage_value(self):
        """편집된 값을 저장"""
        if not self.selected_storage_path:
            return
        
        key, field_path = self.selected_storage_path
        new_value_text = self.detail_text.get("1.0", tk.END).strip()
        
        try:
            # JSON 파싱 시도
            if new_value_text.startswith('{') or new_value_text.startswith('['):
                new_value = json.loads(new_value_text)
            else:
                # 일반 문자열로 처리
                new_value = new_value_text
            
            # 저장소 업데이트
            if not field_path:
                # 최상위 키 전체 교체
                TOOL_RESULT_STORAGE[key] = new_value
            else:
                # 중첩 필드 업데이트
                data = TOOL_RESULT_STORAGE[key]
                current = data
                
                for i, field in enumerate(field_path[:-1]):
                    if field == "(value)":
                        continue
                    if isinstance(current, dict):
                        current = current[field]
                    elif isinstance(current, list):
                        import re
                        match = re.match(r'.*\[(\d+)\]', field)
                        if match:
                            idx = int(match.group(1))
                            current = current[idx]
                
                # 마지막 필드에 값 설정
                last_field = field_path[-1]
                if last_field != "(value)":
                    if isinstance(current, dict):
                        current[last_field] = new_value
                    elif isinstance(current, list):
                        import re
                        match = re.match(r'.*\[(\d+)\]', last_field)
                        if match:
                            idx = int(match.group(1))
                            current[idx] = new_value
            
            # UI 업데이트
            self.refresh_storage_tree()
            self.append_text(f"[System] 💾 Saved: ${key}" + ("." + ".".join(field_path) if field_path else "") + "\n", "system")
            messagebox.showinfo("Success", "Value saved successfully!")
            
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Invalid JSON format:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save value:\n{str(e)}")

    def delete_storage_value(self):
        """선택된 값 삭제"""
        if not self.selected_storage_path:
            return
        
        key, field_path = self.selected_storage_path
        
        # 확인 대화상자
        path_str = f"${key}" + ("." + ".".join(field_path) if field_path else "")
        if not messagebox.askyesno("Confirm Delete", f"Delete this item?\n\n{path_str}"):
            return
        
        try:
            if not field_path:
                # 최상위 키 전체 삭제
                del TOOL_RESULT_STORAGE[key]
            else:
                # 중첩 필드 삭제
                data = TOOL_RESULT_STORAGE[key]
                current = data
                
                for i, field in enumerate(field_path[:-1]):
                    if field == "(value)":
                        continue
                    if isinstance(current, dict):
                        current = current[field]
                    elif isinstance(current, list):
                        import re
                        match = re.match(r'.*\[(\d+)\]', field)
                        if match:
                            idx = int(match.group(1))
                            current = current[idx]
                
                # 마지막 필드 삭제
                last_field = field_path[-1]
                if last_field != "(value)":
                    if isinstance(current, dict):
                        del current[last_field]
                    elif isinstance(current, list):
                        import re
                        match = re.match(r'.*\[(\d+)\]', last_field)
                        if match:
                            idx = int(match.group(1))
                            current.pop(idx)
            
            # UI 업데이트
            self._clear_detail_view()
            self.refresh_storage_tree()
            self.append_text(f"[System] 🗑️ Deleted: {path_str}\n", "system")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete value:\n{str(e)}")

    def cancel_edit(self):
        """편집 취소 (원래 값으로 복원)"""
        if self.original_value is not None:
            self._display_value(self.original_value, "")
            self.append_text("[System] ↩️ Edit cancelled\n", "system")

    def refresh_history(self):
        """대화 히스토리 TreeView 갱신"""
        # 기존 항목 제거
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        if not self.agent:
            self.history_count_label.config(text="(No agent)")
            return
        
        history = self.agent.conversation_history
        self.history_count_label.config(text=f"({len(history)} messages)")
        
        # 각 메시지를 TreeView에 추가
        for idx, msg in enumerate(history):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Preview 생성 (첫 100자)
            if isinstance(content, str):
                preview = content[:100].replace("\n", " ")
                if len(content) > 100:
                    preview += "..."
            else:
                preview = str(content)[:100]
            
            # Role 색상 매핑
            role_colors = {
                "system": "#9C27B0",
                "user": "#2196F3",
                "assistant": "#4CAF50"
            }
            
            item_id = self.history_tree.insert("", tk.END, text=f"{idx+1}", values=(role, preview))
            
            # Role별 태그 설정
            if role in role_colors:
                self.history_tree.item(item_id, tags=(role,))
        
        # 태그 색상 설정
        for role, color in {"system": "#9C27B0", "user": "#2196F3", "assistant": "#4CAF50"}.items():
            self.history_tree.tag_configure(role, foreground=color)
        
        # System Prompt 업데이트
        self.update_system_prompt_display()

    def on_history_select(self, event):
        """History TreeView 항목 선택 시 호출"""
        selection = self.history_tree.selection()
        if not selection:
            return
        
        item_id = selection[0]
        idx_text = self.history_tree.item(item_id, "text")
        
        try:
            idx = int(idx_text) - 1
            if not self.agent or idx >= len(self.agent.conversation_history):
                return
            
            msg = self.agent.conversation_history[idx]
            
            # 상세 내용 표시
            self.history_detail.config(state=tk.NORMAL)
            self.history_detail.delete("1.0", tk.END)
            
            # Role 표시
            role = msg.get("role", "unknown")
            self.history_detail.insert(tk.END, f"Role: ", "role")
            self.history_detail.insert(tk.END, f"{role}\n\n", "content")
            
            # Content 표시
            content = msg.get("content", "")
            self.history_detail.insert(tk.END, "Content:\n", "role")
            
            # JSON 파싱 시도
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                    self.history_detail.insert(tk.END, formatted, "json")
                except:
                    self.history_detail.insert(tk.END, content, "content")
            else:
                self.history_detail.insert(tk.END, str(content), "content")
            
            self.history_detail.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"History select error: {e}")

    def update_system_prompt_display(self):
        """현재 System Prompt 표시"""
        if not self.agent:
            self.system_prompt_display.config(state=tk.NORMAL)
            self.system_prompt_display.delete("1.0", tk.END)
            self.system_prompt_display.insert("1.0", "No agent connected")
            self.system_prompt_display.config(state=tk.DISABLED)
            return
        
        try:
            prompt = self.agent._create_system_prompt()
            self.system_prompt_display.config(state=tk.NORMAL)
            self.system_prompt_display.delete("1.0", tk.END)
            self.system_prompt_display.insert("1.0", prompt)
            self.system_prompt_display.config(state=tk.DISABLED)
        except Exception as e:
            print(f"System prompt update error: {e}")

    def copy_history_json(self):
        """대화 히스토리를 JSON으로 복사"""
        if not self.agent:
            messagebox.showwarning("Warning", "No conversation history")
            return
        
        try:
            history_json = json.dumps(self.agent.conversation_history, indent=2, ensure_ascii=False)
            self.root.clipboard_clear()
            self.root.clipboard_append(history_json)
            messagebox.showinfo("Success", "History copied to clipboard as JSON!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy history:\n{str(e)}")

    def refresh_storage_tree(self):
        """Storage TreeView 새로고침"""
        # 기존 항목 제거
        for item in self.storage_tree.get_children():
            self.storage_tree.delete(item)
        
        # Storage 내용 표시
        for key, value in TOOL_RESULT_STORAGE.items():
            parent_id = self.storage_tree.insert("", tk.END, text=f"${key}", open=True)
            self._add_tree_items(parent_id, value)

    def _add_tree_items(self, parent, value):
        """재귀적으로 트리 항목 추가"""
        if isinstance(value, dict):
            for field, field_value in value.items():
                if isinstance(field_value, (dict, list)):
                    child_id = self.storage_tree.insert(parent, tk.END, text=field, values=("",))
                    self._add_tree_items(child_id, field_value)
                else:
                    display_value = self._format_tree_value(field_value)
                    self.storage_tree.insert(parent, tk.END, text=field, values=(display_value,))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    child_id = self.storage_tree.insert(parent, tk.END, text=f"[{i}]", values=("",))
                    self._add_tree_items(child_id, item)
                else:
                    display_value = self._format_tree_value(item)
                    self.storage_tree.insert(parent, tk.END, text=f"[{i}]", values=(display_value,))
        else:
            display_value = self._format_tree_value(value)
            self.storage_tree.insert(parent, tk.END, text="(value)", values=(display_value,))

    def _format_tree_value(self, value: Any) -> str:
        """트리뷰에 표시할 값 포맷팅"""
        if isinstance(value, str):
            if len(value) > 50:
                return f"<{len(value)} chars>"
            return value
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
                self.log_debug(f"Connecting to {self.url_entry.get()}...", "info")
                url = urllib.parse.urljoin(self.url_entry.get(), "/api/tags")
                with urllib.request.urlopen(url, timeout=10) as response:
                    data = json.load(response)
                    models = [model["name"] for model in data["models"]]

                def _update_ui():
                    self.available_models = models
                    self.agent_model_combo["values"] = models
                    self.ask_llm_model_combo["values"] = models
                    
                    # 저장된 설정이 있으면 적용, 없으면 첫 번째 모델 선택
                    saved_agent_model = self.config.get("agent_model", "")
                    saved_ask_llm_model = self.config.get("ask_llm_model", "")
                    
                    if saved_agent_model in models:
                        self.agent_model_combo.set(saved_agent_model)
                    elif models:
                        self.agent_model_combo.set(models[0])
                    
                    if saved_ask_llm_model in models:
                        self.ask_llm_model_combo.set(saved_ask_llm_model)
                    elif models:
                        self.ask_llm_model_combo.set(models[0])
                    
                    self._create_agent()
                    self.set_status("● Connected", "green")
                    self.append_text(f"[System] Connected to {self.url_entry.get()}\n", "system")
                    self.append_text(f"[System] Models: {', '.join(models)}\n", "system")
                    self.send_btn.config(state=tk.NORMAL)
                    self.reset_btn.config(state=tk.NORMAL)
                    
                    self.log_debug(f"Connected successfully. Found {len(models)} models", "success")
                self.root.after(0, _update_ui)

            except Exception as e:
                def _show_error():
                    self.set_status("● Connection failed", "red")
                    self.append_text(f"[System] ❌ Connection failed: {str(e)}\n", "system")
                    self.log_debug(f"Connection failed: {str(e)}", "error")
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
                    self.root.after(0, self.refresh_history)
                    # Debug 로그 추가
                    if "Executing:" in status:
                        self.log_debug(status, "tool")
                    elif "Error" in status or "❌" in status:
                        self.log_debug(status, "error")
                    elif "Complete" in status or "✅" in status:
                        self.log_debug(status, "success")
                    elif "Warning" in status or "⚠️" in status:
                        self.log_debug(status, "warning")
                    else:
                        self.log_debug(status, "info")
                def confirm_cb(tool_name, arguments):
                    result_container = [None]
                    event = threading.Event()
                    def _ask():
                        result_container[0] = self.confirm_tool_execution_dialog(tool_name, arguments)
                        event.set()
                    self.root.after(0, _ask)
                    event.wait()
                    
                    # Debug 로그
                    if result_container[0]:
                        self.log_debug(f"Tool '{tool_name}' confirmed by user", "info")
                    else:
                        self.log_debug(f"Tool '{tool_name}' cancelled by user", "warning")
                    
                    return result_container[0]

                def stats_cb(event_type: str, detail: str, success: bool):
                    """통계 수집 콜백"""
                    if event_type == "message":
                        self.stats.record_message(detail)
                    elif event_type == "tool":
                        self.stats.record_tool_call(detail, success)
                    elif event_type == "iteration":
                        self.stats.record_iteration()
                    elif event_type == "storage":
                        self.stats.record_storage_key()

                self.log_debug(f"Starting chat with query: {user_input[:100]}...", "info")
                self.agent.chat(user_input, stream_callback=stream_cb, status_callback=status_cb, 
                               confirm_callback=confirm_cb, stats_callback=stats_cb)
                self.append_text("\n\n" + "=" * 80 + "\n\n")
                self.log_debug("Chat completed successfully", "success")
                
                # 통계 갱신
                self.root.after(0, self.refresh_statistics)

            except Exception as e:
                self.append_text(f"\n\n❌ Error: {str(e)}\n\n", "system")
                self.log_debug(f"Chat error: {str(e)}", "error")
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
        self.refresh_history()
        self.append_text("[System] Chat reset! 🔄\n\n", "system")


# ============================================================================
# Main
# ============================================================================

def main():
    root = tk.Tk()
    app = AgentGUI(root)
    app.reset_chat()
    
    # 종료 시 자동으로 설정 저장
    def on_closing():
        try:
            geometry = root.geometry()
            app.config["window_geometry"] = geometry
            app.config_manager.save_config(app.config)
        except:
            pass
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
