#!/usr/bin/env python3
"""
LLM Agent with Ollama - GUI 버전 (Streaming 지원)
"""

import os
import json
import re
import sys
import threading
import urllib.parse
import urllib.request
from typing import List, Dict, Any, Optional
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
# 도구 정의
# ============================================================================

def get_current_time() -> Dict[str, str]:
    """현재 시간"""
    now = datetime.now()
    return {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "current_week": now.strftime("%W"),
        "day_of_week": now.strftime("%A")
    }

def get_file(base_dir: str = ".", pattern: str = "*") -> Dict[str, Any]:
    """
    현재 디렉토리를 기준으로 재귀적으로 모든 파일을 상대 경로로 가져오는 함수.
    """
    try:
        base_path = Path(base_dir).resolve()
        
        # 디렉토리가 존재하는지 확인
        if not base_path.exists():
            return {
                "base_dir": base_dir,
                "files": [],
                "count": 0,
                "result": "failure",
                "error": f"Directory '{base_dir}' does not exist"
            }
        
        # 재귀적으로 모든 파일 찾기
        if pattern == "*":
            all_files = [
                str(f.relative_to(base_path))
                for f in base_path.rglob("*")
                if f.is_file()
            ]
        else:
            all_files = [
                str(f.relative_to(base_path))
                for f in base_path.rglob(pattern)
                if f.is_file()
            ]
        
        result = {
            "base_dir": str(base_path),
            "files": sorted(all_files),
            "count": len(all_files),
            "result": "success"
        }
        
        return result

    except Exception as e:
        return {
            "base_dir": base_dir,
            "files": [],
            "count": 0,
            "result": "failure",
            "error": str(e)
        }

def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    파일의 내용을 읽어오는 함수.
    """
    try:
        # 파일 존재 확인
        if not os.path.exists(file_path):
            return {
                "filename": file_path,
                "content": None,
                "result": "failure",
                "error": f"File '{file_path}' does not exist"
            }
        
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        file_size = os.path.getsize(file_path)
        
        result = {
            "filename": file_path,
            "content": content,
            "size": file_size,
            "lines": len(content.splitlines()),
            "result": "success"
        }
        
        return result
        
    except UnicodeDecodeError:
        # 바이너리 파일 처리
        try:
            with open(file_path, 'rb') as f:
                binary_content = f.read()
            
            return {
                "filename": file_path,
                "content": f"<binary file, {len(binary_content)} bytes>",
                "size": len(binary_content),
                "result": "success",
                "note": "Binary file, content not displayed"
            }
        except Exception as e:
            return {
                "filename": file_path,
                "content": None,
                "result": "failure",
                "error": str(e)
            }
    except Exception as e:
        return {
            "filename": file_path,
            "content": None,
            "result": "failure",
            "error": str(e)
        }

def write_file(file_path: str, content: object) -> Dict[str, Any]:
    """
    파일을 저장하는 함수.
    """
    try:
        # 디렉토리 경로 추출
        dir_path = os.path.dirname(file_path)
        
        # 디렉토리가 있고 비어있지 않으면 생성
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(content))
        
        file_size = os.path.getsize(file_path)
        
        result = {
            "filename": file_path,
            "size": file_size,
            "result": "success"
        }
        
        return result
        
    except Exception as e:
        return {
            "filename": file_path,
            "result": "failure",
            "error": str(e)
        }

def delete_file(file_path: str) -> Dict[str, Any]:
    """Deletes a file."""
    try:
        if not os.path.exists(file_path):
            return {
                "filename": file_path,
                "result": "failure",
                "error": f"File '{file_path}' does not exist"
            }

        os.remove(file_path)
        return {
                "filename": file_path,
                "result": "success",
        }

    except Exception as e:
        return {
            "filename": file_path,
            "result": "failure",
            "error": str(e)
        }

TOOLS = {
    "get_current_time": {
        "function": get_current_time,
        "description": "Get current date and time",
        "parameters": {}
    },
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
                "description": "Content to write to the file"
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
    }
}

class OllamaClient:
    """Ollama API 클라이언트 with streaming"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def chat_stream(self, model: str, messages: List[Dict[str, str]], 
                   temperature: float = 0.7, max_tokens: int = 4000,
                   callback=None) -> str:
        """
        ⭐ Streaming chat - 실시간으로 토큰 생성
        
        Args:
            callback: 각 토큰마다 호출될 함수 callback(token)
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            headers = {'Content-Type': 'application/json'}
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(f"{self.base_url}/api/chat", data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=1800) as response:
                if response.getcode() != 200:
                    raise Exception(f"Ollama API error: {response.getcode()}")
                full_content = ""
                for line in response:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if "message" in chunk:
                                content = chunk["message"].get("content", "")
                                full_content += content
                                
                                # ⭐ 콜백으로 실시간 전달
                                if callback:
                                    callback(content)
                            
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                return full_content
            
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")

# ============================================================================
# Agent
# ============================================================================

class OllamaAgent:
    """Ollama Agent"""
    
    def __init__(self, ollama_url: str, model: str):
        self.ollama = OllamaClient(ollama_url)
        self.model = model
        self.conversation_history = []
    
    def _create_system_prompt(self) -> str:
        tools_desc = []
        for name, info in TOOLS.items():
            desc = f"- {name}: {info['description']}"
            
            # 파라미터 상세 정보
            if info['parameters']:
                params = []
                for param_name, param_info in info['parameters'].items():
                    required = "required" if param_info['required'] else "optional"
                    param_type = param_info['type']
                    param_str = f"{param_name} ({param_type}, {required}"
                    
                    if 'default' in param_info:
                        param_str += f", default={param_info['default']}"
                    param_str += ")"
                    
                    if 'description' in param_info:
                        param_str += f" - {param_info['description']}"
                    
                    params.append(param_str)
                
                desc += "\n  Parameters:\n    " + "\n    ".join(params)
            else:
                desc += "\n  Parameters: None"
            
            tools_desc.append(desc)
        
        tools_text = "\n\n".join(tools_desc)
        
        return f"""You are a WiFi driver development assistant.

Available tools:
{tools_text}

When you need a tool, respond EXACTLY in this format:
TOOL_CALL: tool_name
ARGUMENTS: {{"param_name": "value"}}

Important guidelines:
- Use correct JSON types: strings in "quotes", numbers without quotes, booleans as true/false
- All required parameters must be provided
- Optional parameters can be omitted (defaults will be used)
- Follow the parameter descriptions carefully
- Only use tools that are directly relevant to the user's request
- If the request is not clear, ask the user to clarify before using tools
- Do not make assumptions about file paths or parameters - ask if uncertain
- When using tool outputs as inputs for other tools, preserve the exact values without modification

Language guideline:
- ALWAYS respond in the same language the user is using
- If user writes in Korean (한글), respond in Korean
- If user writes in English, respond in English
- If user writes in Japanese, respond in Japanese
- Maintain consistent language throughout the conversation unless user switches languages

Be concise and helpful."""

    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        tool_match = re.search(r'TOOL_CALL:\s*(\w+)', response, re.IGNORECASE)
        if not tool_match:
            return None
        
        tool_name = tool_match.group(1)
        args_match = re.search(r'ARGUMENTS:\s*({.*?})', response, re.DOTALL | re.IGNORECASE)
        
        arguments = {}
        if args_match:
            try:
                arguments = json.loads(args_match.group(1))
            except:
                pass
        return {"tool": tool_name, "arguments": arguments}
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name not in TOOLS:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            return TOOLS[tool_name]["function"](**arguments)
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, user_message: str, stream_callback=None, 
             status_callback=None, confirm_callback=None, max_iterations: int = 5,
             max_tokens: int = 4000) -> str:
        """
        Agent 메인 루프
        
        Args:
            stream_callback: 스트리밍 텍스트 콜백
            status_callback: 상태 메시지 콜백
            confirm_callback: 도구 실행 확인 콜백 - confirm_callback(tool_name, arguments) -> bool
            max_tokens: 최대 생성 토큰 수
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        for iteration in range(max_iterations):
            if status_callback:
                status_callback(f"🔄 Iteration {iteration + 1}")
            
            # ⭐ 매 iteration마다 최신 히스토리로 messages 생성
            messages = [
                {"role": "system", "content": self._create_system_prompt()}
            ] + self.conversation_history
            
            try:
                # LLM 호출 (streaming)
                llm_response = self.ollama.chat_stream(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens,  # 파라미터로 받은 값 사용
                    callback=stream_callback
                )
                
                # 도구 호출 확인
                tool_call = self._parse_tool_call(llm_response)
                
                if tool_call is None:
                    # 최종 응답
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": llm_response
                    })
                    if status_callback:
                        status_callback("✅ Complete")
                    return llm_response
                
                # 도구 실행
                tool_name = tool_call["tool"]
                arguments = tool_call["arguments"]
                
                # ⭐ 사용자에게 확인 요청
                if confirm_callback:
                    if status_callback:
                        status_callback(f"⏸️ Waiting for confirmation...")
                    
                    confirmed = confirm_callback(tool_name, arguments)
                    
                    if not confirmed:
                        # 사용자가 거부함
                        if status_callback:
                            status_callback("❌ Tool execution cancelled by user")
                        
                        return "Tool execution was cancelled by user. How would you like to proceed?"
                
                if status_callback:
                    status_callback(f"🔧 Calling tool: {tool_name}")
                
                tool_result = self._execute_tool(tool_name, arguments)
                
                if status_callback:
                    status_callback(f"📊 Tool completed")
                
                # 결과를 대화에 추가
                self.conversation_history.append({
                    "role": "assistant",
                    "content": llm_response
                })
                
                # ⭐ 도구 결과를 강조해서 전달
                tool_result_message = f"""TOOL_RESULT:
{json.dumps(tool_result, indent=2)}

IMPORTANT: If you need to use any values from this result (like file paths, names, IDs), 
copy them EXACTLY as shown above. Do not modify, translate, or alter them in any way."""
                
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_result_message
                })
                
                # 줄바꿈 추가
                if stream_callback:
                    stream_callback("\n\n")
                
            except Exception as e:
                if status_callback:
                    status_callback(f"❌ Error: {str(e)}")
                return f"Error: {str(e)}"
        
        return "Max iterations reached"
    
    def reset(self):
        self.conversation_history = []


# ============================================================================
# GUI
# ============================================================================

class AgentGUI:
    """GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Agent 🤖")
        self.root.geometry("1000x800")
        
        self.agent = None
        self.processing = False
        self.confirm_tool_execution = tk.BooleanVar(value=True)  # 기본값: 확인 요청
    
        self.setup_ui()
    
    def setup_ui(self):
        """UI 구성"""
        
        # ===== 설정 프레임 =====
        config_frame = ttk.LabelFrame(self.root, text="⚙️ Ollama 설정", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # URL
        ttk.Label(config_frame, text="URL:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.url_entry = ttk.Entry(config_frame, width=35)
        self.url_entry.insert(0, "http://192.168.0.30:11434")
        self.url_entry.grid(row=0, column=1, padx=5)
        
        # Model
        def _model_update(event):
            self.agent = OllamaAgent(self.url_entry.get(), self.model_entry.get())
        ttk.Label(config_frame, text="Model:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.model_var = tk.StringVar(value="llama3.1")
        self.model_entry = ttk.Combobox(config_frame, state="readonly", width=30)
        self.model_entry.bind("<<ComboboxSelected>>", _model_update)
        self.model_entry.grid(row=0, column=3, padx=5)
        
        # Connect button
        self.refresh_btn = ttk.Button(config_frame, text="Refresh", command=self.connect)
        self.refresh_btn.grid(row=0, column=4, padx=10)
        
        # Status
        self.status_label = ttk.Label(config_frame, text="● Not connected", foreground="red")
        self.status_label.grid(row=0, column=5, padx=10)

        # Max tokens 설정
        ttk.Label(config_frame, text="Max tokens:").grid(row=1, column=0, padx=5, sticky=tk.W, pady=5)
        self.max_tokens_var = tk.IntVar(value=4000)
        max_tokens_spinbox = ttk.Spinbox(
            config_frame, 
            from_=1000, 
            to=32000, 
            increment=1000,
            textvariable=self.max_tokens_var,
            width=10
        )
        max_tokens_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # 도구 실행 확인 옵션
        confirm_check = ttk.Checkbutton(
            config_frame, 
            text="Confirm tool execution",
            variable=self.confirm_tool_execution
        )
        confirm_check.grid(row=1, column=2, padx=10)

        # ===== 채팅 영역 =====
        chat_frame = ttk.LabelFrame(self.root, text="💬 Conversation", padding=10)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 채팅 디스플레이
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=100,
            height=30,
            font=("Consolas", 10),
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # 태그 설정
        self.chat_display.tag_config("user", foreground="#2196F3", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("assistant", foreground="#4CAF50", font=("Consolas", 10))
        self.chat_display.tag_config("system", foreground="#FF9800", font=("Consolas", 9, "italic"))
        self.chat_display.tag_config("tool", foreground="#9C27B0", font=("Consolas", 9))
        
        # ===== 입력 영역 =====
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X)
        
        self.input_text = tk.Text(input_frame, height=3, font=("Consolas", 10))
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_text.bind("<Control-Return>", lambda e: self.send_message())
        
        # 버튼 프레임
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
        """
        ⭐ 도구 실행 전 사용자 확인 다이얼로그
        
        Returns:
            True: 실행 승인, False: 실행 거부
        """
        # 확인 모드가 아니면 바로 승인
        if not self.confirm_tool_execution.get():
            return True
        
        # 인수를 보기 좋게 포맷팅
        args_formatted = json.dumps(arguments, indent=2, ensure_ascii=False)
        
        # 도구 설명 가져오기
        tool_desc = TOOLS.get(tool_name, {}).get('description', 'No description')
        
        message = f"""The agent wants to execute a tool:

Tool: {tool_name}
Description: {tool_desc}

Arguments:
{args_formatted}

Do you want to proceed?"""
        
        # 메시지 박스 표시 (Yes/No)
        result = messagebox.askyesno(
            "Confirm Tool Execution",
            message,
            icon='question'
        )
        
        return result

    def connect(self):
        """Ollama 연결"""

        def _connect():
            def _update_model_select():
                def _fetch_models(base_url) -> List[str]:
                    url = urllib.parse.urljoin(base_url, "/api/tags")
                    with urllib.request.urlopen(url) as response:
                        data = json.load(response)
                        models = [model["name"] for model in data["models"]]
                        return models

                def _show_error(text):
                    self.model_entry.set(text)
                    self.model_entry.config(foreground="red")
                    self.model_entry["values"] = []

                try:
                    models = _fetch_models(self.url_entry.get())
                    self.model_entry["values"] = models
                    if models:
                        self.model_entry.set(models[0])
                    else:
                        _show_error("You need download a model!")
                except Exception:  # noqa
                    _show_error("Error! Please check the host.")
            try:
                _update_model_select()
                self.set_status(f"● Connected", "green")
                self.append_text(f"[System] Connected to {self.url_entry.get()}\n", "system")
                self.send_btn.config(state=tk.NORMAL)
                self.reset_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.append_text(f"[System] ❌ Connection failed: {str(e)}\n", "system")
                self.set_status("● Connection failed", "red")
        
        threading.Thread(target=_connect, daemon=True).start()

    def send_message(self):
        """메시지 전송"""
        if self.processing or not self.agent:
            return
        
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            return
        
        # 입력창 클리어
        self.input_text.delete("1.0", tk.END)
        
        # 사용자 메시지 표시
        self.append_text(f"👤 You:\n", "user")
        self.append_text(f"{user_input}\n\n")
        
        # 버튼 비활성화
        self.processing = True
        self.send_btn.config(state=tk.DISABLED)
        self.input_text.config(state=tk.DISABLED)
        
        # Assistant 헤더
        self.append_text(f"🤖 Assistant:\n", "assistant")
        
        # 백그라운드에서 처리
        def _process():
            try:
                # 스트리밍 콜백
                def stream_cb(token):
                    self.append_text(token)
                
                # 상태 콜백
                def status_cb(status):
                    self.append_text(f"\n[{status}]\n", "system")
                
                # ⭐ 확인 콜백 (메인 스레드에서 실행)
                def confirm_cb(tool_name, arguments):
                    # threading.Event로 동기화
                    result_container = [None]
                    event = threading.Event()
                    
                    def _ask():
                        result_container[0] = self.confirm_tool_execution_dialog(tool_name, arguments)
                        event.set()
                    
                    self.root.after(0, _ask)
                    event.wait()  # 사용자 응답 대기
                    
                    return result_container[0]

                # Agent 실행
                self.agent.chat(
                    user_input,
                    stream_callback=stream_cb,
                    status_callback=status_cb,
                    confirm_callback=confirm_cb,
                    max_tokens=self.max_tokens_var.get()
                )
                
                self.append_text("\n\n" + "="*80 + "\n\n")
                
            except Exception as e:
                self.append_text(f"\n\n❌ Error: {str(e)}\n\n", "system")
            
            finally:
                # 버튼 재활성화
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
