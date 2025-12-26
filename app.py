"""
Multi-Agent Chatbot UI (Updated)
================================
Gradio 기반 웹 인터페이스 - 실시간 Streaming 개선

변경사항:
- chat_stream에서 실시간으로 진행 상황 표시
- 각 Step 결과를 즉시 채팅창에 반영
- 상태 표시줄 실시간 업데이트
"""

import os
import sys
import warnings
from typing import List, Generator, Optional, Dict

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.shared_storage import SharedStorage
from core.base_tool import ToolRegistry
from core.orchestrator import Orchestrator, StepType
from core.workspace_manager import WorkspaceManager, ConfigManager
from tools.file_tool import FileTool
from tools.llm_tool import LLMTool


# =============================================================================
# Global State
# =============================================================================

class AppState:
    """애플리케이션 상태 관리"""
    
    def __init__(self, workspace_path: str = os.path.join(".", "workspace")):
        self.workspace_path = workspace_path
        
        self.workspace_manager = WorkspaceManager(workspace_path)
        self.config_manager = ConfigManager(os.path.join(workspace_path, "config"))
        
        self.llm_config = self.config_manager.load_llm_config()
        
        self.storage = SharedStorage(debug_enabled=True)
        self.registry = ToolRegistry(debug_enabled=True)
        
        self._setup_tools()
        
        self.orchestrator = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            llm_config=self.llm_config,
            max_steps=self.llm_config.max_steps,
            debug_enabled=True
        )
        
        self.chat_history: List[Dict] = []
        self.system_prompt_history: List[Dict] = []
        self.ollama_connected: bool = False
        self.available_models: List[Dict] = []
    
    def _setup_tools(self):
        """Tools 설정"""
        files_dir = os.path.join(self.workspace_path, "files")
        os.makedirs(files_dir, exist_ok=True)

        self.registry.register(FileTool(base_path=files_dir, debug_enabled=True))
        self.registry.register(LLMTool(base_path=files_dir, debug_enabled=True, use_mock=False))
    
    def update_llm_config(self, **kwargs):
        """LLM 설정 업데이트 및 저장"""
        self.llm_config.update(**kwargs)
        self.orchestrator.update_llm_config(**kwargs)
        
        if 'max_steps' in kwargs:
            self.orchestrator.max_steps = kwargs['max_steps']
        
        self.config_manager.save_llm_config(self.llm_config)
    
    def fetch_ollama_models(self, base_url: str = None) -> tuple[bool, List[Dict]]:
        """Ollama 서버에서 모델 목록 조회"""
        if base_url is None:
            base_url = self.llm_config.base_url
        
        try:
            import requests
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model in data.get("models", []):
                name = model.get("name", "unknown")
                size_bytes = model.get("size", 0)
                details = model.get("details", {})
                
                if size_bytes >= 1024 * 1024 * 1024:
                    size_str = f"{size_bytes / (1024**3):.1f}GB"
                else:
                    size_str = f"{size_bytes / (1024**2):.0f}MB"
                
                family = details.get("family", "unknown")
                param_size = details.get("parameter_size", "")
                quantization = details.get("quantization_level", "")
                
                features = []
                if param_size:
                    features.append(param_size)
                if quantization:
                    features.append(quantization)
                if family and family != "unknown":
                    features.append(family)
                
                models.append({
                    "name": name,
                    "size": size_str,
                    "features": ", ".join(features) if features else "N/A",
                    "display": f"{name} ({', '.join(features) if features else size_str})"
                })
            
            self.ollama_connected = True
            self.available_models = models
            return True, models
            
        except Exception as e:
            print(f"[WARN] Ollama 모델 목록 조회 실패: {e}")
            self.ollama_connected = False
            self.available_models = []
            return False, []


app_state: Optional[AppState] = None


def get_app_state() -> AppState:
    global app_state
    if app_state is None:
        app_state = AppState()
    return app_state


# =============================================================================
# Chat Functions (Updated - Real-time Streaming)
# =============================================================================

def format_tool_result(content: str, tool_name: str, action: str, max_length: int = 300) -> str:
    """Tool 결과를 포맷팅 (길이 제한)"""
    content_str = str(content)
    
    if len(content_str) > max_length:
        content_str = content_str[:max_length] + f"\n... ({len(str(content))} chars total)"
    
    return f"```\n{content_str}\n```"


def build_streaming_response(
    plan_content: str,
    step_outputs: List[Dict],
    current_thinking: str = "",
    final_answer: str = ""
) -> str:
    """
    실시간 스트리밍 응답 구성
    
    Args:
        plan_content: 실행 계획
        step_outputs: 완료된 step들의 결과
        current_thinking: 현재 진행 중인 생각/상태
        final_answer: 최종 답변 (있을 경우)
    """
    parts = []
    
    # 1. 실행 계획 (접이식)
    if plan_content:
        parts.append(f"<details>\n<summary>📋 실행 계획</summary>\n\n{plan_content}\n</details>")
    
    # 2. 완료된 Steps (각각 접이식)
    if step_outputs:
        for step in step_outputs:
            header = step.get("header", "")
            result = step.get("result", "")
            if header and result:
                parts.append(header + result)
    
    # 3. 현재 진행 상황 (실시간)
    if current_thinking and not final_answer:
        parts.append(f"\n💭 *{current_thinking}*")
    
    # 4. 최종 답변
    if final_answer:
        if parts:
            parts.append("\n---\n")
        parts.append(final_answer)
    
    return "\n\n".join(filter(None, parts))


def chat_stream(message: str, history: List[Dict]) -> Generator[List[Dict], None, None]:
    """
    채팅 메시지 처리 (Real-time Streaming)
    
    각 Step의 진행 상황을 실시간으로 채팅창에 표시합니다.
    """
    state = get_app_state()
    
    state.orchestrator._stopped = False

    if not message.strip():
        yield history, gr.update(interactive=False), gr.update(interactive=True)
        return
    
    # 파일 목록을 context로 준비 (orchestrator에 전달)
    files_context = state.workspace_manager.get_files_for_prompt()

    # 새 대화 추가 (user 메시지)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]

    # 상태 변수
    plan_content = ""
    step_outputs = []
    current_thinking = ""
    final_answer = ""
    current_step_info = {}
    was_stopped = False

    # 초기 상태 표시
    yield history, gr.update(interactive=False), gr.update(interactive=True)

    try:
        for step in state.orchestrator.run_stream(message, files_context):
            # ===== 중지 체크 =====
            if state.orchestrator._stopped:
                was_stopped = True
                break
            
            # ===== Planning Phase =====
            if step.type == StepType.PLANNING:
                current_thinking = step.content
                history[-1]["content"] = f"📋 *{step.content}*"
                yield history, gr.update(interactive=False), gr.update(interactive=True)
            
            # ===== Plan Ready =====
            elif step.type == StepType.PLAN_READY:
                plan_content = step.content
                current_thinking = "계획 수립 완료, 실행 시작..."
                
                # 계획 표시
                response = build_streaming_response(
                    plan_content=plan_content,
                    step_outputs=step_outputs,
                    current_thinking=current_thinking
                )
                history[-1]["content"] = response
                yield history, gr.update(interactive=False), gr.update(interactive=True)
            
            # ===== Thinking =====
            elif step.type == StepType.THINKING:
                current_thinking = step.content
                
                response = build_streaming_response(
                    plan_content=plan_content,
                    step_outputs=step_outputs,
                    current_thinking=current_thinking
                )
                history[-1]["content"] = response
                yield history, gr.update(interactive=False), gr.update(interactive=True)
            
            # ===== Tool Call (시작) =====
            elif step.type == StepType.TOOL_CALL:
                tool_name = step.tool_name or "unknown"
                action = step.action or "unknown"
                
                # 새 step 시작
                current_step_info = {
                    "step": step.step,
                    "tool_name": tool_name,
                    "action": action,
                    "header": f"\n<details open>\n<summary>🔧 Step {step.step}: {tool_name}.{action}</summary>\n\n",
                    "result": "",
                    "status": "running"
                }
                
                # 진행 중 표시 (접이식 열린 상태)
                temp_outputs = step_outputs + [{
                    "header": current_step_info["header"],
                    "result": f"⏳ *실행 중...*\n</details>"
                }]
                
                response = build_streaming_response(
                    plan_content=plan_content,
                    step_outputs=temp_outputs,
                    current_thinking=""
                )
                history[-1]["content"] = response
                yield history, gr.update(interactive=False), gr.update(interactive=True)
            
            # ===== Tool Result (완료) =====
            elif step.type == StepType.TOOL_RESULT:
                tool_name = step.tool_name or current_step_info.get("tool_name", "unknown")
                action = step.action or current_step_info.get("action", "unknown")
                
                # 결과 포맷팅
                formatted_result = format_tool_result(step.content, tool_name, action)
                
                # Step 완료 정보 저장
                completed_step = {
                    "header": current_step_info.get("header", f"\n<details>\n<summary>🔧 Step {step.step}: {tool_name}.{action}</summary>\n\n"),
                    "result": f"✅ 완료\n\n{formatted_result}\n</details>"
                }
                step_outputs.append(completed_step)
                current_step_info = {}
                
                # 업데이트
                response = build_streaming_response(
                    plan_content=plan_content,
                    step_outputs=step_outputs,
                    current_thinking=""
                )
                history[-1]["content"] = response
                yield history, gr.update(interactive=False), gr.update(interactive=True)
            
            # ===== Final Answer =====
            elif step.type == StepType.FINAL_ANSWER:
                final_answer = step.content
                
                response = build_streaming_response(
                    plan_content=plan_content,
                    step_outputs=step_outputs,
                    current_thinking="",
                    final_answer=final_answer
                )
                history[-1]["content"] = response
                yield history, gr.update(interactive=False), gr.update(interactive=True)
            
            # ===== Error =====
            elif step.type == StepType.ERROR:
                error_msg = f"❌ **오류 발생**\n\n{step.content}"
                
                response = build_streaming_response(
                    plan_content=plan_content,
                    step_outputs=step_outputs,
                    current_thinking="",
                    final_answer=error_msg
                )
                history[-1]["content"] = response
                yield history, gr.update(interactive=True), gr.update(interactive=False)
        
        # ===== 중지된 경우 메시지 표시 =====
        if was_stopped:
            stop_msg = "⏹️ **사용자에 의해 중지됨**"
            response = build_streaming_response(
                plan_content=plan_content,
                step_outputs=step_outputs,
                current_thinking="",
                final_answer=stop_msg
            )
            history[-1]["content"] = response
            yield history, gr.update(interactive=True), gr.update(interactive=False)
    
    except GeneratorExit:
        # Gradio가 generator를 중단할 때
        state.orchestrator._stopped = True
        stop_msg = "⏹️ **중지됨**"
        response = build_streaming_response(
            plan_content=plan_content,
            step_outputs=step_outputs,
            current_thinking="",
            final_answer=stop_msg
        )
        history[-1]["content"] = response
        yield history, gr.update(interactive=True), gr.update(interactive=False)

    except Exception as e:
        error_msg = f"❌ **예외 발생**\n\n{str(e)}"
        history[-1]["content"] = error_msg
        yield history, gr.update(interactive=True), gr.update(interactive=False)
    
    finally:
        # 세션 정리
        state.orchestrator._stopped = False
    
    state.chat_history = history


def stop_generation():
    """생성 중지"""
    state = get_app_state()
    state.orchestrator.stop()


def clear_chat():
    """대화 초기화"""
    state = get_app_state()
    state.chat_history = []
    state.storage.reset()
    state.system_prompt_history = []
    return []


# =============================================================================
# LLM Settings Functions
# =============================================================================

def load_settings_for_modal():
    """설정 모달 열 때 현재 값 로드"""
    state = get_app_state()
    config = state.llm_config
    
    connected, models = state.fetch_ollama_models(config.base_url)
    
    if connected and models:
        model_choices = [m["display"] for m in models]
        current_display = config.model
        for m in models:
            if m["name"] == config.model:
                current_display = m["display"]
                break
    else:
        model_choices = [config.model]
        current_display = config.model
    
    url_status = "✅ 연결됨" if connected else "❌ 연결 실패"
    
    return (
        gr.update(visible=True),
        config.base_url,
        url_status,
        gr.update(choices=model_choices, value=current_display),
        config.temperature,
        config.max_tokens,
        config.top_p,
        config.repeat_penalty,
        config.num_ctx,
        config.max_steps,
        config.timeout
    )


def close_settings_modal():
    return gr.update(visible=False)


def on_url_change(url: str):
    state = get_app_state()
    connected, models = state.fetch_ollama_models(url)
    
    if connected and models:
        model_choices = [m["display"] for m in models]
        current_model = state.llm_config.model
        current_display = current_model
        for m in models:
            if m["name"] == current_model:
                current_display = m["display"]
                break
        
        return (
            "✅ 연결됨",
            gr.update(choices=model_choices, value=current_display if current_display in model_choices else model_choices[0])
        )
    else:
        return (
            "❌ 연결 실패",
            gr.update(choices=[state.llm_config.model], value=state.llm_config.model)
        )


def save_settings(url, model_display, temperature, max_tokens, top_p, repeat_penalty, num_ctx, max_steps, timeout):
    state = get_app_state()
    
    model_name = model_display.split(" (")[0] if " (" in model_display else model_display
    
    state.update_llm_config(
        base_url=url,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        max_steps=max_steps,
        timeout=timeout
    )
    
    return gr.update(visible=False)


# =============================================================================
# File Management Functions
# =============================================================================

def upload_files(files):
    """파일 업로드"""
    state = get_app_state()
    
    if not files:
        return get_file_list_html(), "파일을 선택해주세요."
    
    uploaded = []
    for file in files:
        file_info = state.workspace_manager.save_upload(file.name)
        if file_info:
            uploaded.append(file_info.name)
    
    if uploaded:
        return get_file_list_html(), f"✅ {len(uploaded)}개 파일 업로드됨"
    return get_file_list_html(), "❌ 업로드 실패"


def get_file_list_html() -> str:
    state = get_app_state()
    files = state.workspace_manager.list_files()

    if not files:
        return "<p style='color: gray;'>파일이 없습니다.</p>"
    
    html = "<table style='width:100%; border-collapse:collapse;'>"
    html += "<tr style='background:#f0f0f0;'><th>이름</th><th>크기</th><th>출처</th></tr>"
    
    for f in files:
        size = f"{f.size / 1024:.1f}KB" if f.size >= 1024 else f"{f.size}B"
        source = "📤 업로드" if f.source == "upload" else "🤖 생성"
        html += f"<tr><td>{f.name}</td><td>{size}</td><td>{source}</td></tr>"
    
    html += "</table>"
    return html


def get_file_choices() -> List[str]:
    state = get_app_state()
    files = state.workspace_manager.list_files()
    return [f.name for f in files]


def delete_selected_files(selected: List[str]):
    """선택된 파일 삭제"""
    state = get_app_state()
    
    if not selected:
        return get_file_list_html(), "삭제할 파일을 선택해주세요.", gr.update(choices=get_file_choices())
    
    deleted = 0
    for name in selected:
        if state.workspace_manager.delete_file(name):
            deleted += 1
    
    return get_file_list_html(), f"✅ {deleted}개 파일 삭제됨", gr.update(choices=get_file_choices(), value=[])


def delete_all_files():
    """전체 파일 삭제"""
    state = get_app_state()
    count = state.workspace_manager.delete_all_files()
    return get_file_list_html(), f"✅ {count}개 파일 삭제됨", gr.update(choices=[], value=[])


def refresh_file_list():
    """파일 목록 새로고침"""
    return get_file_list_html(), gr.update(choices=get_file_choices())


# =============================================================================
# SharedStorage Functions
# =============================================================================

def get_shared_storage_tree() -> str:
    state = get_app_state()
    storage = state.storage
    
    html = "<div style='font-family: monospace; font-size: 13px;'>"
    
    context = storage.get_context()
    html += "<details open><summary><b>📁 Context</b></summary>"
    html += "<div style='margin-left: 20px;'>"
    
    if context:
        user_query = context.get('user_query', 'N/A')
        if user_query and len(user_query) > 50:
            user_query = user_query[:50] + "..."
        html += f"<div>user_query: <code>{user_query}</code></div>"
        html += f"<div>session_id: <code>{context.get('session_id', 'N/A')}</code></div>"
    else:
        html += "<div style='color: gray;'>세션 없음</div>"
    
    html += "</div></details>"
    
    results = storage.get_results()
    html += f"<details open><summary><b>📁 Results ({len(results)})</b></summary>"
    html += "<div style='margin-left: 20px;'>"
    
    if not results:
        html += "<div style='color: gray;'>결과 없음</div>"
    else:
        for i, r in enumerate(results):
            status_icon = "✅" if r.get("success", False) else "❌"
            tool = r.get("executor", "unknown")
            action = r.get("action", "unknown")
            output_preview = str(r.get("output", ""))
            
            html += f"<details><summary>{status_icon} Step {i+1}: {tool}.{action}</summary>"
            html += f"<div style='margin-left: 20px; background: #f5f5f5; padding: 8px; border-radius: 4px;'>"
            html += f"<pre style='white-space: pre-wrap; margin: 0;'>{output_preview}</pre>"
            html += "</div></details>"
    
    html += "</div></details>"
    
    history = storage.get_history()
    html += f"<details><summary><b>📁 History ({len(history)})</b></summary>"
    html += "<div style='margin-left: 20px;'>"
    
    if not history:
        html += "<div style='color: gray;'>히스토리 없음</div>"
    else:
        for i, h in enumerate(history[-5:]):
            query = h.get("user_query", "")[:30]
            status = "✅" if h.get("status") == "completed" else "⚠️"
            html += f"<div>{status} {query}...</div>"
    
    html += "</div></details>"
    html += "</div>"
    
    return html


def refresh_shared_storage() -> str:
    return get_shared_storage_tree()


# =============================================================================
# History Functions
# =============================================================================

def get_history_html() -> str:
    state = get_app_state()
    history_data = state.storage.get_history()
    
    if not history_data:
        return "<p style='color: gray;'>히스토리가 없습니다.</p>"
    
    html = ""
    for i, session in enumerate(reversed(history_data[-10:])):
        query = session.get("user_query", "")[:50]
        if len(session.get("user_query", "")) > 50:
            query += "..."
        
        status_icon = "✅" if session.get("status") == "completed" else "⚠️"
        time_str = session.get("created_at", "")[:16]
        
        html += f"""
        <div style='padding:8px; margin:4px 0; background:#f5f5f5; border-radius:4px;'>
            <span>{status_icon}</span>
            <strong>{query}</strong>
            <span style='color:gray; font-size:0.8em;'>{time_str}</span>
        </div>
        """
    
    return html


# =============================================================================
# Build UI
# =============================================================================

def create_ui() -> gr.Blocks:
    """Gradio UI 생성"""

    def show_toast(message):
        """Gradio toast 메시지를 표시하는 JavaScript 함수."""
        js_code = f"""
        gradioApp().showToast("{message}");
        """
        return js_code

    state = get_app_state()
    
    with gr.Blocks(title="Multi-Agent Chatbot") as app:
        
        # Header
        with gr.Row():
            gr.Markdown("# 🤖 Multi-Agent Chatbot")
            settings_btn = gr.Button("⚙️ Settings", scale=1, variant="secondary")
        
        # Settings Modal
        with gr.Column(visible=False, elem_classes=["settings-modal"]) as settings_modal:
            gr.Markdown("## ⚙️ LLM Settings")
            
            with gr.Row():
                url_input = gr.Textbox(
                        label="Ollama URL",
                        value=state.llm_config.base_url,
                        placeholder="http://localhost:11434",
                        info="ollama 서버 주소를 입력하세요."
                )
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=[state.llm_config.model],
                    value=state.llm_config.model,
                    allow_custom_value=True,
                    info="사용할 LLM 모델을 선택하세요. (예: gpt-3.5-turbo, llama-2-7b)"
                )
                timeout_slider = gr.Slider(30.0, 1800.0, 300, value=state.llm_config.timeout, label="Timeout",
                                           info="LLM 응답 타임아웃 값을 지정합니다.")

            with gr.Row():
                url_status = gr.Markdown("⏳", elem_id="url-status")

            
            with gr.Row():
                temperature_slider = gr.Slider(0.0, 2.0, 0.1, value=state.llm_config.temperature, label="Temperature",
                                               info="LLM의 응답의 무작위성을 조절합니다. 높은 값일수록 답변의 창의적성이 증가 하고 일관성이 감소 합니다.")
                max_tokens_slider = gr.Slider(256, 4096, 256, value=state.llm_config.max_tokens, label="Max Tokens",
                                              info="LLM이 생성할 최대 토큰(단어 또는 단어 조각) 수를 지정합니다. 값이 너무 작으면 응답이 잘릴 수 있고, 값이 너무 크면 불필요하게 긴 응답이 생성될 수 있습니다.")
                num_ctx_slider = gr.Slider(2048, 32768, 1024, value=state.llm_config.num_ctx, label="Context Window",
                                           info="LLM이 한 번에 처리할 수 있는 최대 컨텍스트 길이(토큰 수)를 지정합니다. 값이 너무 작으면 LLM이 중요한 정보를 놓칠 수 있고, 값이 너무 크면 성능이 저하될 수 있습니다.")
            
            with gr.Row():
                top_p_slider = gr.Slider(0.0, 1.0, 0.05, value=state.llm_config.top_p, label="Top-p",
                                         info="LLM이 다음 단어를 선택할 때 고려하는 확률 분포의 누적 확률을 조절합니다. 높은 값일수록 다양한 단어를 선택하고 일관성이 감소 합니다.")
                repeat_penalty_slider = gr.Slider(1.0, 2.0, 0.1, value=state.llm_config.repeat_penalty, label="Repeat Penalty",
                                                  info="LLM이 이미 생성한 단어를 반복하는 것을 방지합니다. 값이 높을 수록 더 다양한 응답을 생성하고 값이 낮을 수록 더 자연스러운 응답을 생성합니다.")
                max_steps_slider = gr.Slider(10, 100, 50, value=state.llm_config.max_steps, label="Max Steps",
                                             info="하나의 요청에 대해서 최대 Step(도구 호출 횟수)을 제한 합니다. 값이 높을 수록 더 많은 도구들을 호출을 할 수 있지만, 그만큼 더 많은 시간이 소요 됩니다.")
            
            with gr.Row():
                save_btn = gr.Button("💾 Save", variant="primary")
                cancel_btn = gr.Button("Cancel")
        
        # Chat Area
        with gr.Column():
            chatbot = gr.Chatbot(
                label="대화",
                elem_classes=["chatbot"],
                height=500
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="메시지를 입력하세요... (예: sample.c 파일을 읽어서 정적 분석하고 out.md에 저장해줘)",
                    label="",
                    scale=10,
                    container=False
                )
                send_btn = gr.Button("▶️전송", variant="primary", scale=1, interactive=True)
                stop_btn = gr.Button("⏹️중지", variant="stop", scale=1, interactive=False)
                clear_btn = gr.Button("🗑️삭제", scale=1)
        
        gr.Markdown("---")
        
        # Tab Panels
        with gr.Tabs():
            
            # Workspace Files Tab
            with gr.TabItem("📁 Workspace Files"):
                with gr.Row():
                    file_upload = gr.File(label="파일 업로드", file_count="multiple", file_types=None)
                    upload_btn = gr.Button("📤 업로드", size="sm")
                
                file_list_html = gr.HTML(get_file_list_html())
                
                with gr.Row():
                    file_select = gr.CheckboxGroup(choices=get_file_choices(), label="삭제할 파일 선택")
                
                with gr.Row():
                    delete_selected_btn = gr.Button("🗑️ 선택 삭제", size="sm")
                    delete_all_btn = gr.Button("🗑️ 전체 삭제", size="sm", variant="stop")
                    refresh_files_btn = gr.Button("🔄 새로고침", size="sm")
                
                file_status = gr.Markdown("")
            
            # SharedStorage Tab
            with gr.TabItem("💾 SharedStorage"):
                storage_tree = gr.HTML(get_shared_storage_tree())
                refresh_storage_btn = gr.Button("🔄 새로고침", size="sm")
            
            # History Tab
            with gr.TabItem("📜 History"):
                history_html = gr.HTML(get_history_html())
                refresh_history_btn = gr.Button("🔄 새로고침", size="sm")
        
        # =================================================================
        # Event Handlers
        # =================================================================
        
        # Settings Modal
        settings_btn.click(
            fn=load_settings_for_modal,
            outputs=[settings_modal, url_input, url_status, model_dropdown,
                    temperature_slider, max_tokens_slider, top_p_slider,
                    repeat_penalty_slider, num_ctx_slider, max_steps_slider, timeout_slider]
        )

        cancel_btn.click(fn=close_settings_modal, outputs=[settings_modal])
        
        url_input.change(fn=on_url_change, inputs=[url_input], outputs=[url_status, model_dropdown])
        
        save_btn.click(
            fn=save_settings,
            inputs=[url_input, model_dropdown, temperature_slider, max_tokens_slider,
                   top_p_slider, repeat_penalty_slider, num_ctx_slider, max_steps_slider, timeout_slider],
            outputs=[settings_modal]
        )
        
        # Chat events
        def chat_stream_with_clear(message: str, history: List[Dict]):
            """
            Wrapper that clears input immediately on submit, then streams chat.

            First yield: clears input field immediately
            Subsequent yields: stream from chat_stream
            """
            # First yield to clear input immediately
            yield history, gr.update(interactive=False), gr.update(interactive=True), ""

            # Then stream the actual chat
            for chatbot_update, send_update, stop_update in chat_stream(message, history):
                yield chatbot_update, send_update, stop_update, ""

        def on_chat_complete():
            return get_history_html(), get_shared_storage_tree(), gr.update(interactive=True), gr.update(interactive=False)
        
        msg_input.submit(
            fn=chat_stream_with_clear,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, send_btn, stop_btn, msg_input]
        ).then(
            fn=on_chat_complete,
            outputs=[history_html, storage_tree, send_btn, stop_btn]
        )
        
        send_btn.click(
            fn=chat_stream_with_clear,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, send_btn, stop_btn, msg_input]
        ).then(
            fn=on_chat_complete,
            outputs=[history_html, storage_tree, send_btn, stop_btn]
        )
        
        stop_btn.click(fn=stop_generation)
        
        clear_btn.click(fn=clear_chat, outputs=[chatbot]).then(
            fn=on_chat_complete,
            outputs=[history_html, storage_tree, send_btn, stop_btn]
        )
        
        # File management
        upload_btn.click(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[file_list_html, file_status]
        ).then(
            fn=lambda: gr.update(choices=get_file_choices()),
            outputs=[file_select]
        )
        
        delete_selected_btn.click(
            fn=delete_selected_files,
            inputs=[file_select],
            outputs=[file_list_html, file_status, file_select]
        )
        
        delete_all_btn.click(
            fn=delete_all_files,
            outputs=[file_list_html, file_status, file_select]
        )
        
        refresh_files_btn.click(fn=refresh_file_list, outputs=[file_list_html, file_select])
        
        # SharedStorage & History
        refresh_storage_btn.click(fn=refresh_shared_storage, outputs=[storage_tree])
        refresh_history_btn.click(fn=get_history_html, outputs=[history_html])
        
        # Page Load
        def on_page_load():
            return (
                get_file_list_html(),
                gr.update(choices=get_file_choices()),
                get_shared_storage_tree(),
                get_history_html()
            )
        
        app.load(fn=on_page_load, outputs=[file_list_html, file_select, storage_tree, history_html])

    return app


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Multi-Agent Chatbot (with Real-time Streaming)")
    print("=" * 60)
    
    app = create_ui()
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        css ="""
            .chatbot .message {
                transition: all 0.2s ease;
            }
            """
    )


if __name__ == "__main__":
    main()
