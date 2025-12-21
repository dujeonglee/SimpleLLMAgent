"""
Multi-Agent Chatbot UI
======================
Gradio 기반 웹 인터페이스

Layout:
- Header: 타이틀 + Settings 버튼
- Row 1: 채팅 영역 (대화창, 입력창)
- Row 2: 탭 패널 (Workspace Files, SharedStorage, History)
- Modal: LLM Settings (팝업)
"""

import os
import sys
import warnings
from datetime import datetime
from typing import List, Tuple, Generator, Optional, Dict

# Gradio 내부 DeprecationWarning 숨기기 (starlette 호환성 문제)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")

import gradio as gr

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.shared_storage import SharedStorage
from core.base_tool import ToolRegistry
from core.orchestrator import Orchestrator, LLMConfig, StepInfo, StepType
from core.workspace_manager import WorkspaceManager, ConfigManager, FileInfo
from tools.file_tool import FileTool
from tools.web_tool import WebTool
from tools.llm_tool import LLMTool

# =============================================================================
# Global State
# =============================================================================

class AppState:
    """애플리케이션 상태 관리"""
    
    def __init__(self, workspace_path: str = os.path.join(".", "workspace")):
        self.workspace_path = workspace_path
        
        # 매니저 초기화
        self.workspace_manager = WorkspaceManager(workspace_path)
        self.config_manager = ConfigManager(os.path.join(workspace_path, "config"))
        
        # LLM 설정 로드
        self.llm_config = self.config_manager.load_llm_config()
        
        # Core 컴포넌트 초기화
        self.storage = SharedStorage(debug_enabled=True)
        self.registry = ToolRegistry(debug_enabled=True)
        
        # Tools 등록
        self._setup_tools()
        
        # Orchestrator 생성
        self.orchestrator = Orchestrator(
            tools=self.registry,
            storage=self.storage,
            llm_config=self.llm_config,
            max_steps=self.llm_config.max_steps,
            debug_enabled=True
        )
        
        # 대화 히스토리 (UI용)
        self.chat_history: List[Dict] = []
        
        # System Prompt 히스토리
        self.system_prompt_history: List[Dict] = []
        
        # Ollama 연결 상태
        self.ollama_connected: bool = False
        self.available_models: List[Dict] = []
    
    def _setup_tools(self):
        """Tools 설정"""
        files_dir = os.path.join(self.workspace_path, "files")
        os.makedirs(files_dir, exist_ok=True)
        
        self.registry.register(FileTool(base_path=files_dir, debug_enabled=True))
        self.registry.register(WebTool(debug_enabled=True, use_mock=False))
        self.registry.register(LLMTool(debug_enabled=True, use_mock=False))
    
    def update_llm_config(self, **kwargs):
        """LLM 설정 업데이트 및 저장"""
        self.llm_config.update(**kwargs)
        self.orchestrator.update_llm_config(**kwargs)
        
        # max_steps는 orchestrator에 직접 설정
        if 'max_steps' in kwargs:
            self.orchestrator.max_steps = kwargs['max_steps']
        
        self.config_manager.save_llm_config(self.llm_config)
    
    def fetch_ollama_models(self, base_url: str = None) -> Tuple[bool, List[Dict]]:
        """Ollama 서버에서 모델 목록 및 상세 정보 조회
        
        Returns:
            Tuple[bool, List[Dict]]: (연결 성공 여부, 모델 정보 리스트)
            모델 정보: {"name": str, "size": str, "family": str, "parameters": str}
        """
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
                
                # 크기 포맷팅
                if size_bytes >= 1024 * 1024 * 1024:
                    size_str = f"{size_bytes / (1024**3):.1f}GB"
                else:
                    size_str = f"{size_bytes / (1024**2):.0f}MB"
                
                # 모델 정보 추출
                family = details.get("family", "unknown")
                param_size = details.get("parameter_size", "")
                quantization = details.get("quantization_level", "")
                
                # 특성 문자열 생성
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
    
    def add_system_prompt_history(self, prompt: str, step: int):
        """System Prompt 히스토리 추가"""
        self.system_prompt_history.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt
        })


# 전역 상태
app_state: Optional[AppState] = None


def get_app_state() -> AppState:
    """앱 상태 가져오기 (lazy initialization)"""
    global app_state
    if app_state is None:
        app_state = AppState()
    return app_state


# =============================================================================
# Chat Functions
# =============================================================================

def format_tool_result(content: str, tool_name: str, action: str) -> str:
    """Tool 결과를 포맷팅"""
    content_str = str(content)
    
    if len(content_str) > 500:
        content_str = content_str[:500] + "\n... (truncated)"
    return f"```\n{content_str}\n```"


def chat_stream(message: str, history: List[Dict]) -> Generator[Tuple[List[Dict], str], None, None]:
    """채팅 메시지 처리 (Streaming)"""
    state = get_app_state()
    
    if not message.strip():
        yield history, "메시지를 입력해주세요."
        return
    
    # 파일 목록을 context에 추가
    files_context = state.workspace_manager.get_files_for_prompt()
    full_query = f"{message}\n\n{files_context}"
    
    # 새 대화 추가
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]
    current_response = ""
    step_outputs = []
    thoughts = []
    plan_content = ""
    
    yield history, "🔄 처리 중..."
    
    try:
        for step in state.orchestrator.run_stream(full_query):
            if step.type == StepType.PLANNING:
                yield history, f"📋 {step.content}"
            
            elif step.type == StepType.PLAN_READY:
                plan_content = step.content
                yield history, "📋 계획 수립 완료"
            
            elif step.type == StepType.THINKING:
                thoughts.append(step.content)
                yield history, f"💭 {step.content}"
            
            elif step.type == StepType.TOOL_CALL:
                step_output = f"\n\n<details>\n<summary>🔧 Step {step.step}: {step.tool_name}.{step.action}</summary>\n\n"
                if thoughts:
                    step_output += f"💭 **Thought:** {thoughts[-1]}\n\n"
                step_outputs.append({
                    "header": step_output, 
                    "result": "",
                    "tool_name": step.tool_name,
                    "action": step.action
                })
                yield history, f"🔧 {step.tool_name}.{step.action} 실행 중..."
            
            elif step.type == StepType.TOOL_RESULT:
                if step_outputs:
                    last_step = step_outputs[-1]
                    formatted_result = format_tool_result(
                        step.content, 
                        last_step.get("tool_name", ""),
                        last_step.get("action", "")
                    )
                    step_outputs[-1]["result"] = f"{formatted_result}\n</details>"
                yield history, f"✅ {step.tool_name}.{step.action} 완료"
            
            elif step.type == StepType.FINAL_ANSWER:
                response_parts = []
                
                if plan_content:
                    response_parts.append(f"<details>\n<summary>📋 실행 계획</summary>\n\n{plan_content}\n</details>")
                
                if step_outputs:
                    steps_md = ""
                    for s in step_outputs:
                        steps_md += s["header"] + s["result"]
                    response_parts.append(steps_md)
                
                response_parts.append(step.content)
                
                current_response = "\n\n---\n\n".join(filter(None, response_parts))
                
                history[-1] = {"role": "assistant", "content": current_response}
                yield history, "✅ 완료"
            
            elif step.type == StepType.ERROR:
                current_response = f"❌ **오류 발생**\n\n{step.content}"
                history[-1] = {"role": "assistant", "content": current_response}
                yield history, "❌ 오류 발생"
    
    except Exception as e:
        error_msg = f"❌ **예외 발생**\n\n{str(e)}"
        history[-1] = {"role": "assistant", "content": error_msg}
        yield history, f"❌ {str(e)}"
    
    state.chat_history = history


def stop_generation():
    """생성 중지"""
    state = get_app_state()
    state.orchestrator.stop()
    return "⏹️ 중지됨"


def clear_chat():
    """대화 초기화"""
    state = get_app_state()
    state.chat_history = []
    state.storage.reset()
    state.system_prompt_history = []
    return [], "대화가 초기화되었습니다."


# =============================================================================
# LLM Settings Functions (Modal)
# =============================================================================

def load_settings_for_modal():
    """설정 모달 열 때 현재 값 로드"""
    state = get_app_state()
    config = state.llm_config
    
    # Ollama 모델 목록 가져오기
    connected, models = state.fetch_ollama_models(config.base_url)
    
    # 모델 선택 목록 생성
    if connected and models:
        model_choices = [m["display"] for m in models]
        # 현재 모델 찾기
        current_display = config.model
        for m in models:
            if m["name"] == config.model:
                current_display = m["display"]
                break
    else:
        model_choices = [config.model]
        current_display = config.model
    
    # URL 상태 표시
    url_status = "✅" if connected else "❌"
    
    return (
        gr.update(visible=True),  # modal visible
        config.base_url,
        url_status,
        gr.update(choices=model_choices, value=current_display),
        config.temperature,
        config.max_tokens,
        config.top_p,
        config.repeat_penalty,
        config.num_ctx,
        config.max_steps
    )


def close_settings_modal():
    """설정 모달 닫기"""
    return gr.update(visible=False)


def on_url_change(url: str):
    """Ollama URL 변경 시 모델 목록 자동 갱신"""
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
            "✅",
            gr.update(choices=model_choices, value=current_display if current_display in model_choices else model_choices[0])
        )
    else:
        return (
            "❌",
            gr.update(choices=[state.llm_config.model], value=state.llm_config.model)
        )


def save_settings(url, model_display, temperature, max_tokens, top_p, repeat_penalty, num_ctx, max_steps):
    """설정 저장"""
    state = get_app_state()
    
    # 모델 이름 추출 (display에서 실제 이름)
    model_name = model_display.split(" (")[0] if " (" in model_display else model_display
    
    # 설정 업데이트
    state.update_llm_config(
        base_url=url,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        max_steps=max_steps
    )
    
    return gr.update(visible=False)


# =============================================================================
# File Management Functions
# =============================================================================

def upload_files(files) -> Tuple[str, str]:
    """파일 업로드 처리"""
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
    """파일 목록 HTML 생성"""
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
    """파일 선택 목록"""
    state = get_app_state()
    files = state.workspace_manager.list_files()
    return [f.name for f in files]


def delete_selected_files(selected: List[str]) -> Tuple[str, str, gr.update]:
    """선택된 파일 삭제"""
    state = get_app_state()
    
    if not selected:
        return get_file_list_html(), "삭제할 파일을 선택해주세요.", gr.update(choices=get_file_choices())
    
    deleted = 0
    for name in selected:
        if state.workspace_manager.delete_file(name):
            deleted += 1
    
    return get_file_list_html(), f"✅ {deleted}개 파일 삭제됨", gr.update(choices=get_file_choices(), value=[])


def delete_all_files() -> Tuple[str, str, gr.update]:
    """전체 파일 삭제"""
    state = get_app_state()
    count = state.workspace_manager.delete_all_files()
    return get_file_list_html(), f"✅ {count}개 파일 삭제됨", gr.update(choices=[], value=[])


def refresh_file_list() -> Tuple[str, gr.update]:
    """파일 목록 새로고침"""
    return get_file_list_html(), gr.update(choices=get_file_choices())


# =============================================================================
# SharedStorage Functions
# =============================================================================

def get_shared_storage_tree() -> str:
    """SharedStorage 내용을 트리뷰 형태로 HTML 생성"""
    state = get_app_state()
    storage = state.storage
    
    html = "<div style='font-family: monospace; font-size: 13px;'>"
    
    # Context
    context = storage.get_context()
    html += "<details open><summary><b>📁 Context</b></summary>"
    html += "<div style='margin-left: 20px;'>"
    
    if context:
        user_query = context.get('user_query', 'N/A')
        if user_query and len(user_query) > 50:
            user_query = user_query[:50] + "..."
        html += f"<div>user_query: <code>{user_query}</code></div>"
        html += f"<div>current_step: <code>{context.get('current_step', 0)}</code></div>"
        html += f"<div>session_id: <code>{context.get('session_id', 'N/A')}</code></div>"
    else:
        html += "<div style='color: gray;'>세션 없음</div>"
    
    html += "</div></details>"
    
    # Results
    results = storage.get_results()
    html += f"<details open><summary><b>📁 Results ({len(results)})</b></summary>"
    html += "<div style='margin-left: 20px;'>"
    
    if not results:
        html += "<div style='color: gray;'>결과 없음</div>"
    else:
        for i, r in enumerate(results):
            status_icon = "✅" if r.get("status") == "success" else "❌"
            tool = r.get("tool", "unknown")
            action = r.get("action", "unknown")
            output_preview = str(r.get("output", ""))[:100]
            
            html += f"<details><summary>{status_icon} Step {i+1}: {tool}.{action}</summary>"
            html += f"<div style='margin-left: 20px; background: #f5f5f5; padding: 8px; border-radius: 4px;'>"
            html += f"<pre style='white-space: pre-wrap; margin: 0;'>{output_preview}...</pre>"
            html += "</div></details>"
    
    html += "</div></details>"
    
    # History
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
    """SharedStorage 새로고침"""
    return get_shared_storage_tree()


# =============================================================================
# History Functions
# =============================================================================

def get_history_html() -> str:
    """대화 히스토리 HTML"""
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
    """Gradio UI 생성 - Settings Modal 포함"""
    
    state = get_app_state()
    
    with gr.Blocks(title="Multi-Agent Chatbot") as app:
        
        # =================================================================
        # Header: 타이틀 + Settings 버튼
        # =================================================================
        with gr.Row():
            gr.Markdown("# 🤖 Multi-Agent Chatbot")
            settings_btn = gr.Button("⚙️ Settings", scale=1, variant="secondary")
        
        # =================================================================
        # Settings Modal (숨김 상태로 시작)
        # =================================================================
        with gr.Column(visible=False, elem_classes=["settings-modal"]) as settings_modal:
            gr.Markdown("## ⚙️ LLM Settings")
            
            with gr.Row():
                with gr.Column(scale=4):
                    url_input = gr.Textbox(
                        label="Ollama URL",
                        value=state.llm_config.base_url,
                        placeholder="http://localhost:11434"
                    )
                with gr.Column(scale=1):
                    url_status = gr.Markdown("⏳", elem_id="url-status")
            
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=[state.llm_config.model],
                value=state.llm_config.model,
                allow_custom_value=True,
                info="모델 선택 (특성 정보 포함)"
            )
            
            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.1,
                    value=state.llm_config.temperature,
                    label="Temperature"
                )
                max_tokens_slider = gr.Slider(
                    minimum=256, maximum=4096, step=256,
                    value=state.llm_config.max_tokens,
                    label="Max Tokens"
                )
            
            with gr.Row():
                top_p_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05,
                    value=state.llm_config.top_p,
                    label="Top-p"
                )
                repeat_penalty_slider = gr.Slider(
                    minimum=1.0, maximum=2.0, step=0.1,
                    value=state.llm_config.repeat_penalty,
                    label="Repeat Penalty"
                )
            
            with gr.Row():
                num_ctx_slider = gr.Slider(
                    minimum=2048, maximum=32768, step=1024,
                    value=state.llm_config.num_ctx,
                    label="Context Window"
                )
                max_steps_slider = gr.Slider(
                    minimum=1, maximum=20, step=1,
                    value=state.llm_config.max_steps,
                    label="Max Steps"
                )
            
            with gr.Row():
                save_btn = gr.Button("💾 Save", variant="primary")
                cancel_btn = gr.Button("Cancel")
        
        # =================================================================
        # Row 1: 채팅 영역
        # =================================================================
        with gr.Column():
            chatbot = gr.Chatbot(
                label="대화",
                elem_classes=["chatbot"],
                height=400
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="메시지를 입력하세요...",
                    label="",
                    scale=8,
                    container=False
                )
                send_btn = gr.Button("전송", variant="primary", scale=1)
                stop_btn = gr.Button("중지", variant="stop", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ 대화 초기화", size="sm")
                status_text = gr.Markdown("Ready")
        
        gr.Markdown("---")
        
        # =================================================================
        # Row 2: 탭 패널 (LLM Settings 제외)
        # =================================================================
        with gr.Tabs():
            
            # Tab 1: Workspace Files
            with gr.TabItem("📁 Workspace Files"):
                with gr.Row():
                    file_upload = gr.File(
                        label="파일 업로드",
                        file_count="multiple",
                        file_types=None
                    )
                    upload_btn = gr.Button("📤 업로드", size="sm")
                
                file_list_html = gr.HTML(get_file_list_html())
                
                with gr.Row():
                    file_select = gr.CheckboxGroup(
                        choices=get_file_choices(),
                        label="삭제할 파일 선택"
                    )
                
                with gr.Row():
                    delete_selected_btn = gr.Button("🗑️ 선택 삭제", size="sm")
                    delete_all_btn = gr.Button("🗑️ 전체 삭제", size="sm", variant="stop")
                    refresh_files_btn = gr.Button("🔄 새로고침", size="sm")
                
                file_status = gr.Markdown("")
            
            # Tab 2: SharedStorage
            with gr.TabItem("💾 SharedStorage"):
                storage_tree = gr.HTML(get_shared_storage_tree())
                refresh_storage_btn = gr.Button("🔄 새로고침", size="sm")
            
            # Tab 3: History
            with gr.TabItem("📜 History"):
                history_html = gr.HTML(get_history_html())
                refresh_history_btn = gr.Button("🔄 새로고침", size="sm")
            
        # =================================================================
        # Event Handlers
        # =================================================================
        
        # Settings Modal events
        settings_btn.click(
            fn=load_settings_for_modal,
            outputs=[
                settings_modal,
                url_input,
                url_status,
                model_dropdown,
                temperature_slider,
                max_tokens_slider,
                top_p_slider,
                repeat_penalty_slider,
                num_ctx_slider,
                max_steps_slider
            ]
        )
        
        cancel_btn.click(
            fn=close_settings_modal,
            outputs=[settings_modal]
        )
        
        url_input.change(
            fn=on_url_change,
            inputs=[url_input],
            outputs=[url_status, model_dropdown]
        )
        
        save_btn.click(
            fn=save_settings,
            inputs=[
                url_input,
                model_dropdown,
                temperature_slider,
                max_tokens_slider,
                top_p_slider,
                repeat_penalty_slider,
                num_ctx_slider,
                max_steps_slider
            ],
            outputs=[settings_modal]
        )
        
        # Chat events
        def on_chat_complete():
            return (
                get_history_html(),
                get_shared_storage_tree()
            )
        
        msg_input.submit(
            fn=chat_stream,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, status_text]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        ).then(
            fn=on_chat_complete,
            outputs=[history_html, storage_tree]
        )
        
        send_btn.click(
            fn=chat_stream,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, status_text]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        ).then(
            fn=on_chat_complete,
            outputs=[history_html, storage_tree]
        )
        
        stop_btn.click(fn=stop_generation, outputs=[status_text])
        clear_btn.click(
            fn=clear_chat, 
            outputs=[chatbot, status_text]
        ).then(
            fn=on_chat_complete,
            outputs=[history_html, storage_tree]
        )
        
        # File management events
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
        
        refresh_files_btn.click(
            fn=refresh_file_list,
            outputs=[file_list_html, file_select]
        )
        
        # SharedStorage events
        refresh_storage_btn.click(fn=refresh_shared_storage, outputs=[storage_tree])
        
        # History events
        refresh_history_btn.click(fn=get_history_html, outputs=[history_html])
        
        # =================================================================
        # Page Load Event - 브라우저 새로고침 시 최신 데이터 로드
        # =================================================================
        def on_page_load():
            return (
                get_file_list_html(),
                gr.update(choices=get_file_choices()),
                get_shared_storage_tree(),
                get_history_html()
            )
        
        app.load(
            fn=on_page_load,
            outputs=[file_list_html, file_select, storage_tree, history_html]
        )

    return app


# =============================================================================
# Main
# =============================================================================

def main():
    """메인 함수"""
    print("=" * 60)
    print("Multi-Agent Chatbot")
    print("=" * 60)
    
    app = create_ui()
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
