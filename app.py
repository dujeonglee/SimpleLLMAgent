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
from core.orchestrator import Orchestrator
from core.workspace_manager import WorkspaceManager, ConfigManager
from core.execution_events import (
    ExecutionEvent,
    PlanPromptEvent,
    PlanReadyEvent,
    ThinkingEvent,
    StepPromptEvent,
    ToolCallEvent,
    ToolResultEvent,
    FinalAnswerEvent,
    ErrorEvent
)
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
# Chat Functions (Updated - Real-time Streaming with ExecutionEvents)
# =============================================================================

def chat_stream(message: str, history: List[Dict]) -> Generator[tuple, None, None]:
    """
    채팅 메시지 처리 (Real-time Streaming)

    각 ExecutionEvent의 to_display() 메서드를 사용하여 실시간으로 표시합니다.
    Returns: (history, send_btn_update, stop_btn_update, file_list_html_update)
    """
    state = get_app_state()

    state.orchestrator._stopped = False

    if not message.strip():
        yield history, gr.update(interactive=False), gr.update(interactive=True), gr.update()
        return

    # 파일 목록을 context로 준비 (orchestrator에 전달)
    files_context = state.workspace_manager.get_files_for_prompt()

    # 새 대화 추가 (user 메시지)
    history = history + [
        {"role": "user", "content": message}
    ]

    # 누적 응답 구성
    accumulated_output = []
    was_stopped = False

    # 초기 상태 표시
    yield history, gr.update(interactive=False), gr.update(interactive=True), gr.update()

    try:
        for event in state.orchestrator.run_stream(message, files_context):
            # ===== 중지 체크 =====
            if state.orchestrator._stopped:
                was_stopped = True
                break

            # 각 이벤트의 to_display() 메서드로 출력 생성
            display_text = event.to_display()

            # StepPromptEvent는 빈 문자열을 반환 (ToolResultEvent에 포함되므로)
            # 따라서 빈 문자열이 아닐 때만 추가
            if display_text:
                temporal_event_str = '''<div style="display: flex; align-items: center; gap: 8px;"><div class="spinner"></div>'''
                if accumulated_output and temporal_event_str in accumulated_output[-1]:
                    accumulated_output.pop()

                accumulated_output.append(display_text)

            # 최종 응답 업데이트
            response = "\n".join(accumulated_output)

            # file_tool의 write/delete 완료 시 workspace UI 업데이트
            file_list_update = gr.update()
            if isinstance(event, ToolResultEvent):
                if event.tool_name == "file_tool" and event.action in ["write", "delete"] and event.success:
                    file_list_update = gr.update(value=get_files_data())

            # FinalAnswerEvent나 ErrorEvent가 오면 종료 처리
            if isinstance(event, (FinalAnswerEvent, ErrorEvent)):
                history[-1] = {"role": "assistant", "content": response}
                yield history, gr.update(interactive=True), gr.update(interactive=False), file_list_update
                break
            else:
                # 중간 진행 상황 업데이트
                if len(history) > 0 and history[-1]["role"] == "assistant":
                    history[-1] = {"role": "assistant", "content": response}
                else:
                    history = history + [{"role": "assistant", "content": response}]
                yield history, gr.update(interactive=False), gr.update(interactive=True), file_list_update

        # ===== 중지된 경우 메시지 표시 =====
        if was_stopped:
            accumulated_output.append("\n⏹️ **중지됨**")
            response = "\n".join(accumulated_output)
            history[-1] = {"role": "assistant", "content": response}
            yield history, gr.update(interactive=True), gr.update(interactive=False), gr.update()

    except GeneratorExit:
        # Gradio가 generator를 중단할 때
        state.orchestrator._stopped = True
        accumulated_output.append("\n⏹️ **중지됨**")
        response = "\n".join(accumulated_output)
        if len(history) > 0 and history[-1]["role"] == "assistant":
            history[-1]["content"] = response
        yield history, gr.update(interactive=True), gr.update(interactive=False), gr.update()

    except Exception as e:
        accumulated_output.append(f"\n❌ **예외 발생**\n\n{str(e)}")
        response = "\n".join(accumulated_output)
        if len(history) > 0 and history[-1]["role"] == "assistant":
            history[-1]["content"] = response
        yield history, gr.update(interactive=True), gr.update(interactive=False), gr.update()

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
        config.top_k,
        config.repeat_penalty,
        config.frequency_penalty,
        config.presence_penalty,
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


def save_settings(url, model_display, temperature, max_tokens, top_p, top_k, repeat_penalty,
                  frequency_penalty, presence_penalty, num_ctx, max_steps, timeout):
    state = get_app_state()

    model_name = model_display.split(" (")[0] if " (" in model_display else model_display

    state.update_llm_config(
        base_url=url,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        num_ctx=num_ctx,
        max_steps=max_steps,
        timeout=timeout
    )

    return gr.update(visible=False)


# =============================================================================
# File Management Functions
# =============================================================================

def get_files_data() -> List[Dict]:
    """파일 정보를 딕셔너리 리스트로 반환"""
    state = get_app_state()
    files = state.workspace_manager.list_files()
    files_dir = os.path.join(state.workspace_path, "files")

    result = []
    for f in files:
        size = f"{f.size / 1024:.1f}KB" if f.size >= 1024 else f"{f.size}B"
        source = "📤 업로드" if f.source == "upload" else "🤖 생성"

        # 파일 내용 읽기
        file_path = os.path.join(files_dir, f.name)
        content = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp949') as file:
                    content = file.read()
            except:
                content = "[파일 내용을 읽을 수 없습니다 - 바이너리 파일일 수 있습니다]"
        except Exception as e:
            content = f"[오류 발생: {str(e)}]"

        result.append({
            'name': f.name,
            'size': size,
            'source': source,
            'content': content
        })

    return result


def upload_files(files):
    """파일 업로드"""
    state = get_app_state()

    if not files:
        return get_files_data(), "파일을 선택해주세요."

    uploaded = []
    for file in files:
        file_info = state.workspace_manager.save_upload(file.name)
        if file_info:
            uploaded.append(file_info.name)

    if uploaded:
        return get_files_data(), f"✅ {len(uploaded)}개 파일 업로드됨"
    return get_files_data(), "❌ 업로드 실패"


def delete_single_file(filename: str):
    """단일 파일 삭제"""
    state = get_app_state()
    success = state.workspace_manager.delete_file(filename)
    if success:
        return get_files_data(), f"✅ '{filename}' 파일 삭제됨"
    return get_files_data(), f"❌ '{filename}' 파일 삭제 실패"


def delete_all_files():
    """전체 파일 삭제"""
    state = get_app_state()
    count = state.workspace_manager.delete_all_files()
    return get_files_data(), f"✅ {count}개 파일 삭제됨"


def refresh_file_list():
    """파일 목록 새로고침"""
    return get_files_data(), ""


# =============================================================================
# Build UI
# =============================================================================

def create_ui() -> gr.Blocks:
    """Gradio UI 생성"""

    state = get_app_state()
    
    with gr.Blocks(title="Multi-Agent Chatbot") as app:

        # Header
        gr.Markdown("# 🤖 Multi-Agent Chatbot")

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
                timeout_slider = gr.Slider(10, 3600, 10, value=state.llm_config.timeout, label="Timeout",
                                           info="LLM 응답 타임아웃 값을 지정합니다.")

            with gr.Row():
                url_status = gr.Markdown("⏳", elem_id="url-status")

            
            with gr.Row():
                temperature_slider = gr.Slider(0.0, 2.0, 0.1, value=state.llm_config.temperature, label="Temperature",
                                               info="LLM의 응답의 무작위성을 조절합니다. 높은 값일수록 답변의 창의적성이 증가 하고 일관성이 감소 합니다.")
                max_tokens_slider = gr.Slider(256, 24576, 256, value=state.llm_config.max_tokens, label="Max Tokens",
                                              info="LLM이 생성할 최대 토큰(단어 또는 단어 조각) 수를 지정합니다. 값이 너무 작으면 응답이 잘릴 수 있고, 값이 너무 크면 불필요하게 긴 응답이 생성될 수 있습니다.")
                num_ctx_slider = gr.Slider(2048, 32768, 1024, value=state.llm_config.num_ctx, label="Context Window",
                                           info="LLM이 한 번에 처리할 수 있는 최대 컨텍스트 길이(토큰 수)를 지정합니다. 값이 너무 작으면 LLM이 중요한 정보를 놓칠 수 있고, 값이 너무 크면 성능이 저하될 수 있습니다.")
            
            with gr.Row():
                top_p_slider = gr.Slider(0.0, 1.0, 0.05, value=state.llm_config.top_p, label="Top-p",
                                         info="LLM이 다음 단어를 선택할 때 고려하는 확률 분포의 누적 확률을 조절합니다. 높은 값일수록 다양한 단어를 선택하고 일관성이 감소 합니다.")
                top_k_slider = gr.Slider(1, 100, 1, value=state.llm_config.top_k, label="Top-k",
                                         info="다음 토큰 선택 시 고려할 상위 후보 개수입니다. 낮은 값은 더 일관성 있지만 덜 다양한 응답을 생성합니다.")
                repeat_penalty_slider = gr.Slider(1.0, 2.0, 0.1, value=state.llm_config.repeat_penalty, label="Repeat Penalty",
                                                  info="LLM이 이미 생성한 단어를 반복하는 것을 방지합니다. 값이 높을 수록 더 다양한 응답을 생성하고 값이 낮을 수록 더 자연스러운 응답을 생성합니다.")

            with gr.Row():
                frequency_penalty_slider = gr.Slider(0.0, 2.0, 0.1, value=state.llm_config.frequency_penalty, label="Frequency Penalty",
                                                     info="토큰의 빈도에 따라 페널티를 적용합니다. 높은 값은 자주 사용된 토큰의 재사용을 줄입니다.")
                presence_penalty_slider = gr.Slider(0.0, 2.0, 0.1, value=state.llm_config.presence_penalty, label="Presence Penalty",
                                                    info="이미 등장한 토큰에 페널티를 적용합니다. 높은 값은 새로운 주제로의 전환을 촉진합니다.")
                max_steps_slider = gr.Slider(10, 100, 1, value=state.llm_config.max_steps, label="Max Steps",
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
                settings_btn = gr.Button("⚙️ Settings", scale=1, variant="secondary")
        
        gr.Markdown("---")
        
        # Tab Panels
        with gr.Tabs():
            
            # Workspace Files Tab
            with gr.TabItem("📁 Workspace Files"):
                file_upload = gr.File(
                    label="파일을 드래그하거나 클릭하여 업로드",
                    file_count="multiple",
                    file_types=None
                )

                with gr.Row():
                    delete_all_btn = gr.Button("🗑️ 전체 삭제", size="sm", variant="stop")
                    refresh_files_btn = gr.Button("🔄 새로고침", size="sm")

                file_status = gr.Markdown("")

                # 파일 목록을 위한 State
                files_state = gr.State([])

                # 동적 파일 리스트를 렌더링하는 함수
                @gr.render(inputs=[files_state])
                def render_file_list(files):
                    if not files:
                        gr.Markdown("_파일이 없습니다._")
                        return

                    for file_info in files:
                        with gr.Row():
                            with gr.Column(scale=8):
                                with gr.Accordion(file_info['name'], open=False):
                                    gr.Code(
                                        value=file_info['content'],
                                        language=None,
                                        interactive=False,
                                        max_lines=20
                                    )
                            with gr.Column(scale=2):
                                gr.Markdown(f"**크기:** {file_info['size']}")
                                gr.Markdown(f"**출처:** {file_info['source']}")
                            with gr.Column(scale=1):
                                delete_btn = gr.Button("🗑️", size="sm", variant="stop")
                                delete_btn.click(
                                    fn=lambda fname=file_info['name']: delete_single_file(fname),
                                    outputs=[files_state, file_status]
                                )
            
        
        # =================================================================
        # Event Handlers
        # =================================================================
        
        # Settings Modal
        settings_btn.click(
            fn=load_settings_for_modal,
            outputs=[settings_modal, url_input, url_status, model_dropdown,
                    temperature_slider, max_tokens_slider, top_p_slider, top_k_slider,
                    repeat_penalty_slider, frequency_penalty_slider, presence_penalty_slider,
                    num_ctx_slider, max_steps_slider, timeout_slider]
        )

        cancel_btn.click(fn=close_settings_modal, outputs=[settings_modal])

        url_input.change(fn=on_url_change, inputs=[url_input], outputs=[url_status, model_dropdown])

        save_btn.click(
            fn=save_settings,
            inputs=[url_input, model_dropdown, temperature_slider, max_tokens_slider,
                   top_p_slider, top_k_slider, repeat_penalty_slider,
                   frequency_penalty_slider, presence_penalty_slider,
                   num_ctx_slider, max_steps_slider, timeout_slider],
            outputs=[settings_modal]
        )
        
        # Chat events
        def chat_stream_with_clear(message: str, history: List[Dict]):
            """
            Wrapper that clears input immediately on submit, then streams chat.

            First yield: clears input field immediately
            Subsequent yields: stream from chat_stream
            Returns: (chatbot, send_btn, stop_btn, msg_input, files_state)
            """
            # First yield to clear input immediately
            yield history, gr.update(interactive=False), gr.update(interactive=True), "", gr.update()

            # Then stream the actual chat
            for chatbot_update, send_update, stop_update, file_list_update in chat_stream(message, history):
                yield chatbot_update, send_update, stop_update, "", file_list_update

        def on_chat_complete():
            return gr.update(interactive=True), gr.update(interactive=False)

        msg_input.submit(
            fn=chat_stream_with_clear,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, send_btn, stop_btn, msg_input, files_state],
            concurrency_limit=1
        ).then(
            fn=on_chat_complete,
            outputs=[send_btn, stop_btn]
        )

        send_btn.click(
            fn=chat_stream_with_clear,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, send_btn, stop_btn, msg_input, files_state],
            concurrency_limit=1
        ).then(
            fn=on_chat_complete,
            outputs=[send_btn, stop_btn]
        )

        stop_btn.click(fn=stop_generation)

        clear_btn.click(fn=clear_chat, outputs=[chatbot]).then(
            fn=on_chat_complete,
            outputs=[send_btn, stop_btn]
        )
        
        # File management
        # 파일이 업로드되면 자동으로 처리하고 파일 선택 창 초기화
        file_upload.upload(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[files_state, file_status]
        ).then(
            fn=lambda: None,
            outputs=[file_upload]
        )

        delete_all_btn.click(
            fn=delete_all_files,
            outputs=[files_state, file_status]
        )

        refresh_files_btn.click(
            fn=refresh_file_list,
            outputs=[files_state, file_status]
        )

        # Page Load
        def on_page_load():
            return get_files_data()

        app.load(fn=on_page_load, outputs=[files_state])

    return app


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Multi-Agent Chatbot (with Real-time Streaming)")
    print("=" * 60)
    
    app = create_ui()
    app.queue()
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        css ="""
            .chatbot .message {
                transition: all 0.2s ease;
            }
            .spinner {
                width: 14px;
                height: 14px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            """
            )


if __name__ == "__main__":
    main()
