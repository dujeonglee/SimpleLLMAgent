"""
Execution Events Module
=======================
Orchestrator에서 발생하는 다양한 실행 이벤트를 정의합니다.
각 이벤트는 자신만의 출력 포맷을 정의할 수 있습니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


# =============================================================================
# Base Class
# =============================================================================

@dataclass
class ExecutionEvent(ABC):
    """실행 이벤트 Base Class"""
    timestamp: str = field(init=False, default_factory=lambda: datetime.now().isoformat())

    @abstractmethod
    def to_display(self) -> str:
        """UI 표시용 문자열 생성 (각 자식 클래스에서 구현)"""
        pass

    def to_dict(self) -> Dict:
        """디버깅/로깅용 딕셔너리 변환"""
        return {
            "event_type": self.__class__.__name__,
            "timestamp": self.timestamp
        }


# =============================================================================
# Planning Events
# =============================================================================

@dataclass
class PlanningEvent(ExecutionEvent):
    """계획 수립 중 이벤트"""
    message: str = "작업 계획 수립 중..."

    def to_display(self) -> str:
        return f"📋 *{self.message}*"

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d["message"] = self.message
        return d


@dataclass
class PlanPromptEvent(ExecutionEvent):
    """계획 생성 프롬프트 정보 이벤트"""
    system_prompt: str
    user_prompt: str
    raw_response: str

    def to_display(self) -> str:
        """구조화된 텍스트 포맷으로 출력"""
        output = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += "🔍 **Plan Generation Details**\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        output += "📋 **System Prompt**\n"
        output += "────────────────────────────────────────────────\n"
        output += f"```\n{self.system_prompt}\n```\n\n"

        output += "💬 **User Prompt**\n"
        output += "────────────────────────────────────────────────\n"
        output += f"```\n{self.user_prompt}\n```\n\n"

        output += "🤖 **LLM Response**\n"
        output += "────────────────────────────────────────────────\n"
        output += f"```json\n{self.raw_response}\n```\n"

        return output

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_response": self.raw_response
        })
        return d


@dataclass
class PlanReadyEvent(ExecutionEvent):
    """실행 계획 준비 완료 이벤트"""
    plan_content: str

    def to_display(self) -> str:
        output = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += "📋 **실행 계획**\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"{self.plan_content}\n"
        return output

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d["plan_content"] = self.plan_content
        return d


# =============================================================================
# Execution Events
# =============================================================================

@dataclass
class ThinkingEvent(ExecutionEvent):
    """진행 상황 표시 이벤트"""
    message: str

    def to_display(self) -> str:
        return f"\n💭 *{self.message}*"

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d["message"] = self.message
        return d


@dataclass
class StepPromptEvent(ExecutionEvent):
    """Step 실행 프롬프트 정보 이벤트"""
    step: int
    tool_name: str
    action: str
    system_prompt: str
    user_prompt: str
    raw_response: str

    def to_display(self) -> str:
        """이 이벤트는 별도로 표시하지 않고 ToolResultEvent에서 함께 표시"""
        return ""

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({
            "step": self.step,
            "tool_name": self.tool_name,
            "action": self.action,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_response": self.raw_response
        })
        return d


@dataclass
class ToolCallEvent(ExecutionEvent):
    """Tool 호출 시작 이벤트"""
    step: int
    tool_name: str
    action: str
    arguments: Dict

    def to_display(self) -> str:
        output = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"🔧 **Step {self.step}: {self.tool_name}.{self.action}**\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        output += "⏳ *실행 중...*\n"
        return output

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({
            "step": self.step,
            "tool_name": self.tool_name,
            "action": self.action,
            "arguments": self.arguments
        })
        return d


@dataclass
class ToolResultEvent(ExecutionEvent):
    """Tool 실행 결과 이벤트"""
    step: int
    tool_name: str
    action: str
    result: str
    success: bool = True
    prompt_info: Optional[StepPromptEvent] = None

    def to_display(self) -> str:
        output = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += f"🔧 **Step {self.step}: {self.tool_name}.{self.action}**\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        # Execution Details (프롬프트 정보가 있으면 표시)
        if self.prompt_info:
            output += "🔍 **Execution Details**\n\n"

            output += "📋 System Prompt\n"
            output += "────────────────────────────────────────────────\n"
            output += f"```\n{self.prompt_info.system_prompt}\n```\n\n"

            output += "💬 User Prompt\n"
            output += "────────────────────────────────────────────────\n"
            output += f"```\n{self.prompt_info.user_prompt}\n```\n\n"

            output += "🤖 LLM Response\n"
            output += "────────────────────────────────────────────────\n"
            output += f"```json\n{self.prompt_info.raw_response}\n```\n\n"

        # 결과
        status_emoji = "✅" if self.success else "❌"
        output += f"{status_emoji} **{'완료' if self.success else '실패'}**\n\n"
        output += f"{self.result}\n"

        return output

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({
            "step": self.step,
            "tool_name": self.tool_name,
            "action": self.action,
            "result": self.result,
            "success": self.success
        })
        if self.prompt_info:
            d["prompt_info"] = self.prompt_info.to_dict()
        return d


# =============================================================================
# Final Events
# =============================================================================

@dataclass
class FinalAnswerEvent(ExecutionEvent):
    """최종 답변 이벤트"""
    answer: str

    def to_display(self) -> str:
        output = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        output += self.answer
        return output

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d["answer"] = self.answer
        return d


@dataclass
class ErrorEvent(ExecutionEvent):
    """에러 이벤트"""
    error_message: str

    def to_display(self) -> str:
        return f"❌ **오류 발생**\n\n{self.error_message}"

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d["error_message"] = self.error_message
        return d
