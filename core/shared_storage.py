"""
SharedStorage Module
====================
Multi-Agent Chatbot의 핵심 데이터 저장소.
Agent/Tool 간 데이터 공유 및 실행 결과 관리를 담당.

구조:
- context: 현재 작업 컨텍스트 (user_query, current_plan, current_step)
- results: step별 실행 결과 리스트
- history: 완료된 세션들의 히스토리
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid


# =============================================================================
# Debug Logger
# =============================================================================
class DebugLevel(Enum):
    INFO = "INFO"
    DEBUG = "DEBUG"
    WARN = "WARN"
    ERROR = "ERROR"


class DebugLogger:
    """콘솔 디버그 로거"""
    
    def __init__(self, module_name: str, enabled: bool = True):
        self.module_name = module_name
        self.enabled = enabled
    
    def _log(self, level: DebugLevel, message: str, data: Any = None):
        if not self.enabled:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        prefix = f"[{timestamp}] [{level.value}] [{self.module_name}]"
        
        print(f"{prefix} {message}")
        if data is not None:
            if isinstance(data, (dict, list)):
                print(f"{prefix} └─ Data: {json.dumps(data, ensure_ascii=False, indent=2, default=str)}")
            else:
                print(f"{prefix} └─ Data: {data}")
    
    def info(self, message: str, data: Any = None):
        self._log(DebugLevel.INFO, message, data)
    
    def debug(self, message: str, data: Any = None):
        self._log(DebugLevel.DEBUG, message, data)
    
    def warn(self, message: str, data: Any = None):
        self._log(DebugLevel.WARN, message, data)
    
    def error(self, message: str, data: Any = None):
        self._log(DebugLevel.ERROR, message, data)


# =============================================================================
# Data Classes
# =============================================================================
# ExecutionResult는 제거되고 ToolResult로 통합됨
# from core.base_tool import ToolResult 사용


@dataclass
class Context:
    """현재 작업 컨텍스트"""
    user_query: str
    current_plan: List[str] = field(default_factory=list)
    current_step: int = 0
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Context":
        return cls(**data)


@dataclass
class SessionHistory:
    """완료된 세션 히스토리"""
    session_id: str
    user_query: str
    results: List[Dict]
    final_response: str
    status: str             # "completed" 또는 "failed"
    created_at: str
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SessionHistory":
        return cls(**data)


# =============================================================================
# SharedStorage Main Class
# =============================================================================
class SharedStorage:
    """
    Multi-Agent Chatbot의 중앙 데이터 저장소
    
    역할:
    1. context: 현재 작업 상태 관리
    2. results: 실행 결과 누적
    3. history: 완료된 세션 보관
    4. get_summary(): LLM에 전달할 컨텍스트 요약 생성
    """
    
    # Output truncate 설정
    OUTPUT_PREVIEW_LENGTH = 500
    
    def __init__(self, debug_enabled: bool = True):
        from core.base_tool import ToolResult  # 순환 import 방지를 위해 여기서 import

        self.logger = DebugLogger("SharedStorage", enabled=debug_enabled)

        # 핵심 데이터 구조
        self._context: Optional[Context] = None
        self._results: List[ToolResult] = []
        self._history: List[SessionHistory] = []

        self.logger.info("SharedStorage 초기화 완료")
    
    # =========================================================================
    # Context 관리
    # =========================================================================
    def start_session(self, user_query: str, plan: List[str] = None) -> str:
        """새 세션 시작"""
        self._context = Context(
            user_query=user_query,
            current_plan=plan or [],
            current_step=0
        )
        self._results = []
        
        self.logger.info(f"새 세션 시작: {self._context.session_id}", {
            "user_query": user_query,
            "plan": plan
        })
        
        return self._context.session_id
    
    def update_plan(self, plan: List[str]):
        """실행 계획 업데이트"""
        if not self._context:
            self.logger.error("활성 세션 없음 - start_session() 먼저 호출 필요")
            raise RuntimeError("No active session")
        
        old_plan = self._context.current_plan
        self._context.current_plan = plan
        
        self.logger.debug("실행 계획 업데이트", {
            "old_plan": old_plan,
            "new_plan": plan
        })
    
    def advance_step(self) -> int:
        """다음 step으로 이동"""
        if not self._context:
            raise RuntimeError("No active session")
        
        self._context.current_step += 1
        self.logger.debug(f"Step 이동: {self._context.current_step - 1} → {self._context.current_step}")
        
        return self._context.current_step
    
    def get_context(self) -> Optional[Dict]:
        """현재 context 반환"""
        if not self._context:
            return None
        return self._context.to_dict()
    
    # =========================================================================
    # Results 관리
    # =========================================================================
    def add_result(self, tool_result):
        """
        실행 결과 추가 (ToolResult를 직접 받음)

        Args:
            tool_result: ToolResult 객체

        Returns:
            ToolResult: 저장된 실행 결과
        """
        from core.base_tool import ToolResult

        if not self._context:
            raise RuntimeError("No active session")

        # step 번호 자동 할당 (설정되지 않은 경우)
        if tool_result.step == 0:
            self.logger.error("Step 정보가 기록되지 않음")
            raise RuntimeError("Step info is missing from ToolResult")

        self._results.append(tool_result)

        # 로그 출력 (output은 truncate)
        output_preview = self._truncate_output(str(tool_result.output))
        self.logger.info(f"결과 추가: Step {tool_result.step} - {tool_result.executor}.{tool_result.action}", {
            "status": tool_result.status,
            "output_preview": output_preview,
            "error": tool_result.error
        })

        return tool_result
    
    def get_results(self) -> List[Dict]:
        """모든 실행 결과 반환"""
        return [r.to_dict() for r in self._results]
    
    def get_last_result(self) -> Optional[Dict]:
        """마지막 실행 결과 반환"""
        if not self._results:
            return None
        return self._results[-1].to_dict()
    
    def get_last_output(self) -> Any:
        """마지막 실행의 output만 반환"""
        if not self._results:
            return None
        return self._results[-1].output
    
    def get_result_by_step(self, step: int) -> Optional[Dict]:
        """특정 step의 결과 반환"""
        for r in self._results:
            if r.step == step:
                return r.to_dict()
        return None
    
    def get_output_by_step(self, step: int) -> Any:
        """특정 step의 output만 반환"""
        for r in self._results:
            if r.step == step:
                return r.output
        return None
    
    # =========================================================================
    # History 관리
    # =========================================================================
    def complete_session(self, final_response: str, status: str = "completed") -> SessionHistory:
        """현재 세션 완료 및 히스토리에 저장"""
        if not self._context:
            raise RuntimeError("No active session")
        
        session = SessionHistory(
            session_id=self._context.session_id,
            user_query=self._context.user_query,
            results=[r.to_dict() for r in self._results],
            final_response=final_response,
            status=status,
            created_at=self._context.created_at
        )
        
        self._history.append(session)
        
        self.logger.info(f"세션 완료: {session.session_id}", {
            "status": status,
            "total_steps": len(self._results),
            "final_response_preview": final_response[:200] + "..." if len(final_response) > 200 else final_response
        })
        
        return session
    
    def get_history(self) -> List[Dict]:
        """전체 히스토리 반환"""
        return [h.to_dict() for h in self._history]
    
    def get_history_by_session_id(self, session_id: str) -> Optional[Dict]:
        """특정 세션 히스토리 반환"""
        for h in self._history:
            if h.session_id == session_id:
                return h.to_dict()
        return None
    
    # =========================================================================
    # Summary 생성 (LLM 전달용)
    # =========================================================================
    def get_summary(self, include_all_outputs: bool = True) -> str:
        """
        LLM에 전달할 현재 상태 요약
        
        Args:
            include_all_outputs: True면 모든 결과 포함, False면 마지막 결과만
        """
        if not self._context:
            return "활성 세션 없음"
        
        # 결과 요약 생성
        results_summary = []
        target_results = self._results if include_all_outputs else self._results[-1:] if self._results else []
        
        for r in target_results:
            output_preview = self._truncate_output(r.output)
            status_icon = "✓" if r.status == "success" else "✗"
            results_summary.append(
                f"[Step {r.step}] {status_icon} {r.executor}.{r.action}\n"
                f"  - Input: {json.dumps(r.input, ensure_ascii=False, default=str)}\n"
                f"  - Output: {output_preview}"
            )
        
        # 현재 계획 상태
        plan_status = []
        for i, step in enumerate(self._context.current_plan):
            if i < self._context.current_step:
                plan_status.append(f"  ✓ {i+1}. {step}")
            elif i == self._context.current_step:
                plan_status.append(f"  → {i+1}. {step} (현재)")
            else:
                plan_status.append(f"    {i+1}. {step}")
        
        summary = f"""## 세션 정보
- Session ID: {self._context.session_id}
- 생성 시간: {self._context.created_at}

## 사용자 요청
{self._context.user_query}

## 실행 계획
{chr(10).join(plan_status) if plan_status else "(계획 없음)"}

## 실행 결과
{chr(10).join(results_summary) if results_summary else "(아직 실행된 결과 없음)"}
"""
        
        self.logger.debug("Summary 생성 완료", {
            "session_id": self._context.session_id,
            "results_count": len(target_results)
        })
        
        return summary
    
    # =========================================================================
    # 파일 저장/로드
    # =========================================================================
    def save_to_file(self, filepath: str):
        """현재 상태를 파일에 저장"""
        data = {
            "context": self._context.to_dict() if self._context else None,
            "results": [r.to_dict() for r in self._results],
            "history": [h.to_dict() for h in self._history],
            "saved_at": datetime.now().isoformat()
        }
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"파일 저장 완료: {filepath}", {
            "context_exists": self._context is not None,
            "results_count": len(self._results),
            "history_count": len(self._history)
        })
    
    def load_from_file(self, filepath: str) -> bool:
        """파일에서 상태 로드"""
        if not os.path.exists(filepath):
            self.logger.error(f"파일 없음: {filepath}")
            return False
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            from core.base_tool import ToolResult

            # Context 복원
            if data.get("context"):
                self._context = Context.from_dict(data["context"])
            else:
                self._context = None

            # Results 복원
            self._results = [ToolResult.from_dict(r) for r in data.get("results", [])]

            # History 복원
            self._history = [SessionHistory.from_dict(h) for h in data.get("history", [])]
            
            self.logger.info(f"파일 로드 완료: {filepath}", {
                "context_exists": self._context is not None,
                "results_count": len(self._results),
                "history_count": len(self._history),
                "saved_at": data.get("saved_at")
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"파일 로드 실패: {filepath}", {"error": str(e)})
            return False
    
    # =========================================================================
    # 유틸리티
    # =========================================================================
    def _truncate_output(self, output: Any) -> str:
        """Output을 미리보기용으로 truncate"""
        if output is None:
            return "None"
        
        if isinstance(output, (dict, list)):
            output_str = json.dumps(output, ensure_ascii=False, default=str)
        else:
            output_str = str(output)
        
        if len(output_str) > self.OUTPUT_PREVIEW_LENGTH:
            return output_str[:self.OUTPUT_PREVIEW_LENGTH] + f"... (총 {len(output_str)}자)"
        return output_str
    
    def clear_history(self):
        """히스토리 초기화"""
        count = len(self._history)
        self._history = []
        self.logger.info(f"히스토리 초기화: {count}개 세션 삭제")
    
    def reset(self):
        """전체 초기화"""
        self._context = None
        self._results = []
        self._history = []
        self.logger.info("SharedStorage 전체 초기화 완료")
    
    # =========================================================================
    # 디버그 상태 출력
    # =========================================================================
    def debug_print_state(self):
        """현재 상태를 디버그 출력"""
        print("\n" + "=" * 60)
        print("SharedStorage 현재 상태")
        print("=" * 60)
        
        print("\n[Context]")
        if self._context:
            print(f"  Session ID: {self._context.session_id}")
            print(f"  User Query: {self._context.user_query}")
            print(f"  Current Step: {self._context.current_step}")
            print(f"  Plan: {self._context.current_plan}")
        else:
            print("  (활성 세션 없음)")
        
        print("\n[Results]")
        if self._results:
            for r in self._results:
                print(f"  Step {r.step}: {r.executor}.{r.action} - {r.status}")
        else:
            print("  (결과 없음)")
        
        print("\n[History]")
        if self._history:
            for h in self._history:
                print(f"  {h.session_id}: {h.user_query[:30]}... - {h.status}")
        else:
            print("  (히스토리 없음)")
        
        print("=" * 60 + "\n")
