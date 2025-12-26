"""
SharedStorage Module
====================
Multi-Agent Chatbot의 핵심 데이터 저장소.
Agent/Tool 간 데이터 공유 및 실행 결과 관리를 담당.

구조:
- context: 현재 작업 컨텍스트 (user_query, session_id)
- results: step별 실행 결과 리스트
- history: 완료된 세션들의 히스토리
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import uuid

from .debug_logger import DebugLogger


# =============================================================================
# Data Classes
# =============================================================================
# ExecutionResult는 제거되고 ToolResult로 통합됨
# from core.base_tool import ToolResult 사용


@dataclass
class Context:
    """현재 작업 컨텍스트"""
    user_query: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

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
        self._results: List[ToolResult] = []  # 현재 세션의 results
        self._all_results: Dict[str, ToolResult] = {}  # 모든 results (result_id → ToolResult)
        self._history: List[SessionHistory] = []

        self.logger.info("SharedStorage 초기화 완료")
    
    # =========================================================================
    # Context 관리
    # =========================================================================
    def start_session(self, user_query: str) -> str:
        """새 세션 시작"""
        self._context = Context(
            user_query=user_query
        )
        self._results = []

        self.logger.info(f"새 세션 시작: {self._context.session_id}", {
            "user_query": user_query
        })

        return self._context.session_id
    
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

        # session_id 설정
        tool_result.session_id = self._context.session_id

        # 현재 세션의 results에 추가
        self._results.append(tool_result)

        # 영구 저장소에도 추가 (result_id를 키로 사용)
        self._all_results[tool_result.result_id] = tool_result

        # 로그 출력 (output은 truncate)
        output_preview = self._truncate_output(str(tool_result.output))
        self.logger.info(f"결과 추가: Step {tool_result.step} - {tool_result.executor}.{tool_result.action}", {
            "result_id": tool_result.result_id,
            "session_id": tool_result.session_id,
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

    def get_result_by_id(self, result_id: str) -> Optional[Dict]:
        """
        result_id로 결과 조회 (모든 세션 검색)

        Args:
            result_id: 결과 ID

        Returns:
            Dict: 결과 dict 또는 None
        """
        result = self._all_results.get(result_id)
        if result:
            return result.to_dict()
        return None

    def get_output_by_id(self, result_id: str) -> Any:
        """
        result_id로 output만 조회

        Args:
            result_id: 결과 ID

        Returns:
            Any: output 또는 None
        """
        result = self._all_results.get(result_id)
        if result:
            return result.output
        return None

    def get_results_by_session(self, session_id: str) -> List[Dict]:
        """
        특정 세션의 모든 결과 조회

        Args:
            session_id: 세션 ID

        Returns:
            List[Dict]: 결과 리스트
        """
        results = [r for r in self._all_results.values() if r.session_id == session_id]
        return [r.to_dict() for r in results]

    def list_all_result_ids(self) -> List[str]:
        """모든 result_id 목록 반환"""
        return list(self._all_results.keys())

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
            created_at=datetime.now().isoformat()
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
    def get_summary(
        self,
        include_all_outputs: bool = True,
        include_previous_sessions: bool = False,
        include_plan: bool = True,
        include_session_info: bool = False,
        max_output_length: int = None,
        max_previous_sessions: int = 10
    ) -> str:
        """
        LLM에 전달할 현재 상태 요약 (영어 기반, 토큰 효율 최적화)

        Args:
            include_all_outputs: True면 모든 결과 포함, False면 마지막 결과만
            include_previous_sessions: True면 이전 세션의 결과도 포함
            include_plan: 실행 계획 포함 여부
            include_session_info: 세션 ID, 생성 시간 등 메타데이터 포함 여부
            max_output_length: Output 미리보기 최대 길이 (None이면 기본값 사용)
            max_previous_sessions: 이전 세션에서 가져올 최대 결과 수
        """
        if not self._context:
            return "No active session"

        sections = []

        # 1. 세션 정보 (옵션)
        if include_session_info:
            sections.append(
                f"## Session Info\n"
                f"- ID: {self._context.session_id}"
            )

        # 2. 사용자 요청 (항상 포함)
        sections.append(f"## User Request\n{self._context.user_query}")

        # 3. 실행 결과
        target_results = self._results if include_all_outputs else (self._results[-1:] if self._results else [])

        if target_results:
            results_lines = []
            output_max = max_output_length or self.OUTPUT_PREVIEW_LENGTH

            for r in target_results:
                output_preview = self._truncate_output_with_length(r.output, output_max)
                status_icon = "✓" if r.status == "success" else "✗"

                # Input 간소화: 중요한 파라미터만
                input_summary = self._summarize_input(r.input)

                results_lines.append(
                    f"[Step {r.step}] {status_icon} {r.executor}.{r.action}\n"
                    f"  Parameters: {input_summary}\n"
                    f"  Output: {output_preview}\n"
                    f"  Ref: [RESULT:{r.result_id}]"
                )

            sections.append(f"## Execution Results\n" + "\n\n".join(results_lines))
        else:
            sections.append("## Execution Results\nNo results yet")

        # 5. 이전 세션 결과 (옵션)
        if include_previous_sessions and self._all_results:
            other_results = [
                r for r in self._all_results.values()
                if r.session_id != self._context.session_id
            ]

            if other_results:
                prev_lines = []
                for r in other_results[-max_previous_sessions:]:
                    output_preview = self._truncate_output_with_length(
                        r.output,
                        max_output_length or 100  # 이전 세션은 더 짧게
                    )
                    status_icon = "✓" if r.status == "success" else "✗"
                    prev_lines.append(
                        f"  {status_icon} {r.executor}.{r.action}\n"
                        f"    Output: {output_preview}\n"
                        f"    Ref: [RESULT:{r.result_id}]"
                    )

                sections.append(
                    f"## Previous Sessions (Reference Only)\n" + "\n\n".join(prev_lines)
                )

        self.logger.debug("Summary 생성 완료", {
            "session_id": self._context.session_id,
            "results_count": len(target_results),
            "previous_results_included": include_previous_sessions,
            "plan_included": include_plan,
            "session_info_included": include_session_info
        })

        return "\n\n".join(sections)
    
    # =========================================================================
    # 파일 저장/로드
    # =========================================================================
    def save_to_file(self, filepath: str):
        """현재 상태를 파일에 저장"""
        data = {
            "context": self._context.to_dict() if self._context else None,
            "results": [r.to_dict() for r in self._results],
            "all_results": {rid: r.to_dict() for rid, r in self._all_results.items()},
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
            "all_results_count": len(self._all_results),
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

            # All Results 복원
            all_results_data = data.get("all_results", {})
            self._all_results = {rid: ToolResult.from_dict(r) for rid, r in all_results_data.items()}

            # History 복원
            self._history = [SessionHistory.from_dict(h) for h in data.get("history", [])]

            self.logger.info(f"파일 로드 완료: {filepath}", {
                "context_exists": self._context is not None,
                "results_count": len(self._results),
                "all_results_count": len(self._all_results),
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
        """Output을 미리보기용으로 truncate (기본 길이 사용)"""
        return self._truncate_output_with_length(output, self.OUTPUT_PREVIEW_LENGTH)

    def _truncate_output_with_length(self, output: Any, max_length: int) -> str:
        """유연한 길이로 Output을 truncate"""
        if output is None:
            return "None"

        if isinstance(output, (dict, list)):
            output_str = json.dumps(output, ensure_ascii=False, default=str)
        else:
            output_str = str(output)

        if len(output_str) > max_length:
            return output_str[:max_length] + f"... ({len(output_str)} chars total)"
        return output_str

    def _summarize_input(self, input_data: Dict) -> str:
        """Input 데이터를 간략하게 요약 (중요한 것만)"""
        if not input_data:
            return "{}"

        # 3개 이하 파라미터면 전체 표시
        if len(input_data) <= 3:
            return json.dumps(input_data, ensure_ascii=False, default=str)

        # 많으면 키만 표시
        keys = list(input_data.keys())
        return f"{{{', '.join(keys[:3])}, ... ({len(input_data)} params)}}"
    
    def clear_history(self):
        """히스토리 초기화"""
        count = len(self._history)
        self._history = []
        self.logger.info(f"히스토리 초기화: {count}개 세션 삭제")

    def clear_all_results(self):
        """모든 results 초기화 (영구 저장소 포함)"""
        count = len(self._all_results)
        self._all_results = {}
        self.logger.info(f"모든 Results 초기화: {count}개 결과 삭제")

    def reset(self):
        """전체 초기화 (현재 세션만, all_results는 유지)"""
        self._context = None
        self._results = []
        self.logger.info("SharedStorage 초기화 완료 (현재 세션만, all_results 보존)")

    def reset_all(self):
        """완전 초기화 (all_results 포함)"""
        self._context = None
        self._results = []
        self._all_results = {}
        self._history = []
        self.logger.info("SharedStorage 완전 초기화 완료")
    
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
        else:
            print("  (활성 세션 없음)")
        
        print("\n[Results - Current Session]")
        if self._results:
            for r in self._results:
                print(f"  Step {r.step}: {r.executor}.{r.action} - {r.status} (ID: {r.result_id})")
        else:
            print("  (결과 없음)")

        print("\n[All Results - Persistent Storage]")
        if self._all_results:
            print(f"  총 {len(self._all_results)}개 결과 저장됨")
            # 최근 5개만 표시
            recent = list(self._all_results.values())[-5:]
            for r in recent:
                print(f"  {r.result_id}: [{r.session_id}] {r.executor}.{r.action} - {r.status}")
        else:
            print("  (저장된 결과 없음)")

        print("\n[History]")
        if self._history:
            for h in self._history:
                print(f"  {h.session_id}: {h.user_query[:30]}... - {h.status}")
        else:
            print("  (히스토리 없음)")

        print("=" * 60 + "\n")
