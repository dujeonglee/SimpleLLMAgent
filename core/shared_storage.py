"""
SharedStorage Module
====================
Multi-Agent Chatbot의 핵심 데이터 저장소.
Agent/Tool 간 데이터 공유 및 실행 결과 관리를 담당.

구조:
- context: 현재 작업 컨텍스트 (user_query, session_id)
- all_results: 모든 세션의 실행 결과 영구 저장소 (session_id_result_id → ToolResult)
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


# =============================================================================
# SharedStorage Main Class
# =============================================================================
class SharedStorage:
    """
    Multi-Agent Chatbot의 중앙 데이터 저장소

    역할:
    1. context: 현재 작업 상태 관리 (user_query, session_id)
    2. all_results: 모든 세션의 실행 결과 저장소 (session_id_result_id → ToolResult)
    3. get_available_results_summary(): 현재 세션의 결과 요약 생성 (LLM 전달용)
    """
    
    # Output truncate 설정
    OUTPUT_PREVIEW_LENGTH = 500
    
    def __init__(self, debug_enabled: bool = True):
        from core.base_tool import ToolResult  # 순환 import 방지를 위해 여기서 import

        self.logger = DebugLogger("SharedStorage", enabled=debug_enabled)

        # 핵심 데이터 구조
        self._context: Optional[Context] = None
        self._all_results: Dict[str, ToolResult] = {}  # 모든 results (session_id_result_id → ToolResult)

        self.logger.info("SharedStorage 초기화 완료")
    
    # =========================================================================
    # Context 관리
    # =========================================================================
    def start_session(self, user_query: str) -> str:
        """새 세션 시작"""
        self._context = Context(
            user_query=user_query
        )

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

        # 영구 저장소에 추가 (session_id_result_id를 키로 사용)
        composite_key = f"{tool_result.session_id}_{tool_result.result_id}"
        self._all_results[composite_key] = tool_result

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
        """현재 세션의 모든 실행 결과 반환"""
        if not self._context:
            return []
        current_session_results = [
            r for r in self._all_results.values()
            if r.session_id == self._context.session_id
        ]
        # Step 순서대로 정렬
        current_session_results.sort(key=lambda r: r.step)
        return [r.to_dict() for r in current_session_results]

    def get_last_result(self) -> Optional[Dict]:
        """현재 세션의 마지막 실행 결과 반환"""
        results = self.get_results()
        if not results:
            return None
        return results[-1]

    def get_last_output(self) -> Any:
        """현재 세션의 마지막 실행 output만 반환"""
        result = self.get_last_result()
        if not result:
            return None
        return result.get('output')

    def get_result_by_step(self, step: int) -> Optional[Dict]:
        """현재 세션의 특정 step 결과 반환"""
        if not self._context:
            return None
        for r in self._all_results.values():
            if r.session_id == self._context.session_id and r.step == step:
                return r.to_dict()
        return None

    def get_output_by_step(self, step: int) -> Any:
        """현재 세션의 특정 step output만 반환"""
        result = self.get_result_by_step(step)
        if not result:
            return None
        return result.get('output')

    def get_result_by_id(self, result_id: str) -> Optional[Dict]:
        """
        result_id로 결과 조회

        Args:
            result_id: 결과 ID (session_id_result_id 형식)

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
            result_id: 결과 ID (session_id_result_id 형식)

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

    def get_available_results_summary(self, max_output_preview: int = 100) -> str:
        """
        현재 세션의 사용 가능한 결과 목록을 REF 형식으로 반환 (user query 포함)

        Args:
            max_output_preview: Output 미리보기 최대 길이

        Returns:
            str: 사용자 요청 + 포맷된 결과 목록
        """
        sections = []

        # 사용자 요청 (항상 포함)
        if self._context:
            sections.append(f"## User Request\n{self._context.user_query}")

        # 현재 세션의 결과 목록 가져오기
        if not self._context:
            sections.append("## Execution Results\nNo active session.")
        else:
            current_session_results = [
                r for r in self._all_results.values()
                if r.session_id == self._context.session_id
            ]
            # Step 순서대로 정렬
            current_session_results.sort(key=lambda r: r.step)

            if not current_session_results:
                sections.append("## Execution Results\nNo previous results available.")
            else:
                lines = []
                for r in current_session_results:
                    composite_key = f"{r.session_id}_{r.result_id}"
                    output_preview = self._truncate_output_with_length(r.output, max_output_preview)

                    lines.append(
                        f"- [RESULT:{composite_key}]: Step {r.step} ({r.executor}.{r.action})\n"
                        f"  Preview: {output_preview}"
                    )
                sections.append(f"## Execution Results\n" + "\n".join(lines))

        return "\n\n".join(sections)

    # =========================================================================
    # Session 관리
    # =========================================================================
    def complete_session(self, final_response: str, status: str = "completed"):
        """현재 세션 완료"""
        if not self._context:
            raise RuntimeError("No active session")

        # 현재 세션의 결과 개수 계산
        current_session_results = [
            r for r in self._all_results.values()
            if r.session_id == self._context.session_id
        ]

        self.logger.info(f"세션 완료: {self._context.session_id}", {
            "status": status,
            "total_steps": len(current_session_results),
            "final_response_preview": final_response[:200] + "..." if len(final_response) > 200 else final_response
        })
    
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

    def clear_all_results(self):
        """모든 results 초기화 (영구 저장소 포함)"""
        count = len(self._all_results)
        self._all_results = {}
        self.logger.info(f"모든 Results 초기화: {count}개 결과 삭제")

    def reset(self):
        """전체 초기화 (현재 세션만, all_results는 유지)"""
        self._context = None
        self.logger.info("SharedStorage 초기화 완료 (현재 세션만, all_results 보존)")

    def reset_all(self):
        """완전 초기화 (all_results 포함)"""
        self._context = None
        self._all_results = {}
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
        if self._context:
            current_session_results = [
                r for r in self._all_results.values()
                if r.session_id == self._context.session_id
            ]
            current_session_results.sort(key=lambda r: r.step)

            if current_session_results:
                for r in current_session_results:
                    print(f"  Step {r.step}: {r.executor}.{r.action} - {r.status} (ID: {r.result_id})")
            else:
                print("  (결과 없음)")
        else:
            print("  (활성 세션 없음)")

        print("\n[All Results - Persistent Storage]")
        if self._all_results:
            print(f"  총 {len(self._all_results)}개 결과 저장됨")
            # 최근 5개만 표시
            recent = list(self._all_results.values())[-5:]
            for r in recent:
                print(f"  {r.result_id}: [{r.session_id}] {r.executor}.{r.action} - {r.status}")
        else:
            print("  (저장된 결과 없음)")

        print("=" * 60 + "\n")
