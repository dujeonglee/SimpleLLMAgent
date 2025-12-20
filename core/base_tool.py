"""
BaseTool Module
===============
모든 Tool의 공통 인터페이스를 정의하는 추상 클래스.

핵심 원칙:
- Tool = 순수 실행기 (Input → 실행 → Output)
- 다른 Tool에 대한 의존성 없음
- SharedStorage 직접 접근 안 함
- Tool 간 의존성은 Orchestrator가 관리
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .shared_storage import DebugLogger


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ToolResult:
    """Tool 실행 결과"""
    success: bool                    # 성공 여부
    output: Any                      # 다음 step에서 사용할 데이터
    error: Optional[str] = None      # 에러 메시지 (실패 시)
    metadata: Dict = field(default_factory=dict)  # 부가 정보
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def success_result(cls, output: Any, metadata: Dict = None) -> "ToolResult":
        """성공 결과 생성 헬퍼"""
        return cls(
            success=True,
            output=output,
            error=None,
            metadata=metadata or {}
        )
    
    @classmethod
    def error_result(cls, error: str, metadata: Dict = None) -> "ToolResult":
        """에러 결과 생성 헬퍼"""
        return cls(
            success=False,
            output=None,
            error=error,
            metadata=metadata or {}
        )


@dataclass
class ActionParam:
    """액션 파라미터 정의"""
    name: str                        # 파라미터 이름
    param_type: str                  # "str", "int", "bool", "dict", "list"
    required: bool                   # 필수 여부
    description: str                 # 설명
    default: Any = None              # 기본값 (optional일 때)
    
    def to_dict(self) -> Dict:
        result = {
            "name": self.name,
            "type": self.param_type,
            "description": self.description,
            "required": self.required
        }
        if not self.required and self.default is not None:
            result["default"] = self.default
        return result


@dataclass
class ActionSchema:
    """액션 스키마 정의"""
    name: str                        # 액션 이름
    description: str                 # 액션 설명
    params: List[ActionParam]        # 파라미터 목록
    output_type: str                 # "str", "dict", "list", "bool", "None"
    output_description: str          # Output 설명 (한 줄)
    
    def to_dict(self) -> Dict:
        """LLM 전달용 dict 변환"""
        # parameters를 JSON Schema 형식으로 변환
        properties = {}
        required = []
        
        for param in self.params:
            prop = {
                "type": self._convert_type(param.param_type),
                "description": param.description
            }
            if not param.required and param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            },
            "output": {
                "type": self._convert_type(self.output_type),
                "description": self.output_description
            }
        }
    
    def _convert_type(self, param_type: str) -> str:
        """Python 타입을 JSON Schema 타입으로 변환"""
        type_map = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "dict": "object",
            "list": "array",
            "None": "null"
        }
        return type_map.get(param_type, "string")


# =============================================================================
# BaseTool Abstract Class
# =============================================================================

class BaseTool(ABC):
    """
    모든 Tool의 기본 추상 클래스
    
    자식 클래스는 다음을 구현해야 함:
    - name: Tool 이름
    - description: Tool 설명
    - _define_actions(): 지원하는 action들 정의
    - _execute_action(): 실제 action 실행 로직
    """
    
    # 자식 클래스에서 정의
    name: str = "base_tool"
    description: str = "Base tool description"
    
    def __init__(self, debug_enabled: bool = True):
        self.logger = DebugLogger(f"Tool:{self.name}", enabled=debug_enabled)
        self._actions: Dict[str, ActionSchema] = self._define_actions()
        
        self.logger.info(f"Tool 초기화 완료", {
            "name": self.name,
            "actions": list(self._actions.keys())
        })
    
    # =========================================================================
    # Abstract Methods (자식 클래스 필수 구현)
    # =========================================================================
    
    @abstractmethod
    def _define_actions(self) -> Dict[str, ActionSchema]:
        """
        Tool이 지원하는 action들 정의
        
        Returns:
            Dict[str, ActionSchema]: action 이름 → 스키마 매핑
        """
        pass
    
    @abstractmethod
    def _execute_action(self, action: str, params: Dict) -> ToolResult:
        """
        실제 action 실행 로직
        
        Args:
            action: 실행할 action 이름
            params: 검증된 파라미터
            
        Returns:
            ToolResult: 실행 결과
        """
        pass
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def execute(self, action: str, params: Dict = None) -> ToolResult:
        """
        Tool action 실행 (공통 로직)
        
        1. action 존재 확인
        2. params 검증
        3. _execute_action 호출
        4. 결과 포맷팅 (성공 시)
        5. 결과 로깅
        """
        params = params or {}
        start_time = datetime.now()
        
        self.logger.info(f"실행 시작: {action}", {"params": params})
        
        # 1. action 존재 확인
        if action not in self._actions:
            error_msg = f"Action '{action}' not found. Available: {list(self._actions.keys())}"
            self.logger.error(error_msg)
            return ToolResult.error_result(error_msg)
        
        # 2. params 검증
        is_valid, error_msg = self.validate_params(action, params)
        if not is_valid:
            self.logger.error(f"파라미터 검증 실패: {error_msg}")
            return ToolResult.error_result(error_msg)
        
        # 3. 기본값 적용
        params = self._apply_defaults(action, params)
        
        # 4. 실행
        try:
            result = self._execute_action(action, params)
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            self.logger.error(error_msg, {"exception": type(e).__name__})
            result = ToolResult.error_result(error_msg)
        
        # 5. 실행 시간 기록
        elapsed = (datetime.now() - start_time).total_seconds()
        result.metadata["execution_time"] = elapsed
        result.metadata["action"] = action
        
        # 6. 성공 시 결과 포맷팅 (output이 명확한 메시지가 아닌 경우)
        if result.success:
            result = self._format_success_result(action, params, result)
        
        # 7. 결과 로깅
        if result.success:
            output_preview = self._truncate_output(result.output)
            self.logger.info(f"실행 완료: {action}", {
                "success": True,
                "output_preview": output_preview,
                "execution_time": f"{elapsed:.3f}s"
            })
        else:
            self.logger.error(f"실행 실패: {action}", {
                "error": result.error,
                "execution_time": f"{elapsed:.3f}s"
            })
        
        return result
    
    def _format_success_result(self, action: str, params: Dict, result: ToolResult) -> ToolResult:
        """
        성공 결과를 LLM이 이해하기 쉬운 메시지로 포맷팅
        
        - output이 이미 명확한 메시지(✅로 시작)면 그대로 반환
        - output이 데이터(파일 내용, 리스트 등)면 그대로 반환
        - output이 불명확하면 자동 포맷팅
        
        하위 클래스에서 오버라이드 가능
        """
        output = result.output
        
        # 이미 명확한 메시지인 경우 그대로 반환
        if isinstance(output, str) and (output.startswith("✅") or output.startswith("❌")):
            return result
        
        # 데이터를 반환하는 action은 포맷팅하지 않음 (하위 클래스에서 정의)
        data_actions = getattr(self, '_data_returning_actions', set())
        if action in data_actions:
            return result
        
        # 리스트나 딕셔너리는 데이터이므로 포맷팅하지 않음
        if isinstance(output, (list, dict)):
            return result
        
        # 긴 문자열(줄바꿈 포함)은 데이터로 간주
        if isinstance(output, str) and (len(output) > 100 or '\n' in output):
            return result
        
        # output이 너무 짧거나 단순한 값인 경우 자동 포맷팅
        needs_formatting = (
            output is None or
            isinstance(output, bool) or
            isinstance(output, (int, float)) or
            (isinstance(output, str) and len(output) < 50)
        )
        
        if needs_formatting:
            # metadata에서 주요 정보 추출
            details = []
            metadata = result.metadata or {}
            
            # 공통 정보
            for key in ['filename', 'size', 'count', 'total', 'url', 'title']:
                if key in metadata:
                    value = metadata[key]
                    if key == 'size' and isinstance(value, (int, float)):
                        value = f"{value} bytes" if value < 1024 else f"{value/1024:.1f}KB"
                    details.append(f"{key}={value}")
            
            # 포맷팅된 메시지 생성
            formatted_output = f"✅ {self.name}.{action} completed successfully"
            if details:
                formatted_output += f" ({', '.join(details)})"
            if output is not None and output != "" and not isinstance(output, bool):
                formatted_output += f"\n→ Result: {output}"
            
            return ToolResult(
                success=result.success,
                output=formatted_output,
                error=result.error,
                metadata=result.metadata
            )
        
        return result
    
    def validate_params(self, action: str, params: Dict) -> Tuple[bool, Optional[str]]:
        """
        파라미터 유효성 검증
        
        Returns:
            Tuple[bool, Optional[str]]: (성공 여부, 에러 메시지)
        """
        if action not in self._actions:
            return False, f"Action '{action}' not found"
        
        schema = self._actions[action]
        
        for param_def in schema.params:
            # 필수 파라미터 체크
            if param_def.required and param_def.name not in params:
                return False, f"Required parameter '{param_def.name}' is missing"
            
            # 타입 체크 (값이 있는 경우만)
            if param_def.name in params:
                value = params[param_def.name]
                if not self._check_type(value, param_def.param_type):
                    return False, (
                        f"Parameter '{param_def.name}' has wrong type. "
                        f"Expected {param_def.param_type}, got {type(value).__name__}"
                    )
        
        return True, None
    
    def get_schema(self) -> Dict:
        """
        LLM에게 전달할 Tool 스키마 반환
        
        Returns:
            Dict: Tool 정보 (이름, 설명, actions)
        """
        actions_schema = {}
        for action_name, action_schema in self._actions.items():
            actions_schema[action_name] = action_schema.to_dict()
        
        return {
            "name": self.name,
            "description": self.description,
            "actions": actions_schema
        }
    
    def get_action_names(self) -> List[str]:
        """지원하는 action 이름 목록"""
        return list(self._actions.keys())
    
    def get_action_schema(self, action: str) -> Optional[ActionSchema]:
        """특정 action의 스키마 반환"""
        return self._actions.get(action)
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _apply_defaults(self, action: str, params: Dict) -> Dict:
        """optional 파라미터에 기본값 적용"""
        schema = self._actions[action]
        result = params.copy()
        
        for param_def in schema.params:
            if param_def.name not in result and param_def.default is not None:
                result[param_def.name] = param_def.default
                self.logger.debug(f"기본값 적용: {param_def.name}={param_def.default}")
        
        return result
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """타입 체크"""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "dict": dict,
            "list": list,
            "None": type(None)
        }
        
        expected = type_map.get(expected_type)
        if expected is None:
            return True  # 알 수 없는 타입은 통과
        
        return isinstance(value, expected)
    
    def _truncate_output(self, output: Any, max_length: int = 200) -> str:
        """Output을 로깅용으로 truncate"""
        if output is None:
            return "None"
        
        if isinstance(output, (dict, list)):
            output_str = json.dumps(output, ensure_ascii=False, default=str)
        else:
            output_str = str(output)
        
        if len(output_str) > max_length:
            return output_str[:max_length] + f"... (총 {len(output_str)}자)"
        return output_str


# =============================================================================
# Tool Registry (Optional - Orchestrator에서 사용)
# =============================================================================

class ToolRegistry:
    """
    Tool 등록 및 조회를 위한 레지스트리
    """
    
    def __init__(self, debug_enabled: bool = True):
        self.logger = DebugLogger("ToolRegistry", enabled=debug_enabled)
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Tool 등록"""
        self._tools[tool.name] = tool
        self.logger.info(f"Tool 등록: {tool.name}", {
            "actions": tool.get_action_names()
        })
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Tool 조회"""
        return self._tools.get(name)
    
    def get_all(self) -> List[BaseTool]:
        """모든 Tool 반환"""
        return list(self._tools.values())
    
    def get_all_schemas(self) -> List[Dict]:
        """모든 Tool의 스키마 반환 (LLM 전달용)"""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def list_tools(self) -> List[str]:
        """등록된 Tool 이름 목록"""
        return list(self._tools.keys())
    
    def execute(self, tool_name: str, action: str, params: Dict = None) -> ToolResult:
        """
        Tool 이름으로 직접 실행
        
        Args:
            tool_name: Tool 이름
            action: Action 이름
            params: 파라미터
            
        Returns:
            ToolResult: 실행 결과
        """
        tool = self.get(tool_name)
        if tool is None:
            error_msg = f"Tool '{tool_name}' not found. Available: {self.list_tools()}"
            self.logger.error(error_msg)
            return ToolResult.error_result(error_msg)
        
        return tool.execute(action, params)
