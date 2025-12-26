"""
Debug Logger Module
===================
콘솔 기반 디버그 로깅 유틸리티

Features:
- 타임스탬프가 포함된 로그 출력
- 4가지 로그 레벨 (INFO, DEBUG, WARN, ERROR)
- 구조화된 데이터 출력 (JSON 형식)
- 모듈별 로거 인스턴스 생성
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any


class DebugLevel(Enum):
    """로그 레벨"""
    INFO = "INFO"
    DEBUG = "DEBUG"
    WARN = "WARN"
    ERROR = "ERROR"


class DebugLogger:
    """콘솔 디버그 로거"""

    def __init__(self, module_name: str, enabled: bool = True):
        """
        Initialize Debug Logger

        Args:
            module_name: 로거를 사용하는 모듈 이름
            enabled: False면 로그 출력 안함 (프로덕션 환경)
        """
        self.module_name = module_name
        self.enabled = enabled

    def _log(self, level: DebugLevel, message: str, data: Any = None):
        """내부 로그 출력 메서드"""
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
        """INFO 레벨 로그"""
        self._log(DebugLevel.INFO, message, data)

    def debug(self, message: str, data: Any = None):
        """DEBUG 레벨 로그"""
        self._log(DebugLevel.DEBUG, message, data)

    def warn(self, message: str, data: Any = None):
        """WARN 레벨 로그"""
        self._log(DebugLevel.WARN, message, data)

    def error(self, message: str, data: Any = None):
        """ERROR 레벨 로그"""
        self._log(DebugLevel.ERROR, message, data)
