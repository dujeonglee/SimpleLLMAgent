"""
FileTool Module
===============
파일 읽기, 쓰기, 수정, 삭제 기능을 제공하는 Tool.
"""

import os
from typing import Dict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_tool import BaseTool, ActionSchema, ActionParam, ToolResult


class FileTool(BaseTool):
    """파일 작업 Tool"""
    
    name = "file_tool"
    description = "파일 읽기, 쓰기, 수정, 삭제"
    
    # 데이터를 그대로 반환해야 하는 action (포맷팅 제외)
    _data_returning_actions = {"read", "list_dir"}
    
    def __init__(self, base_path: str = None, debug_enabled: bool = True):
        """
        Args:
            base_path: 파일 작업의 기본 경로 (보안상 이 경로 하위만 접근 가능)
            debug_enabled: 디버그 로깅 활성화
        """
        self.base_path = base_path or os.getcwd()
        super().__init__(debug_enabled=debug_enabled)
        self.logger.info(f"Base path 설정: {self.base_path}")
    
    def _define_actions(self) -> Dict[str, ActionSchema]:
        return {
            "read": ActionSchema(
                name="read",
                description="파일 내용 읽기",
                params=[
                    ActionParam("read_path", "str", True, "읽을 파일 경로"),
                    ActionParam("read_encoding", "str", False, "파일 인코딩", "utf-8")
                ],
                output_type="str",
                output_description="파일 내용"
            ),
            "write": ActionSchema(
                name="write",
                description="파일에 내용 쓰기 (새 파일 생성 또는 덮어쓰기)",
                params=[
                    ActionParam("write_path", "str", True, "저장할 파일 경로"),
                    ActionParam("write_content", "str", True, "작성할 내용"),
                    ActionParam("write_overwrite", "bool", False, "기존 파일 덮어쓰기 여부", False),
                    ActionParam("write_encoding", "str", False, "파일 인코딩", "utf-8")
                ],
                output_type="str",
                output_description="작성된 파일의 절대 경로"
            ),
            "append": ActionSchema(
                name="append",
                description="파일 끝에 내용 추가",
                params=[
                    ActionParam("append_path", "str", True, "추가할 파일 경로"),
                    ActionParam("append_content", "str", True, "추가할 내용"),
                    ActionParam("append_encoding", "str", False, "파일 인코딩", "utf-8")
                ],
                output_type="str",
                output_description="수정된 파일의 절대 경로"
            ),
            "delete": ActionSchema(
                name="delete",
                description="파일 삭제",
                params=[
                    ActionParam("delete_path", "str", True, "삭제할 파일 경로")
                ],
                output_type="bool",
                output_description="삭제 성공 여부"
            ),
            "exists": ActionSchema(
                name="exists",
                description="파일 존재 여부 확인",
                params=[
                    ActionParam("exists_path", "str", True, "확인할 파일 경로")
                ],
                output_type="bool",
                output_description="파일 존재 여부"
            ),
            "list_dir": ActionSchema(
                name="list_dir",
                description="디렉토리 내 파일 목록 조회",
                params=[
                    ActionParam("list_dir_path", "str", True, "디렉토리 경로"),
                    ActionParam("list_dir_pattern", "str", False, "파일 패턴 (예: *.txt)", "*")
                ],
                output_type="list",
                output_description="파일 이름 목록"
            )
        }
    
    def _execute_action(self, action: str, params: Dict) -> ToolResult:
        """실제 action 실행"""
        # prefix 파라미터 추출 헬퍼 (예: read_path -> path)
        def get_param(name: str, default=None):
            prefixed = f"{action}_{name}"
            return params.get(prefixed, default)
        
        if action == "read":
            return self._read(
                get_param("path"),
                get_param("encoding", "utf-8")
            )
        elif action == "write":
            return self._write(
                get_param("path"),
                get_param("content"),
                get_param("overwrite", False),
                get_param("encoding", "utf-8")
            )
        elif action == "append":
            return self._append(
                get_param("path"),
                get_param("content"),
                get_param("encoding", "utf-8")
            )
        elif action == "delete":
            return self._delete(get_param("path"))
        elif action == "exists":
            return self._exists(get_param("path"))
        elif action == "list_dir":
            return self._list_dir(
                get_param("path"),
                get_param("pattern", "*")
            )
        else:
            return ToolResult.error_result(f"Unknown action: {action}")
    
    # =========================================================================
    # Private Action Implementations
    # =========================================================================
    
    def _normalize_path(self, path: str) -> str:
        """경로 정규화 - 절대경로, 네트워크 경로 등에서 파일명/상대경로 추출
        
        Examples:
            \\\\server\\share\\path\\file.txt -> file.txt
            C:\\Users\\...\\workspace\\files\\file.txt -> file.txt
            /home/user/workspace/files/file.txt -> file.txt
            ./subdir/file.txt -> subdir/file.txt
            file.txt -> file.txt
        """
        if not path:
            return path
        
        # 경로 구분자 통일
        normalized = path.replace('\\', '/')
        
        # base_path 패턴 찾기 (workspace/files 또는 유사 패턴)
        base_markers = ['workspace/files/', 'workspace\\files\\', '/files/', '\\files\\']
        
        for marker in base_markers:
            marker_normalized = marker.replace('\\', '/')
            if marker_normalized in normalized:
                # 마커 이후 부분만 추출
                idx = normalized.rfind(marker_normalized)
                extracted = normalized[idx + len(marker_normalized):]
                if extracted:
                    self.logger.debug(f"경로 정규화: {path} -> {extracted}")
                    return extracted
        
        # UNC 경로 (\\server\share\...) 처리
        if normalized.startswith('//'):
            # 파일명만 추출
            filename = os.path.basename(normalized)
            self.logger.debug(f"UNC 경로에서 파일명 추출: {path} -> {filename}")
            return filename
        
        # Windows 절대 경로 (C:\...) 처리
        if len(normalized) > 2 and normalized[1] == ':':
            # 파일명만 추출
            filename = os.path.basename(normalized)
            self.logger.debug(f"절대 경로에서 파일명 추출: {path} -> {filename}")
            return filename
        
        # Unix 절대 경로 (/...) 처리
        if normalized.startswith('/') and not normalized.startswith('./'):
            filename = os.path.basename(normalized)
            self.logger.debug(f"절대 경로에서 파일명 추출: {path} -> {filename}")
            return filename
        
        # ./ 제거
        if normalized.startswith('./'):
            normalized = normalized[2:]
        
        return normalized
    
    def _resolve_path(self, path: str) -> str:
        """경로를 절대 경로로 변환 (base_path 기준)"""
        # 먼저 경로 정규화
        normalized = self._normalize_path(path)
        
        if os.path.isabs(normalized):
            return normalized
        return os.path.join(self.base_path, normalized)
    
    def _validate_path(self, path: str) -> tuple[bool, str]:
        """경로가 base_path 하위인지 검증 (보안)"""
        resolved = os.path.realpath(self._resolve_path(path))
        base_real = os.path.realpath(self.base_path)
        
        if not resolved.startswith(base_real):
            return False, f"Access denied: path must be under {self.base_path}"
        return True, resolved
    
    def _read(self, path: str, encoding: str) -> ToolResult:
        """파일 읽기"""
        is_valid, resolved_or_error = self._validate_path(path)
        if not is_valid:
            return ToolResult.error_result(resolved_or_error)
        
        resolved = resolved_or_error
        
        if not os.path.exists(resolved):
            return ToolResult.error_result(f"File not found: {path}")
        
        if not os.path.isfile(resolved):
            return ToolResult.error_result(f"Not a file: {path}")
        
        try:
            with open(resolved, "r", encoding=encoding) as f:
                content = f.read()
            
            return ToolResult.success_result(
                output=content,
                metadata={
                    "path": resolved,
                    "size": len(content),
                    "encoding": encoding
                }
            )
        except UnicodeDecodeError as e:
            return ToolResult.error_result(f"Encoding error: {str(e)}")
        except Exception as e:
            return ToolResult.error_result(f"Read error: {str(e)}")
    
    def _write(self, path: str, content: str, overwrite: bool, encoding: str) -> ToolResult:
        """파일 쓰기"""
        is_valid, resolved_or_error = self._validate_path(path)
        if not is_valid:
            return ToolResult.error_result(resolved_or_error)
        
        resolved = resolved_or_error
        filename = os.path.basename(resolved)
        
        # 기존 파일 존재 시 overwrite 체크
        file_existed = os.path.exists(resolved)
        if file_existed and not overwrite:
            return ToolResult.error_result(
                f"File already exists: {path}. Set overwrite=true to overwrite."
            )
        
        try:
            # 디렉토리 생성
            dir_path = os.path.dirname(resolved)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(resolved, "w", encoding=encoding) as f:
                f.write(content)
            
            size = len(content)
            action = "overwritten" if file_existed else "created"
            
            # 명확한 성공 메시지
            result_msg = f"✅ File '{filename}' {action} successfully ({size} bytes)"
            
            return ToolResult.success_result(
                output=result_msg,
                metadata={
                    "path": resolved,
                    "filename": filename,
                    "size": size,
                    "encoding": encoding,
                    "action": action
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Write error: {str(e)}")
    
    def _append(self, path: str, content: str, encoding: str) -> ToolResult:
        """파일에 내용 추가"""
        is_valid, resolved_or_error = self._validate_path(path)
        if not is_valid:
            return ToolResult.error_result(resolved_or_error)
        
        resolved = resolved_or_error
        filename = os.path.basename(resolved)
        
        if not os.path.exists(resolved):
            return ToolResult.error_result(f"File not found: {path}")
        
        try:
            with open(resolved, "a", encoding=encoding) as f:
                f.write(content)
            
            # 새 파일 크기 확인
            new_size = os.path.getsize(resolved)
            appended_size = len(content)
            
            # 명확한 성공 메시지
            result_msg = f"✅ Appended {appended_size} bytes to '{filename}' (total: {new_size} bytes)"
            
            return ToolResult.success_result(
                output=result_msg,
                metadata={
                    "path": resolved,
                    "filename": filename,
                    "appended_size": appended_size,
                    "total_size": new_size,
                    "encoding": encoding
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Append error: {str(e)}")
    
    def _delete(self, path: str) -> ToolResult:
        """파일 삭제"""
        is_valid, resolved_or_error = self._validate_path(path)
        if not is_valid:
            return ToolResult.error_result(resolved_or_error)
        
        resolved = resolved_or_error
        filename = os.path.basename(resolved)
        
        if not os.path.exists(resolved):
            return ToolResult.error_result(f"File not found: {path}")
        
        if not os.path.isfile(resolved):
            return ToolResult.error_result(f"Not a file: {path}")
        
        try:
            os.remove(resolved)
            
            # 명확한 성공 메시지
            result_msg = f"✅ File '{filename}' deleted successfully"
            
            return ToolResult.success_result(
                output=result_msg,
                metadata={
                    "deleted_path": resolved,
                    "filename": filename
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"Delete error: {str(e)}")
    
    def _exists(self, path: str) -> ToolResult:
        """파일 존재 확인"""
        is_valid, resolved_or_error = self._validate_path(path)
        if not is_valid:
            return ToolResult.error_result(resolved_or_error)
        
        resolved = resolved_or_error
        filename = os.path.basename(resolved)
        exists = os.path.exists(resolved)
        
        # 명확한 결과 메시지
        if exists:
            is_file = os.path.isfile(resolved)
            file_type = "file" if is_file else "directory"
            result_msg = f"✅ '{filename}' exists ({file_type})"
        else:
            result_msg = f"❌ '{filename}' does not exist"
        
        return ToolResult.success_result(
            output=result_msg,
            metadata={
                "path": resolved,
                "filename": filename,
                "exists": exists,
                "is_file": os.path.isfile(resolved) if exists else None,
                "is_dir": os.path.isdir(resolved) if exists else None
            }
        )
    
    def _list_dir(self, path: str, pattern: str) -> ToolResult:
        """디렉토리 목록 조회"""
        is_valid, resolved_or_error = self._validate_path(path)
        if not is_valid:
            return ToolResult.error_result(resolved_or_error)
        
        resolved = resolved_or_error
        
        if not os.path.exists(resolved):
            return ToolResult.error_result(f"Directory not found: {path}")
        
        if not os.path.isdir(resolved):
            return ToolResult.error_result(f"Not a directory: {path}")
        
        try:
            import fnmatch
            
            all_files = os.listdir(resolved)
            
            if pattern and pattern != "*":
                files = [f for f in all_files if fnmatch.fnmatch(f, pattern)]
            else:
                files = all_files
            
            return ToolResult.success_result(
                output=sorted(files),
                metadata={
                    "path": resolved,
                    "pattern": pattern,
                    "total_count": len(files)
                }
            )
        except Exception as e:
            return ToolResult.error_result(f"List dir error: {str(e)}")
