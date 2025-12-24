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
    """File operations tool for reading, writing, modifying, and deleting files"""

    name = "file_tool"
    description = "Read, write, modify, and delete files within the workspace"
    
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
                description="Read the contents of a file from the workspace",
                params=[
                    ActionParam("read_path", "str", True, f"Path to the file to read (relative to {self.base_path})"),
                    ActionParam("read_encoding", "str", False, "Character encoding of the file (e.g., utf-8, ascii, latin-1)", "utf-8")
                ],
                output_type="str",
                output_description="The complete text content of the file"
            ),
            "write": ActionSchema(
                name="write",
                description="Write content to a file (creates new file or overwrites existing file if overwrite=true)",
                params=[
                    ActionParam("write_path", "str", True, f"Path where the file will be saved (relative to {self.base_path})"),
                    ActionParam("write_content", "str", True, "The text content to write to the file"),
                    ActionParam("write_overwrite", "bool", False, "Whether to overwrite the file if it already exists (default: false for safety)", False),
                    ActionParam("write_encoding", "str", False, "Character encoding to use when writing the file (e.g., utf-8, ascii)", "utf-8")
                ],
                output_type="str",
                output_description="The written content echoed back for confirmation"
            ),
            "append": ActionSchema(
                name="append",
                description="Append content to the end of an existing file without overwriting existing content",
                params=[
                    ActionParam("append_path", "str", True, f"Path to the file to append to (must exist)"),
                    ActionParam("append_content", "str", True, "The text content to append at the end of the file"),
                    ActionParam("append_encoding", "str", False, "Character encoding to use when appending (e.g., utf-8, ascii)", "utf-8")
                ],
                output_type="str",
                output_description="Success message with details about bytes appended and total file size"
            ),
            "delete": ActionSchema(
                name="delete",
                description="Permanently delete a file from the workspace",
                params=[
                    ActionParam("delete_path", "str", True, f"Path to the file to delete (relative to {self.base_path})")
                ],
                output_type="bool",
                output_description="Success message confirming file deletion"
            ),
            "exists": ActionSchema(
                name="exists",
                description="Check whether a file or directory exists at the specified path",
                params=[
                    ActionParam("exists_path", "str", True, f"Path to check for existence (relative to {self.base_path})")
                ],
                output_type="bool",
                output_description="Message indicating whether the path exists and whether it's a file or directory"
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
            
            return ToolResult.success_result(
                output=content,
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
