"""
Workspace Manager Module
========================
파일 업로드, 작업 공간, 설정 관리를 담당합니다.

구조:
    workspace/
    ├── config/
    │   └── llm_config.json    ← LLM 설정 (자동 저장/로드)
    ├── uploads/               ← 업로드 원본 보존
    └── files/                 ← FileTool 작업 공간
"""

import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

from .orchestrator import LLMConfig
from .debug_logger import DebugLogger


@dataclass
class FileInfo:
    """파일 정보"""
    name: str
    size: int                    # bytes
    source: str                  # "upload" or "generated"
    created_at: str
    path: str                    # 상대 경로
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def size_str(self) -> str:
        """사람이 읽기 쉬운 크기"""
        if self.size < 1024:
            return f"{self.size} B"
        elif self.size < 1024 * 1024:
            return f"{self.size / 1024:.1f} KB"
        else:
            return f"{self.size / (1024 * 1024):.1f} MB"


class ConfigManager:
    """
    설정 관리자
    - LLM 설정 자동 저장/로드
    """
    
    def __init__(self, config_path: str = "./workspace/config", debug_enabled: bool = True):
        self.config_path = config_path
        self.llm_config_file = os.path.join(config_path, "llm_config.json")
        self.logger = DebugLogger("ConfigManager", enabled=debug_enabled)
        
        # 디렉토리 생성
        os.makedirs(config_path, exist_ok=True)
        
        self.logger.info("ConfigManager 초기화", {"path": config_path})
    
    def load_initial_llm_config(self) -> LLMConfig:
        """하드코딩된 기본 설정으로 LLMConfig 생성 및 파일 저장"""
        config = LLMConfig(
            base_url="http://localhost:11434",
            model="llama3.2",
            timeout=120,
            temperature=0.7,
            max_tokens=2048,
            num_ctx=4096,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_steps=10
        )
        self.save_llm_config(config)
        self.logger.info("초기 LLM 설정 생성 완료", {"model": config.model})
        return config
    
    def load_llm_config(self) -> LLMConfig:
        """저장된 LLM 설정 로드, 없으면 초기 설정 생성"""
        try:
            if os.path.exists(self.llm_config_file):
                with open(self.llm_config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # last_updated는 LLMConfig에 없으므로 제거
                data.pop("last_updated", None)
                
                config = LLMConfig(**data)
                self.logger.info("LLM 설정 로드 완료", {"model": config.model})
                return config
            else:
                self.logger.info("저장된 설정 없음, 초기 설정 생성")
                return self.load_initial_llm_config()
                
        except Exception as e:
            self.logger.error(f"설정 로드 실패: {str(e)}")
            return self.load_initial_llm_config()
    
    def save_llm_config(self, config: LLMConfig):
        """LLM 설정 저장"""
        try:
            os.makedirs(self.config_path, exist_ok=True)
            
            data = config.to_dict()
            data["last_updated"] = datetime.now().isoformat()
            
            with open(self.llm_config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug("LLM 설정 저장 완료")
            
        except Exception as e:
            self.logger.error(f"설정 저장 실패: {str(e)}")
    
    def reset_llm_config(self) -> LLMConfig:
        """설정을 기본값으로 리셋"""
        return self.load_initial_llm_config()


class WorkspaceManager:
    """
    작업 공간 관리자
    - 파일 업로드/삭제
    - 파일 목록 관리
    - FileTool 연동
    """
    
    def __init__(self, workspace_path: str = "./workspace", debug_enabled: bool = True):
        self.workspace_path = workspace_path
        self.uploads_dir = os.path.join(workspace_path, "uploads")
        self.files_dir = os.path.join(workspace_path, "files")
        self.config_dir = os.path.join(workspace_path, "config")
        self.logger = DebugLogger("WorkspaceManager", enabled=debug_enabled)
        
        # 디렉토리 생성
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 파일 출처 추적 (메모리)
        self._file_sources: Dict[str, str] = {}  # filename -> source
        self.logger.info("WorkspaceManager 초기화", {"path": workspace_path})
    
    def save_upload(self, file_path: str, original_name: str = None) -> Optional[FileInfo]:
        """
        업로드 파일 저장
        
        Args:
            file_path: Gradio가 제공하는 임시 파일 경로
            original_name: 원본 파일명 (없으면 file_path에서 추출)
        
        Returns:
            FileInfo: 저장된 파일 정보
        """
        try:
            if original_name is None:
                original_name = os.path.basename(file_path)
            
            # 1. uploads/에 원본 저장
            upload_path = os.path.join(self.uploads_dir, original_name)
            shutil.copy2(file_path, upload_path)
            
            # 2. files/에 복사 (FileTool 작업용)
            work_path = os.path.join(self.files_dir, original_name)
            shutil.copy2(file_path, work_path)
            
            # 3. 출처 기록
            self._file_sources[original_name] = "upload"
            
            # 4. 파일 정보 반환
            stat = os.stat(work_path)
            file_info = FileInfo(
                name=original_name,
                size=stat.st_size,
                source="upload",
                created_at=datetime.now().isoformat(),
                path=original_name
            )
            
            self.logger.info(f"파일 업로드 완료: {original_name}", {
                "size": file_info.size_str
            })
            
            return file_info
            
        except Exception as e:
            self.logger.error(f"파일 업로드 실패: {str(e)}")
            return None
    
    def list_files(self) -> List[FileInfo]:
        """files/ 디렉토리의 파일 목록 반환"""
        files = []
        
        try:
            for name in os.listdir(self.files_dir):
                file_path = os.path.join(self.files_dir, name)
                
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    source = self._file_sources.get(name, "generated")
                    
                    files.append(FileInfo(
                        name=name,
                        size=stat.st_size,
                        source=source,
                        created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        path=name
                    ))
            
            # 이름순 정렬
            files.sort(key=lambda f: f.name)
            
        except Exception as e:
            self.logger.error(f"파일 목록 조회 실패: {str(e)}")
        return files
    
    def get_file_path(self, filename: str) -> Optional[str]:
        """파일의 전체 경로 반환"""
        path = os.path.join(self.files_dir, filename)
        if os.path.exists(path):
            return path
        return None
    
    def delete_file(self, filename: str) -> bool:
        """파일 삭제"""
        try:
            # files/에서 삭제
            work_path = os.path.join(self.files_dir, filename)
            if os.path.exists(work_path):
                os.remove(work_path)
            
            # uploads/에서도 삭제 (있으면)
            upload_path = os.path.join(self.uploads_dir, filename)
            if os.path.exists(upload_path):
                os.remove(upload_path)
            
            # 출처 정보 삭제
            if filename in self._file_sources:
                del self._file_sources[filename]
            
            self.logger.info(f"파일 삭제 완료: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"파일 삭제 실패: {str(e)}")
            return False
    
    def delete_files(self, filenames: List[str]) -> int:
        """여러 파일 삭제, 삭제된 개수 반환"""
        deleted = 0
        for name in filenames:
            if self.delete_file(name):
                deleted += 1
        return deleted
    
    def delete_all_files(self) -> int:
        """모든 파일 삭제"""
        files = self.list_files()
        return self.delete_files([f.name for f in files])
    
    def get_total_size(self) -> int:
        """전체 파일 크기 (bytes)"""
        return sum(f.size for f in self.list_files())
    
    def get_total_size_str(self) -> str:
        """전체 파일 크기 (사람이 읽기 쉬운 형태)"""
        size = self.get_total_size()
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"
    
    def mark_as_generated(self, filename: str):
        """파일을 'generated'로 표시 (FileTool에서 호출)"""
        if filename not in self._file_sources:
            self._file_sources[filename] = "generated"
    
    def get_files_for_prompt(self) -> str:
        """LLM 프롬프트용 파일 목록 문자열"""
        files = self.list_files()
        if not files:
            return "No files available."
        
        lines = ["Available files in workspace:"]
        for f in files:
            source_icon = "📤" if f.source == "upload" else "🤖"
            lines.append(f"  - {f.name} ({f.size_str}) {source_icon}")
        
        return "\n".join(lines)
