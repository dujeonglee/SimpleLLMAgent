"""
LLMTool Module
==============
LLM에 질문하고 응답받는 Tool.
Ollama를 기본 백엔드로 사용합니다.
"""

import json
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_tool import BaseTool, ActionSchema, ActionParam, ToolResult


class LLMTool(BaseTool):
    """LLM 질의 Tool"""
    
    name = "llm_tool"
    description = "LLM에 질문하고 응답받기"
    
    def __init__(
        self,
        model: str = "Gemma3:latest",
        base_url: str = "http://192.168.0.30:11434",
        debug_enabled: bool = True,
        use_mock: bool = False
    ):
        """
        Args:
            model: 사용할 Ollama 모델 이름
            base_url: Ollama 서버 URL
            debug_enabled: 디버그 로깅 활성화
            use_mock: True면 Mock 응답 사용
        """
        self.model = model
        self.base_url = base_url
        self.use_mock = use_mock
        
        super().__init__(debug_enabled=debug_enabled)
        
        self.logger.info(f"LLM 설정", {
            "model": model,
            "base_url": base_url,
            "use_mock": use_mock
        })
    
    def _define_actions(self) -> Dict[str, ActionSchema]:
        return {
            "ask": ActionSchema(
                name="ask",
                description="LLM에 자유 질문",
                params=[
                    ActionParam("prompt", "str", True, "질문 내용"),
                    ActionParam("system_prompt", "str", False, "시스템 프롬프트", None)
                ],
                output_type="str",
                output_description="LLM 응답"
            ),
            "analyze": ActionSchema(
                name="analyze",
                description="텍스트 분석 요청",
                params=[
                    ActionParam("content", "str", True, "분석할 내용"),
                    ActionParam("instruction", "str", False, "분석 지시사항", "이 내용을 분석해주세요."),
                    ActionParam("output_format", "str", False, "출력 형식 (text/json)", "text")
                ],
                output_type="str",
                output_description="분석 결과"
            ),
            "summarize": ActionSchema(
                name="summarize",
                description="텍스트 요약",
                params=[
                    ActionParam("content", "str", True, "요약할 내용"),
                    ActionParam("max_length", "int", False, "최대 길이 (글자수)", 500)
                ],
                output_type="str",
                output_description="요약된 내용"
            ),
            "extract": ActionSchema(
                name="extract",
                description="텍스트에서 정보 추출",
                params=[
                    ActionParam("content", "str", True, "추출 대상 텍스트"),
                    ActionParam("extract_type", "str", True, "추출할 정보 유형 (예: errors, keywords, entities)"),
                    ActionParam("additional_instruction", "str", False, "추가 지시사항", None)
                ],
                output_type="dict",
                output_description="추출된 정보 (JSON 형식)"
            )
        }
    
    def _execute_action(self, action: str, params: Dict) -> ToolResult:
        """실제 action 실행"""
        if action == "ask":
            return self._ask(
                params["prompt"],
                params.get("system_prompt")
            )
        elif action == "analyze":
            return self._analyze(
                params["content"],
                params.get("instruction", "이 내용을 분석해주세요."),
                params.get("output_format", "text")
            )
        elif action == "summarize":
            return self._summarize(
                params["content"],
                params.get("max_length", 500)
            )
        elif action == "extract":
            return self._extract(
                params["content"],
                params["extract_type"],
                params.get("additional_instruction")
            )
        else:
            return ToolResult.error_result(f"Unknown action: {action}")
    
    # =========================================================================
    # Private Action Implementations
    # =========================================================================
    
    def _ask(self, prompt: str, system_prompt: Optional[str]) -> ToolResult:
        """자유 질문"""
        if self.use_mock:
            return self._mock_response(prompt)
        
        return self._call_ollama(prompt, system_prompt)
    
    def _analyze(self, content: str, instruction: str, output_format: str) -> ToolResult:
        """텍스트 분석"""
        prompt = f"""다음 내용을 분석해주세요.

## 내용
{content}

## 분석 지시사항
{instruction}

## 출력 형식
{output_format}
"""
        
        system_prompt = "당신은 텍스트 분석 전문가입니다. 주어진 내용을 분석하고 인사이트를 제공하세요."
        
        if self.use_mock:
            return self._mock_analyze(content, instruction)
        
        return self._call_ollama(prompt, system_prompt)
    
    def _summarize(self, content: str, max_length: int) -> ToolResult:
        """텍스트 요약"""
        prompt = f"""다음 내용을 {max_length}자 이내로 요약해주세요.

## 내용
{content}
"""
        
        system_prompt = "당신은 텍스트 요약 전문가입니다. 핵심 내용만 간결하게 요약하세요."
        
        if self.use_mock:
            return self._mock_summarize(content, max_length)
        
        return self._call_ollama(prompt, system_prompt)
    
    def _extract(self, content: str, extract_type: str, additional_instruction: Optional[str]) -> ToolResult:
        """정보 추출"""
        instruction = additional_instruction or ""
        
        prompt = f"""다음 텍스트에서 {extract_type}을(를) 추출해주세요.

## 텍스트
{content}

## 추출할 정보
{extract_type}

{f"## 추가 지시사항: {instruction}" if instruction else ""}

## 출력 형식
JSON 형식으로 출력하세요.
"""
        
        system_prompt = "당신은 정보 추출 전문가입니다. JSON 형식으로만 응답하세요."
        
        if self.use_mock:
            return self._mock_extract(content, extract_type)
        
        result = self._call_ollama(prompt, system_prompt)
        
        if result.success:
            # JSON 파싱 시도
            try:
                parsed = self._parse_json_response(result.output)
                result.output = parsed
            except:
                # JSON 파싱 실패 시 원본 유지
                pass
        
        return result
    
    # =========================================================================
    # Ollama API Call
    # =========================================================================
    
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> ToolResult:
        """Ollama API 호출"""
        try:
            import requests
            
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            self.logger.debug("Ollama API 호출", {
                "model": self.model,
                "prompt_length": len(prompt)
            })
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            output = data.get("response", "")
            
            return ToolResult.success_result(
                output=output,
                metadata={
                    "model": self.model,
                    "prompt_length": len(prompt),
                    "response_length": len(output),
                    "eval_count": data.get("eval_count"),
                    "eval_duration": data.get("eval_duration")
                }
            )
            
        except ImportError:
            self.logger.warn("requests 패키지가 없습니다. Mock 결과를 반환합니다.")
            return self._mock_response(prompt)
        except requests.exceptions.ConnectionError:
            return ToolResult.error_result(
                f"Ollama 서버에 연결할 수 없습니다: {self.base_url}"
            )
        except requests.exceptions.Timeout:
            return ToolResult.error_result("Ollama API 타임아웃")
        except Exception as e:
            return ToolResult.error_result(f"Ollama API 오류: {str(e)}")
    
    def _parse_json_response(self, response: str) -> Dict:
        """LLM 응답에서 JSON 추출"""
        # JSON 블록 찾기
        import re
        
        # ```json ... ``` 패턴
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            return json.loads(json_match.group(1))
        
        # { ... } 패턴
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group(0))
        
        raise ValueError("JSON not found in response")
    
    # =========================================================================
    # Mock Implementations (테스트용)
    # =========================================================================
    
    def _mock_response(self, prompt: str) -> ToolResult:
        """Mock 응답"""
        self.logger.debug(f"Mock 응답 생성: {prompt[:50]}...")
        
        mock_output = f"""이것은 Mock 응답입니다.

질문: {prompt[:100]}...

Mock LLM 응답:
- 이 응답은 테스트 목적으로 생성되었습니다.
- 실제 LLM 연결 시 의미 있는 응답이 생성됩니다.
- Ollama 서버가 실행 중인지 확인하세요.
"""
        
        return ToolResult.success_result(
            output=mock_output,
            metadata={"model": self.model, "mock": True}
        )
    
    def _mock_analyze(self, content: str, instruction: str) -> ToolResult:
        """Mock 분석 응답"""
        content_preview = content[:100] + "..." if len(content) > 100 else content
        
        mock_output = f"""## 분석 결과 (Mock)

### 입력 내용 요약
{content_preview}

### 분석
- 이것은 Mock 분석 결과입니다.
- 지시사항: {instruction}
- 실제 LLM 연결 시 상세한 분석이 제공됩니다.

### 주요 발견사항
1. Mock 발견 1
2. Mock 발견 2
3. Mock 발견 3
"""
        
        return ToolResult.success_result(
            output=mock_output,
            metadata={"mock": True}
        )
    
    def _mock_summarize(self, content: str, max_length: int) -> ToolResult:
        """Mock 요약 응답"""
        # 간단히 앞부분 잘라서 반환
        if len(content) <= max_length:
            summary = content
        else:
            summary = content[:max_length - 20] + "... (Mock 요약)"
        
        return ToolResult.success_result(
            output=summary,
            metadata={"mock": True, "original_length": len(content)}
        )
    
    def _mock_extract(self, content: str, extract_type: str) -> ToolResult:
        """Mock 추출 응답"""
        mock_output = {
            "extract_type": extract_type,
            "mock": True,
            "items": [
                f"Mock {extract_type} 1",
                f"Mock {extract_type} 2",
                f"Mock {extract_type} 3"
            ],
            "count": 3,
            "source_length": len(content)
        }
        
        return ToolResult.success_result(
            output=mock_output,
            metadata={"mock": True}
        )
