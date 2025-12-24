"""
WebTool Module
==============
웹 검색, 페이지 가져오기 기능을 제공하는 Tool.

Note: 실제 구현에서는 외부 API(Google Search API, DuckDuckGo 등)를 사용해야 합니다.
현재는 테스트용 Mock 구현입니다.
"""

import re
from typing import Dict, List
from urllib.parse import urlparse

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_tool import BaseTool, ActionSchema, ActionParam, ToolResult


class WebTool(BaseTool):
    """웹 검색 및 페이지 가져오기 Tool"""
    
    name = "web_tool"
    description = "웹 검색, 페이지 내용 가져오기"
    
    def __init__(self, debug_enabled: bool = True, use_mock: bool = True):
        """
        Args:
            debug_enabled: 디버그 로깅 활성화
            use_mock: True면 Mock 데이터 사용, False면 실제 웹 요청
        """
        self.use_mock = use_mock
        super().__init__(debug_enabled=debug_enabled)
        
        if use_mock:
            self.logger.warn("Mock 모드로 실행됩니다. 실제 웹 요청이 발생하지 않습니다.")
    
    def _define_actions(self) -> Dict[str, ActionSchema]:
        return {
            "search": ActionSchema(
                name="search",
                description="키워드로 웹 검색",
                params=[
                    ActionParam("search_keyword", "str", True, "검색 키워드"),
                    ActionParam("search_max_results", "int", False, "최대 결과 수", 5)
                ],
                output_type="list",
                output_description="검색 결과 목록 [{title, url, snippet}, ...]"
            ),
            "fetch": ActionSchema(
                name="fetch",
                description="URL에서 페이지 내용 가져오기",
                params=[
                    ActionParam("fetch_url", "str", True, "가져올 URL"),
                    ActionParam("fetch_timeout", "int", False, "타임아웃 (초)", 10)
                ],
                output_type="str",
                output_description="페이지 텍스트 내용"
            )
        }
    
    def _execute_action(self, action: str, params: Dict) -> ToolResult:
        """실제 action 실행"""
        # prefix 파라미터 추출 헬퍼 (예: search_keyword -> keyword)
        def get_param(name: str, default=None):
            prefixed = f"{action}_{name}"
            return params.get(prefixed, default)
        
        if action == "search":
            return self._search(
                get_param("keyword"),
                get_param("max_results", 5)
            )
        elif action == "fetch":
            return self._fetch(
                get_param("url"),
                get_param("timeout", 10)
            )
        else:
            return ToolResult.error_result(f"Unknown action: {action}")
    
    # =========================================================================
    # Private Action Implementations
    # =========================================================================
    
    def _search(self, keyword: str, max_results: int) -> ToolResult:
        """웹 검색"""
        if self.use_mock:
            return self._mock_search(keyword, max_results)
        
        return self._real_search(keyword, max_results)
    
    def _fetch(self, url: str, timeout: int) -> ToolResult:
        """페이지 내용 가져오기"""
        # URL 유효성 검사
        if not self._is_valid_url(url):
            return ToolResult.error_result(f"Invalid URL: {url}")
        
        if self.use_mock:
            return self._mock_fetch(url)
        
        return self._real_fetch(url, timeout)
    
    def _is_valid_url(self, url: str) -> bool:
        """URL 유효성 검사"""
        try:
            result = urlparse(url)
            return all([result.scheme in ('http', 'https'), result.netloc])
        except:
            return False
    
    # =========================================================================
    # Mock Implementations (테스트용)
    # =========================================================================
    
    def _mock_search(self, keyword: str, max_results: int) -> ToolResult:
        """Mock 검색 결과"""
        self.logger.debug(f"Mock 검색: {keyword}")
        
        # 키워드 기반 가상 검색 결과 생성
        mock_results = [
            {
                "title": f"{keyword} - 완벽 가이드",
                "url": f"https://example.com/guide/{keyword.replace(' ', '-')}",
                "snippet": f"{keyword}에 대한 종합적인 가이드입니다. 기초부터 심화까지 다룹니다."
            },
            {
                "title": f"{keyword} 문제 해결 방법",
                "url": f"https://example.com/troubleshoot/{keyword.replace(' ', '-')}",
                "snippet": f"{keyword} 관련 일반적인 문제와 해결 방법을 설명합니다."
            },
            {
                "title": f"{keyword} - Stack Overflow",
                "url": f"https://stackoverflow.com/questions/{keyword.replace(' ', '-')}",
                "snippet": f"개발자 커뮤니티에서 논의된 {keyword} 관련 질문과 답변입니다."
            },
            {
                "title": f"{keyword} 공식 문서",
                "url": f"https://docs.example.com/{keyword.replace(' ', '-')}",
                "snippet": f"{keyword}의 공식 문서입니다. API 레퍼런스와 예제를 포함합니다."
            },
            {
                "title": f"{keyword} 튜토리얼 - YouTube",
                "url": f"https://youtube.com/watch?v={keyword[:8]}",
                "snippet": f"{keyword}를 배우는 영상 튜토리얼입니다."
            }
        ]
        
        results = mock_results[:max_results]
        
        # 명확한 결과 메시지 생성
        result_summary = f"✅ Found {len(results)} results for '{keyword}':\n"
        for i, r in enumerate(results, 1):
            result_summary += f"\n{i}. {r['title']}\n   {r['url']}\n   {r['snippet'][:80]}..."
        
        return ToolResult.success_result(
            output=result_summary,
            metadata={
                "keyword": keyword,
                "result_count": len(results),
                "results": results,  # 상세 데이터는 metadata에
                "mock": True
            }
        )
    
    def _mock_fetch(self, url: str) -> ToolResult:
        """Mock 페이지 내용 - HTML 형태로 반환"""
        self.logger.debug(f"Mock fetch: {url}")
        
        # URL 기반 가상 컨텐츠 생성
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        
        mock_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="description" content="{domain}의 {path} 페이지입니다. 테스트용 Mock 컨텐츠입니다.">
    <title>{domain} - Mock Page</title>
</head>
<body>
    <header>
        <h1>{domain} 페이지</h1>
        <nav>
            <a href="/">홈</a>
            <a href="/about">소개</a>
            <a href="/contact">연락처</a>
        </nav>
    </header>
    
    <main>
        <article>
            <h2>개요</h2>
            <p>이것은 {domain}의 {path} 페이지입니다.</p>
            
            <h2>내용</h2>
            <p>이 페이지는 테스트용 Mock 컨텐츠입니다.</p>
            <p>실제 구현에서는 requests 라이브러리를 사용하여 실제 웹 페이지를 가져옵니다.</p>
            
            <h2>주요 포인트</h2>
            <ul>
                <li>포인트 1: Mock 데이터입니다</li>
                <li>포인트 2: 실제 데이터가 아닙니다</li>
                <li>포인트 3: 테스트 목적으로 사용됩니다</li>
            </ul>
        </article>
    </main>
    
    <footer>
        <p>© 2024 {domain}. All rights reserved.</p>
    </footer>
</body>
</html>"""
        
        return ToolResult.success_result(
            output=mock_html,
            metadata={
                "url": url,
                "title": f"{domain} - Mock Page",
                "domain": domain,
                "content_length": len(mock_html),
                "is_html": True,
                "mock": True
            }
        )
    
    # =========================================================================
    # Real Implementations (실제 웹 요청)
    # =========================================================================
    
    def _real_search(self, keyword: str, max_results: int) -> ToolResult:
        """실제 웹 검색 (DuckDuckGo 사용)"""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(keyword, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
            
            # 명확한 결과 메시지 생성
            result_summary = f"✅ Found {len(results)} results for '{keyword}':\n"
            for i, r in enumerate(results, 1):
                snippet = r['snippet'][:80] + "..." if len(r['snippet']) > 80 else r['snippet']
                result_summary += f"\n{i}. {r['title']}\n   {r['url']}\n   {snippet}"
            
            return ToolResult.success_result(
                output=result_summary,
                metadata={
                    "keyword": keyword,
                    "result_count": len(results),
                    "results": results,  # 상세 데이터는 metadata에
                    "mock": False
                }
            )
        except ImportError:
            self.logger.warn("duckduckgo_search 패키지가 없습니다. Mock 결과를 반환합니다.")
            return self._mock_search(keyword, max_results)
        except Exception as e:
            return ToolResult.error_result(f"Search error: {str(e)}")
    
    def _real_fetch(self, url: str, timeout: int) -> ToolResult:
        """실제 페이지 가져오기 - HTML 원본 + 텍스트 요약"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            html_content = response.text
            
            # HTML 파싱하여 텍스트 추출 (요약용)
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Title 추출
            title = soup.title.string if soup.title else "(제목 없음)"
            
            # 스크립트, 스타일 제거 (텍스트 추출용)
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 텍스트 추출
            text = soup.get_text(separator="\n", strip=True)
            
            # 연속된 빈 줄 제거
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return ToolResult.success_result(
                output=html_content,  # HTML 원본 반환
                metadata={
                    "url": url,
                    "title": title,
                    "status_code": response.status_code,
                    "content_length": len(html_content),
                    "text_preview": text[:500] + "..." if len(text) > 500 else text,
                    "is_html": True,
                    "mock": False
                }
            )
        except ImportError:
            self.logger.warn("requests/beautifulsoup4 패키지가 없습니다. Mock 결과를 반환합니다.")
            return self._mock_fetch(url)
        except Exception as e:
            return ToolResult.error_result(f"Fetch error: {str(e)}")
