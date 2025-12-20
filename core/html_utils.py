"""
HTML Utilities
==============
HTML 파싱, 요약, 미리보기 지원
"""

import re
import html
import hashlib
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class HTMLSummary:
    """HTML 요약 정보"""
    title: str
    text_preview: str
    html_hash: str  # 미리보기용 식별자
    char_count: int
    has_forms: bool
    has_scripts: bool


class HTMLParser:
    """HTML 파싱 및 요약 유틸리티"""
    
    @staticmethod
    def is_html(content: str) -> bool:
        """HTML 여부 판단"""
        content_lower = content.strip().lower()
        return (
            content_lower.startswith('<!doctype html') or
            content_lower.startswith('<html') or
            '<html' in content_lower[:500] or
            bool(re.search(r'<(head|body|div|p|span|a|table)\b', content_lower[:1000]))
        )
    
    @staticmethod
    def extract_title(html_content: str) -> str:
        """HTML에서 title 추출"""
        # <title> 태그
        match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            return html.unescape(title)
        
        # og:title 메타 태그
        match = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1))
        
        # 첫 번째 h1
        match = re.search(r'<h1[^>]*>(.*?)</h1>', html_content, re.IGNORECASE | re.DOTALL)
        if match:
            return HTMLParser.strip_tags(match.group(1)).strip()
        
        return "(제목 없음)"
    
    @staticmethod
    def extract_description(html_content: str) -> str:
        """HTML에서 description 추출"""
        # meta description
        match = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1))
        
        # og:description
        match = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1))
        
        return ""
    
    @staticmethod
    def strip_tags(html_content: str) -> str:
        """HTML 태그 제거하고 텍스트만 추출"""
        # script, style 태그 내용 제거
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # 모든 태그 제거
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # HTML 엔티티 디코딩
        text = html.unescape(text)
        
        # 연속 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_text_preview(html_content: str, max_length: int = 300) -> str:
        """HTML에서 텍스트 미리보기 추출"""
        # 먼저 description 시도
        desc = HTMLParser.extract_description(html_content)
        if desc and len(desc) > 50:
            return desc[:max_length] + "..." if len(desc) > max_length else desc
        
        # body 내용 추출
        body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.IGNORECASE | re.DOTALL)
        if body_match:
            text = HTMLParser.strip_tags(body_match.group(1))
        else:
            text = HTMLParser.strip_tags(html_content)
        
        # 앞부분 추출
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    @staticmethod
    def generate_hash(content: str) -> str:
        """콘텐츠 해시 생성 (미리보기 식별용)"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @staticmethod
    def has_dangerous_content(html_content: str) -> Tuple[bool, bool]:
        """위험 요소 확인 (scripts, forms)"""
        has_scripts = bool(re.search(r'<script\b', html_content, re.IGNORECASE))
        has_forms = bool(re.search(r'<form\b', html_content, re.IGNORECASE))
        return has_scripts, has_forms
    
    @staticmethod
    def summarize(html_content: str) -> HTMLSummary:
        """HTML 요약 정보 생성"""
        title = HTMLParser.extract_title(html_content)
        text_preview = HTMLParser.extract_text_preview(html_content)
        html_hash = HTMLParser.generate_hash(html_content)
        has_scripts, has_forms = HTMLParser.has_dangerous_content(html_content)
        
        return HTMLSummary(
            title=title,
            text_preview=text_preview,
            html_hash=html_hash,
            char_count=len(html_content),
            has_forms=has_forms,
            has_scripts=has_scripts
        )
    
    @staticmethod
    def sanitize_for_iframe(html_content: str) -> str:
        """iframe용 HTML 정리 (위험 요소 제거)"""
        # script 태그 제거
        sanitized = re.sub(r'<script\b[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        
        # on* 이벤트 핸들러 제거
        sanitized = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'\s+on\w+\s*=\s*\S+', '', sanitized, flags=re.IGNORECASE)
        
        # javascript: URL 제거
        sanitized = re.sub(r'href\s*=\s*["\']javascript:[^"\']*["\']', 'href="#"', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    def format_for_chat(html_content: str, html_hash: str) -> str:
        """채팅용 HTML 요약 포맷 생성"""
        summary = HTMLParser.summarize(html_content)
        
        warning = ""
        if summary.has_scripts:
            warning += "⚠️ 스크립트 포함 "
        if summary.has_forms:
            warning += "⚠️ 폼 포함 "
        
        size_kb = summary.char_count / 1024
        
        return f"""📄 **{summary.title}**

{summary.text_preview}

📊 크기: {size_kb:.1f}KB {warning}

`[HTML_PREVIEW:{html_hash}]`"""


# HTML 저장소 (메모리)
_html_storage: dict = {}


def store_html(html_content: str) -> str:
    """HTML 저장하고 해시 반환"""
    html_hash = HTMLParser.generate_hash(html_content)
    _html_storage[html_hash] = html_content
    return html_hash


def get_html(html_hash: str) -> Optional[str]:
    """저장된 HTML 가져오기"""
    return _html_storage.get(html_hash)


def clear_html_storage():
    """HTML 저장소 비우기"""
    _html_storage.clear()