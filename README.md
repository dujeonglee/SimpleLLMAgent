# Multi-Agent Chatbot

LLM 기반 멀티 에이전트 챗봇 시스템입니다. Ollama를 백엔드로 사용하며, 파일 작업 및 LLM 분석 도구를 제공합니다.

## 주요 기능

- **Plan & Execute 패턴**: 사용자 요청을 자동으로 분석하여 실행 계획을 수립하고 단계별로 실행
- **도구 시스템**: 파일 읽기/쓰기/삭제 및 LLM 기반 코드 분석 도구 제공
- **실시간 스트리밍**: 각 단계의 실행 과정을 실시간으로 확인
- **워크스페이스 관리**: 파일 업로드 및 작업 공간 관리
- **LLM 설정 관리**: 온도, top-p, top-k, 페널티 등 다양한 LLM 파라미터 조정 가능

## 시스템 요구사항

- Python 3.8 이상
- Ollama 서버 (로컬 또는 원격)

## 설치 방법

1. **저장소 클론**
   ```bash
   git clone <repository-url>
   cd SimpleLLMAgent
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ollama 설치 및 실행**
   - [Ollama 공식 사이트](https://ollama.ai/)에서 다운로드 및 설치
   - 모델 다운로드:
     ```bash
     ollama pull llama3.2
     ```
   - Ollama 서버 실행 (기본적으로 자동 실행됨)

## 실행 방법

```bash
python app.py
```

브라우저에서 http://localhost:7860 접속

## 사용 방법

### 1. 설정 구성

- 우측 상단의 **⚙️ Settings** 버튼 클릭
- Ollama URL 설정 (기본값: http://localhost:11434)
- 모델 선택 및 LLM 파라미터 조정
- 설정은 자동으로 `workspace/config/llm_config.json`에 저장됨

### 2. 파일 업로드

- **📁 Workspace Files** 탭에서 파일 업로드
- 업로드된 파일은 `workspace/files/` 디렉토리에 저장됨

### 3. 챗봇 사용 예시

```
예시 1: 파일 읽기 및 분석
"sample.c 파일을 읽어서 정적 분석하고 out.md에 저장해줘"

예시 2: 코드 리팩토링
"main.py를 읽어서 코드를 개선하고 main_refactored.py로 저장해줘"

예시 3: 문서 생성
"config.json을 분석해서 README.md 형식의 문서를 만들어줘"
```

## 프로젝트 구조

```
SimpleLLMAgent/
├── app.py                      # Gradio UI 메인 애플리케이션
├── core/
│   ├── base_tool.py           # 도구 기본 클래스 및 레지스트리
│   ├── orchestrator.py        # Plan & Execute 오케스트레이터
│   ├── shared_storage.py      # 세션 및 결과 저장소
│   ├── workspace_manager.py   # 워크스페이스 및 설정 관리
│   ├── execution_events.py    # 실행 이벤트 정의
│   ├── debug_logger.py        # 디버그 로거
│   └── json_parser.py         # JSON 파싱 유틸리티
├── tools/
│   ├── file_tool.py           # 파일 작업 도구
│   └── llm_tool.py            # LLM 기반 분석 도구
├── workspace/
│   ├── config/                # 설정 파일
│   │   └── llm_config.json   # LLM 설정
│   ├── files/                 # 작업 공간 (도구가 접근)
│   └── uploads/               # 업로드 원본 보관
├── requirements.txt           # Python 의존성
└── README.md                  # 프로젝트 문서
```

## LLM 설정 파라미터

| 파라미터 | 설명 | 범위 | 기본값 |
|---------|------|------|--------|
| Temperature | 응답의 창의성/무작위성 조절 | 0.0-2.0 | 0.7 |
| Max Tokens | 생성할 최대 토큰 수 | 256-4096 | 2048 |
| Context Window | 처리할 최대 컨텍스트 길이 | 2048-32768 | 4096 |
| Top-p | 누적 확률 샘플링 | 0.0-1.0 | 0.9 |
| Top-k | 고려할 상위 토큰 개수 | 1-100 | 40 |
| Repeat Penalty | 반복 방지 페널티 | 1.0-2.0 | 1.1 |
| Frequency Penalty | 빈도 기반 페널티 | 0.0-2.0 | 0.0 |
| Presence Penalty | 존재 기반 페널티 | 0.0-2.0 | 0.0 |
| Max Steps | 최대 도구 호출 횟수 | 10-100 | 10 |
| Timeout | LLM API 타임아웃 (초) | 10-3600 | 120 |

## 제공 도구

### FileTool
- **read**: 파일 읽기
- **write**: 파일 쓰기
- **delete**: 파일 삭제
- **list**: 파일 목록 조회

### LLMTool
- **staticanalysis**: C 코드 정적 분석
- **refactor**: 코드 리팩토링
- **analyze**: 일반 파일 분석
- **review**: 코드 리뷰
- **generate**: 코드 생성
- **modify**: 파일 수정

## 로깅 및 디버깅

- 실행 로그는 콘솔에 실시간으로 출력됩니다
- LLM 성능 메트릭 (tokens/sec, TTFT 등)이 자동으로 기록됩니다
- 각 단계별 프롬프트 및 응답을 UI에서 확인 가능합니다

## 라이센스

MIT License

## 기여

이슈 및 PR은 언제든지 환영합니다!