# 🤖 WonyBot - GPT-OSS Personal Assistant with RAG

GPT-OSS 모델 기반의 개인 비서 봇입니다. Ollama를 통해 로컬에서 실행되며, **RAG (Retrieval-Augmented Generation)** 시스템을 통한 문서 기반 질의응답을 지원합니다.

## ✨ 주요 기능

### 대화 시스템
- 🚀 **gpt-oss:20b** 모델 활용 (Ollama)
- 💬 **대화 세션 관리** - 히스토리 저장 및 컨텍스트 유지
- 📝 **프롬프트 템플릿** - 다양한 용도의 시스템 프롬프트
- 🔄 **스트리밍 응답** - 실시간 출력
- 💾 **SQLite 데이터베이스** - 제로 설정, 로컬 저장
- 🎨 **Rich CLI** - 아름다운 터미널 인터페이스

### 🔍 RAG 시스템 (새 기능!)
- 📚 **문서 인덱싱** - PDF, DOCX, Markdown, TXT, HTML 지원
- 🧠 **하이브리드 검색** - 벡터 + 키워드 (BM25) 결합
- 💡 **스마트 청킹** - 의미 단위로 문서 분할
- ⚡ **고성능 임베딩** - Sentence-Transformers 활용
- 🗄️ **ChromaDB** - 벡터 데이터베이스
- 🔄 **리랭킹** - Cross-Encoder 기반 정밀도 향상
- 💾 **캐싱** - 중복 계산 방지로 성능 최적화

### 🧠 대화 메모리 시스템 (새 기능!)
- 🔍 **자동 중요도 분석** - 키워드 및 패턴 기반 중요도 판단
- 💾 **자동 저장** - 중요한 대화 내용 자동으로 RAG에 저장
- 🏷️ **태그 시스템** - task, definition, instruction 등 자동 분류
- 📊 **세션 요약** - 대화 세션 자동 요약 및 저장
- 🔎 **메모리 검색** - 저장된 중요 정보 검색 및 조회

## 📋 요구사항

- Python 3.11 이상
- Ollama 설치 및 실행 중
- 16GB 이상의 RAM (gpt-oss:20b 실행용)

## 🚀 빠른 시작

### 1. Ollama 설치
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Ollama 실행
ollama serve
```

### 2. 프로젝트 설정

#### 방법 1: 자동 설치 (권장) 🚀
```bash
# 저장소 클론
git clone https://github.com/leejaebae/wony-bot.git
cd wony-bot

# 자동 설치 스크립트 실행
./install.sh

# 설치 옵션:
# 1) 심볼릭 링크 (개발 모드, 권장)
# 2) pip install -e . (편집 가능)
# 3) pip install . (시스템 전역)
```

#### 방법 2: 수동 설치
```bash
# 저장소 클론
cd wony-bot

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정 (선택사항)
cp .env.example .env

# 전역 설치 (선택사항)
pip install -e .
```

### 3. 초기 설정 실행
```bash
# 모델 다운로드 및 데이터베이스 초기화
wony setup  # 전역 설치한 경우
# 또는
python -m app.main setup  # 수동 설치한 경우
```

### 4. 대화 시작
```bash
# 새 대화 시작
wony chat

# 특정 프롬프트 사용
wony chat --prompt developer

# 이전 세션 계속
wony chat --session <session-id>
```

## 📖 사용법

### 기본 명령어

```bash
# 대화 시작
wony chat

# 세션 히스토리 보기
wony history

# 프롬프트 관리
wony prompts list
wony prompts show default
wony prompts create my-prompt
wony prompts delete my-prompt

# 설정 확인
wony config show
```

### 🔍 RAG 명령어

```bash
# 문서 인덱싱
wony index ./documents              # 디렉토리 인덱싱
wony index document.pdf             # 단일 파일 인덱싱
wony index ./data --collection research  # 특정 컬렉션

# 문서 검색
wony search "검색어"                # 하이브리드 검색
wony search "검색어" --type vector  # 벡터 검색만
wony search "검색어" --top 10       # 상위 10개 결과

# 질의응답
wony ask "질문 내용"                # RAG 기반 답변
wony ask "질문" --top 10            # 더 많은 문서 참조

# RAG 통계
wony rag-stats                      # 인덱스 통계 확인
wony clear-index                    # 인덱스 초기화
```

### 🧠 메모리 명령어

```bash
# 저장된 메모리 조회
wony memories                       # 최근 메모리 표시
wony memories --importance HIGH     # 중요도별 필터링
wony memories --tags task,personal  # 태그별 필터링
wony memories --limit 20            # 더 많은 메모리 표시

# 메모리 통계
wony memory-stats                   # 메모리 시스템 통계

# 세션 요약
wony summarize-session <session-id> # 특정 세션 요약
wony summarize-session <id> --no-save # 저장 없이 요약만
```

### 대화 중 명령어

- `exit` / `quit` - 대화 종료
- `clear` - 현재 세션 히스토리 삭제
- `new` - 새 세션 시작

### 프롬프트 템플릿

사전 정의된 프롬프트:
- `default` - 기본 어시스턴트
- `developer` - 프로그래밍 도우미
- `creative` - 창의적 글쓰기
- `analyst` - 데이터 분석
- `tutor` - 교육 튜터

## 🔧 고급 설정

### 환경 변수 (.env)

```bash
# Ollama 설정
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b

# 데이터베이스
DATABASE_URL=sqlite+aiosqlite:///./wony_bot.db

# 앱 설정
DEBUG=false
LOG_LEVEL=INFO
```

### 다른 모델 사용

```bash
# gpt-oss:120b 사용 (80GB GPU 필요)
OLLAMA_MODEL=gpt-oss:120b python -m app.main chat
```

### Docker로 PostgreSQL 사용

```bash
# PostgreSQL 실행
docker run -d \
  --name wonybot-db \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=wonybot \
  -p 5432:5432 \
  postgres:15

# .env 수정
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost/wonybot
```

## 🛠️ 설치 및 제거

### 전역 설치
```bash
# 설치
./install.sh

# 제거
./uninstall.sh
```

### 설치 옵션
1. **심볼릭 링크** (권장): 개발 모드, 코드 변경사항 즉시 반영
2. **pip install -e**: 편집 가능 설치, Python 환경에 통합
3. **pip install**: 시스템 전역 설치, 안정적 사용

## 🏗️ 프로젝트 구조

```
wony-bot/
├── app/
│   ├── main.py               # CLI 진입점
│   ├── config.py             # 설정 관리
│   ├── core/                 # 핵심 모듈
│   │   ├── ollama.py         # Ollama 클라이언트
│   │   ├── database.py       # DB 연결
│   │   └── session.py        # 세션 관리
│   ├── services/             # 서비스 레이어
│   │   ├── chat.py           # 대화 서비스
│   │   ├── prompt.py         # 프롬프트 관리
│   │   └── memory_manager.py # 메모리 관리
│   ├── rag/                  # RAG 시스템
│   │   ├── __init__.py       # RAG 초기화
│   │   ├── vector_store.py   # ChromaDB 관리
│   │   ├── embeddings.py     # 임베딩 생성
│   │   ├── document_loader.py# 문서 로더
│   │   ├── retriever.py      # 검색 엔진
│   │   └── rag_chain.py      # RAG 파이프라인
│   └── models/               # 데이터 모델
│       └── schemas.py        # Pydantic 스키마
├── prompts/                  # 프롬프트 템플릿
├── requirements.txt          # 의존성
├── setup.py                  # 패키지 설정
├── install.sh                # 설치 스크립트
├── uninstall.sh              # 제거 스크립트
├── ARCHITECTURE.md           # 시스템 설계
├── RAG_ARCHITECTURE.md       # RAG 설계
└── README.md                # 이 문서
```

## 📊 성능 고려사항

### 메모리 사용량
- **gpt-oss:20b**: 최소 16GB RAM
- **gpt-oss:120b**: 80GB GPU 메모리

### 응답 속도
- GPU 가속 시: ~256 토큰/초 (RTX 5090)
- CPU only: 매우 느림 (10분/600단어)

### 최적화 팁
- 스트리밍 모드 사용 (기본값)
- 히스토리 길이 제한 (기본 10개 메시지)
- SQLite 사용 (로컬 실행)

## 🐛 문제 해결

### Ollama 연결 오류
```bash
# Ollama 상태 확인
curl http://localhost:11434/api/tags

# Ollama 재시작
killall ollama
ollama serve
```

### 모델 다운로드 실패
```bash
# 수동으로 모델 다운로드
ollama pull gpt-oss:20b
```

### 데이터베이스 초기화
```bash
# DB 파일 삭제 후 재생성
rm wony_bot.db
python -m app.main setup
```

## 📝 라이선스

MIT License

## 🔒 보안 기능

- **Path Traversal 방지**: 파일 접근 시 안전한 디렉토리만 허용
- **메모리 오버플로우 방지**: 자동 버퍼 크기 관리
- **SQL Injection 방지**: SQLAlchemy ORM 사용
- **민감 정보 보호**: 로그에서 자동 마스킹

## 🧪 테스트

```bash
# 테스트 실행
pytest

# 커버리지 확인
pytest --cov=app --cov-report=html

# 특정 테스트만 실행
pytest tests/test_memory_manager.py
pytest -m security  # 보안 테스트만
pytest -m unit      # 단위 테스트만

# 코드 품질 검사
black app tests     # 코드 포맷팅
flake8 app tests    # 린팅
mypy app            # 타입 체크
```

## 🔧 개발 도구

### 의존성 주입
- 싱글톤 패턴 적용
- ServiceContainer를 통한 중앙화된 서비스 관리

### 구조화된 로깅
- JSON 형식 로깅 지원
- 민감 정보 자동 마스킹
- 실행 시간 측정 데코레이터

### 에러 처리
- 커스텀 예외 클래스
- 상세한 에러 컨텍스트
- 복구 전략 구현

## 🤝 기여하기

이슈와 PR을 환영합니다!

### 개발 환경 설정
```bash
# 개발 의존성 설치
pip install -r requirements.txt

# pre-commit 훅 설정
pip install pre-commit
pre-commit install
```

### 코드 스타일
- Black으로 자동 포맷팅
- Flake8 린팅 규칙 준수
- Type hints 사용 권장

## 🔗 관련 링크

- [Ollama](https://ollama.ai)
- [GPT-OSS](https://github.com/openai/gpt-oss)
- [OpenAI Cookbook](https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama)