# 🤖 GPT-OSS 기반 개인 비서 봇 아키텍처

## 📌 프로젝트 개요
OpenAI의 gpt-oss 모델을 활용한 개인 비서 봇 시스템. Ollama를 통해 로컬에서 실행되며, 대화 히스토리 관리와 다양한 태스크 수행을 지원합니다.

## 🎯 핵심 기능
- **대화 관리**: 컨텍스트 유지 및 대화 히스토리 저장
- **태스크 실행**: Function calling을 통한 다양한 작업 수행
- **프롬프트 템플릿**: 커스터마이징 가능한 시스템 프롬프트
- **멀티 세션**: 여러 대화 세션 관리

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                     사용자 인터페이스                      │
│                  CLI (Rich/Typer 기반)                   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Core Application Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Session    │  │   Command    │  │   Prompt     │ │
│  │   Manager    │  │   Handler    │  │   Manager    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                     Service Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │    Ollama    │  │   Database   │  │    Tools     │ │
│  │   Service    │  │   Service    │  │   Service    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                   Infrastructure Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Ollama API  │  │   Database   │  │   External   │ │
│  │  (gpt-oss)   │  │  (SQLite/    │  │     APIs     │ │
│  │              │  │  PostgreSQL) │  │              │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 💾 데이터베이스 옵션 비교

### Option 1: SQLite (로컬) ✅ 추천
**장점**:
- 설치/설정 불필요
- 제로 의존성
- 빠른 프로토타이핑
- 파일 기반 백업 용이

**단점**:
- 동시성 제한
- 확장성 제한

### Option 2: PostgreSQL (Docker)
**장점**:
- 프로덕션 레벨 성능
- 풍부한 기능 (JSON, 벡터 DB 확장)
- 동시성 지원

**단점**:
- Docker 필요
- 초기 설정 복잡

### Option 3: Supabase
**장점**:
- 완전 관리형 서비스
- 실시간 기능 내장
- Auth 기능 포함

**단점**:
- 인터넷 연결 필요
- 비용 발생 가능

## 📊 데이터 스키마

```sql
-- 대화 세션
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- 메시지 히스토리
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB -- function calls, tools used, etc.
);

-- 프롬프트 템플릿
CREATE TABLE prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    variables JSONB, -- 변수 정의
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자 설정
CREATE TABLE settings (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🛠️ 기술 스택

### Core
- **Language**: Python 3.11+
- **LLM Runtime**: Ollama (gpt-oss:20b)
- **CLI Framework**: Typer + Rich
- **Async**: asyncio + aiohttp

### Dependencies
```python
# requirements.txt
typer[all]>=0.9.0       # CLI framework
rich>=13.0.0            # Terminal UI
httpx>=0.25.0           # Async HTTP client
pydantic>=2.0.0         # Data validation
sqlalchemy>=2.0.0       # ORM
alembic>=1.13.0         # DB migrations
python-dotenv>=1.0.0    # Environment variables
```

### Optional (데이터베이스 선택에 따라)
```python
# SQLite (기본)
aiosqlite>=0.19.0

# PostgreSQL
asyncpg>=0.29.0
psycopg[binary]>=3.1.0

# Supabase
supabase>=2.3.0
```

## 📁 프로젝트 구조

```
wony-bot/
├── app/
│   ├── __init__.py
│   ├── main.py              # CLI 진입점
│   ├── config.py            # 설정 관리
│   ├── core/
│   │   ├── __init__.py
│   │   ├── session.py       # 세션 관리
│   │   ├── ollama.py        # Ollama 클라이언트
│   │   └── database.py      # DB 연결
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat.py          # 대화 서비스
│   │   ├── prompt.py        # 프롬프트 관리
│   │   └── tools.py         # Function calling
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic 모델
│   └── utils/
│       ├── __init__.py
│       └── helpers.py       # 유틸리티 함수
├── prompts/
│   └── system.txt           # 기본 시스템 프롬프트
├── .env.example             # 환경변수 예제
├── requirements.txt         # 의존성
├── README.md               # 프로젝트 문서
└── ARCHITECTURE.md         # 이 문서
```

## ⚡ 빠른 구현 전략

### Phase 1: MVP (1-2일)
1. ✅ Ollama 연동 기본 대화 기능
2. ✅ SQLite 기반 히스토리 저장
3. ✅ 간단한 CLI 인터페이스

### Phase 2: 핵심 기능 (3-4일)
1. 세션 관리 시스템
2. 프롬프트 템플릿 관리
3. Function calling 기본 도구

### Phase 3: 고급 기능 (5-7일)
1. 웹 검색 통합
2. 파일 처리 기능
3. 플러그인 시스템

## 🔧 Ollama 설정

```bash
# gpt-oss 모델 다운로드 (이미 진행 중)
ollama pull gpt-oss:20b

# 모델 실행 테스트
ollama run gpt-oss:20b "Hello, assistant!"

# API 서버 확인
curl http://localhost:11434/api/generate -d '{
  "model": "gpt-oss:20b",
  "prompt": "Hello"
}'
```

## 🚀 실행 명령어 예시

```bash
# 새 대화 시작
wony chat

# 특정 세션 계속
wony chat --session <session-id>

# 프롬프트 템플릿 사용
wony chat --prompt developer

# 히스토리 조회
wony history

# 설정 변경
wony config set model gpt-oss:120b
```

## 📈 성능 고려사항

### gpt-oss:20b 모델 특성
- **메모리 요구**: 최소 16GB RAM
- **응답 속도**: GPU 없을 시 느림 (10분/600단어)
- **토큰 제한**: 컨텍스트 윈도우 고려한 히스토리 관리

### 최적화 전략
1. **스트리밍 응답**: 실시간 출력
2. **히스토리 요약**: 긴 대화 자동 요약
3. **캐싱**: 자주 사용하는 프롬프트 캐싱
4. **비동기 처리**: 모든 I/O 작업 비동기화

## 🔐 보안 고려사항
- API 키 환경변수 관리
- 로컬 실행으로 데이터 프라이버시 보장
- 선택적 암호화 저장