# WonyBot 사용 설명서

## 소개
WonyBot은 OpenAI의 gpt-oss 모델을 기반으로 한 개인 AI 비서 봇입니다.
Ollama를 통해 로컬에서 실행되며, RAG(Retrieval-Augmented Generation) 시스템을 지원합니다.

## 주요 기능

### 1. 대화 기능
- **실시간 대화**: 스트리밍 응답으로 즉각적인 피드백
- **세션 관리**: 대화 히스토리를 SQLite에 저장
- **프롬프트 템플릿**: developer, creative, analyst 등 다양한 모드

### 2. RAG 시스템
WonyBot의 RAG 시스템은 문서 기반 질의응답을 가능하게 합니다.

#### 지원 문서 형식
- PDF 문서
- Word 문서 (DOCX)
- Markdown 파일
- 일반 텍스트 파일
- HTML 파일

#### 검색 방식
1. **벡터 검색**: 의미적 유사도 기반
2. **키워드 검색**: BM25 알고리즘 사용
3. **하이브리드 검색**: 벡터와 키워드 결합

### 3. 임베딩 모델
- **모델**: BAAI/bge-small-en-v1.5
- **차원**: 384차원
- **캐싱**: 중복 계산 방지를 위한 자동 캐싱

## 사용 방법

### 문서 인덱싱
```bash
# 단일 파일 인덱싱
wony index document.pdf

# 디렉토리 전체 인덱싱
wony index ./documents --recursive

# 특정 컬렉션에 인덱싱
wony index ./data --collection research
```

### 검색
```bash
# 하이브리드 검색
wony search "gpt-oss 모델의 특징"

# 벡터 검색만 사용
wony search "AI 비서" --type vector

# 상위 10개 결과
wony search "Ollama" --top 10
```

### 질문하기
```bash
# RAG를 사용한 질의응답
wony ask "WonyBot의 주요 기능은 무엇인가요?"

# 더 많은 문서 참조
wony ask "RAG 시스템은 어떻게 작동하나요?" --top 10
```

## 성능 최적화

### 1. GPU 가속
- Apple Silicon: MPS 자동 사용
- NVIDIA: CUDA 지원
- CPU 폴백 지원

### 2. 캐싱 전략
- 임베딩 캐시: SHA256 해시 기반
- 쿼리 캐시: LRU 캐시
- 배치 처리: 32개 단위

### 3. 메모리 관리
- 청킹: 512 토큰 단위
- 오버랩: 50 토큰
- 증분 인덱싱 지원

## 고급 기능

### 컬렉션 관리
- documents: 일반 문서
- chat_history: 대화 기록
- web_content: 웹 콘텐츠
- code: 소스 코드

### 리랭킹
- Cross-Encoder 기반 정밀 재순위
- MMR(Maximum Marginal Relevance)로 다양성 확보

## 문제 해결

### Ollama 연결 오류
```bash
ollama serve  # Ollama 서버 시작
ollama pull gpt-oss:20b  # 모델 다운로드
```

### 메모리 부족
- chunk_size를 256으로 줄이기
- 배치 크기를 16으로 조정

### 검색 정확도 개선
- 하이브리드 검색 사용
- top_k 값 증가
- 문서 품질 확인