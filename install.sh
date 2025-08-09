#!/bin/bash

# WonyBot Global Installation Script

set -e

echo "🤖 WonyBot 전역 설치 시작..."

# 현재 디렉토리 저장
WONY_DIR="$(cd "$(dirname "$0")" && pwd)"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Python 버전 확인
echo "📋 Python 버전 확인 중..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    REQUIRED_VERSION="3.11"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        echo -e "${GREEN}✅ Python $PYTHON_VERSION 확인됨${NC}"
    else
        echo -e "${RED}❌ Python 3.11 이상이 필요합니다. 현재 버전: $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Python3가 설치되어 있지 않습니다${NC}"
    exit 1
fi

# 가상환경 확인 및 생성
if [ ! -d "$WONY_DIR/venv" ]; then
    echo "🔧 가상환경 생성 중..."
    python3 -m venv "$WONY_DIR/venv"
fi

# 가상환경 활성화 및 의존성 설치
echo "📦 의존성 설치 중..."
source "$WONY_DIR/venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$WONY_DIR/requirements.txt"

# 설치 방법 선택
echo ""
echo "설치 방법을 선택하세요:"
echo "1) 심볼릭 링크 생성 (개발 모드, 권장)"
echo "2) pip install -e . (편집 가능 설치)"
echo "3) 시스템 전역 설치 (pip install .)"
read -p "선택 [1-3]: " choice

case $choice in
    1)
        # 심볼릭 링크 방식
        echo "🔗 심볼릭 링크 생성 중..."
        
        # wony 실행 스크립트 생성
        cat > "$WONY_DIR/wony-exec" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
python -m app.main "$@"
EOF
        chmod +x "$WONY_DIR/wony-exec"
        
        # 심볼릭 링크 생성
        if [ -w "/usr/local/bin" ]; then
            ln -sf "$WONY_DIR/wony-exec" /usr/local/bin/wony
            echo -e "${GREEN}✅ /usr/local/bin/wony 에 설치됨${NC}"
        else
            echo -e "${YELLOW}⚠️  /usr/local/bin에 쓰기 권한이 없습니다. sudo로 다시 시도합니다...${NC}"
            sudo ln -sf "$WONY_DIR/wony-exec" /usr/local/bin/wony
            echo -e "${GREEN}✅ /usr/local/bin/wony 에 설치됨${NC}"
        fi
        ;;
        
    2)
        # pip 편집 가능 설치
        echo "📦 pip 편집 가능 설치 중..."
        pip install -e "$WONY_DIR"
        echo -e "${GREEN}✅ pip install -e 완료${NC}"
        ;;
        
    3)
        # 시스템 전역 설치
        echo "📦 시스템 전역 설치 중..."
        pip install "$WONY_DIR"
        echo -e "${GREEN}✅ 시스템 전역 설치 완료${NC}"
        ;;
        
    *)
        echo -e "${RED}❌ 잘못된 선택입니다${NC}"
        exit 1
        ;;
esac

# .env 파일 확인
if [ ! -f "$WONY_DIR/.env" ]; then
    echo "⚙️  .env 파일 생성 중..."
    cp "$WONY_DIR/.env.example" "$WONY_DIR/.env"
    echo -e "${YELLOW}📝 .env 파일이 생성되었습니다. 필요시 수정하세요: $WONY_DIR/.env${NC}"
fi

# Ollama 확인
echo ""
echo "🔍 Ollama 상태 확인 중..."
if command -v ollama &> /dev/null; then
    if pgrep -x "ollama" > /dev/null; then
        echo -e "${GREEN}✅ Ollama가 실행 중입니다${NC}"
    else
        echo -e "${YELLOW}⚠️  Ollama가 설치되어 있지만 실행 중이 아닙니다${NC}"
        echo "   실행: ollama serve"
    fi
else
    echo -e "${YELLOW}⚠️  Ollama가 설치되어 있지 않습니다${NC}"
    echo "   설치: https://ollama.ai"
fi

# 설치 완료
echo ""
echo -e "${GREEN}🎉 WonyBot 설치 완료!${NC}"
echo ""
echo "사용법:"
echo "  wony chat                 # 대화 시작"
echo "  wony index <path>         # 문서 인덱싱"
echo "  wony ask <question>       # RAG 질문"
echo "  wony memories            # 저장된 메모리 조회"
echo "  wony --help              # 도움말"
echo ""
echo "첫 실행:"
echo "  wony setup               # 초기 설정 및 모델 다운로드"
echo ""

# 설치 위치 정보 저장 (나중에 언인스톨용)
echo "$WONY_DIR" > ~/.wonybot_location

deactivate 2>/dev/null || true