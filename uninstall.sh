#!/bin/bash

# WonyBot Uninstall Script

set -e

echo "🗑️  WonyBot 제거 시작..."

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 설치 위치 확인
if [ -f ~/.wonybot_location ]; then
    WONY_DIR=$(cat ~/.wonybot_location)
    echo "설치 위치: $WONY_DIR"
else
    WONY_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

# 제거 확인
echo -e "${YELLOW}⚠️  WonyBot을 제거하시겠습니까?${NC}"
echo "다음 항목들이 제거됩니다:"
echo "  - /usr/local/bin/wony (심볼릭 링크)"
echo "  - pip 패키지 (설치된 경우)"
echo ""
read -p "계속하시겠습니까? [y/N]: " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "제거가 취소되었습니다."
    exit 0
fi

# 심볼릭 링크 제거
if [ -L "/usr/local/bin/wony" ]; then
    echo "🔗 심볼릭 링크 제거 중..."
    if [ -w "/usr/local/bin" ]; then
        rm -f /usr/local/bin/wony
    else
        sudo rm -f /usr/local/bin/wony
    fi
    echo -e "${GREEN}✅ 심볼릭 링크 제거됨${NC}"
fi

# wony-exec 파일 제거
if [ -f "$WONY_DIR/wony-exec" ]; then
    rm -f "$WONY_DIR/wony-exec"
fi

# pip 패키지 제거 확인
if command -v pip &> /dev/null; then
    if pip show wonybot &> /dev/null; then
        echo "📦 pip 패키지 제거 중..."
        pip uninstall -y wonybot
        echo -e "${GREEN}✅ pip 패키지 제거됨${NC}"
    fi
fi

# 설치 위치 파일 제거
rm -f ~/.wonybot_location

# 데이터 파일 제거 옵션
echo ""
echo -e "${YELLOW}데이터 파일도 제거하시겠습니까?${NC}"
echo "다음 파일들이 제거됩니다:"
echo "  - $WONY_DIR/wony_bot.db (대화 기록)"
echo "  - $WONY_DIR/data/ (ChromaDB 인덱스)"
echo "  - $WONY_DIR/.env (설정 파일)"
read -p "데이터도 제거하시겠습니까? [y/N]: " remove_data

if [ "$remove_data" = "y" ] || [ "$remove_data" = "Y" ]; then
    echo "🗃️  데이터 파일 제거 중..."
    rm -f "$WONY_DIR/wony_bot.db"
    rm -rf "$WONY_DIR/data/"
    rm -f "$WONY_DIR/.env"
    echo -e "${GREEN}✅ 데이터 파일 제거됨${NC}"
else
    echo "데이터 파일은 보존됩니다."
fi

# 가상환경 제거 옵션
echo ""
echo -e "${YELLOW}가상환경도 제거하시겠습니까?${NC}"
read -p "venv 디렉토리를 제거하시겠습니까? [y/N]: " remove_venv

if [ "$remove_venv" = "y" ] || [ "$remove_venv" = "Y" ]; then
    echo "🗑️  가상환경 제거 중..."
    rm -rf "$WONY_DIR/venv"
    echo -e "${GREEN}✅ 가상환경 제거됨${NC}"
fi

echo ""
echo -e "${GREEN}✅ WonyBot 제거 완료!${NC}"
echo ""
echo "프로젝트 파일들은 그대로 남아있습니다: $WONY_DIR"
echo "전체 삭제를 원하시면 다음 명령을 실행하세요:"
echo "  rm -rf $WONY_DIR"