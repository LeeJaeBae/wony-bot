# WonyBot 글로벌 설치 가이드 🌍

## 방법 1: 시스템 전체 설치 (추천) ⭐

```bash
# 1. 설치 스크립트 실행
./install_system.sh

# 2. 비밀번호 입력 (sudo 권한 필요)
# /usr/local/bin에 심볼릭 링크 생성

# 3. 어디서나 실행 가능!
wony chat
```

## 방법 2: PATH에 추가 📁

```bash
# 1. .zshrc 또는 .bash_profile에 추가
echo 'export PATH="$HOME/wony-bot:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 2. wony 명령어를 전역 alias로 설정
echo 'alias wony="~/wony-bot/wony_launcher.sh"' >> ~/.zshrc
source ~/.zshrc

# 3. 실행
wony chat
```

## 방법 3: pipx 사용 (Python 패키지 매니저) 🐍

```bash
# 1. pipx 설치
brew install pipx
pipx ensurepath

# 2. WonyBot 설치
cd ~/wony-bot
pipx install --editable .

# 3. 실행
wony chat
```

## 방법 4: 직접 실행 🏃

```bash
# 어디서나 전체 경로로 실행
~/wony-bot/wony_launcher.sh chat
```

## 의존성 문제 해결 🔧

만약 "No module named" 에러가 나면:

```bash
# 가상환경 재설정
cd ~/wony-bot
./quick_setup.sh
```

## 제거 방법 🗑️

```bash
# 심볼릭 링크 제거
sudo rm /usr/local/bin/wony

# 또는 alias 제거
# .zshrc에서 wony 관련 라인 삭제
```