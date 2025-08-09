"""CLI styling utilities for WonyBot"""

from rich.theme import Theme
from rich.style import Style

# 워니봇 커스텀 테마
WONY_THEME = Theme({
    "user": "bold cyan",
    "wony": "bold magenta",
    "info": "dim cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "dim": "dim white",
    "highlight": "bold white on magenta",
    "code": "bold yellow on black",
    "timestamp": "dim blue",
    "memory": "dim green",
    "session": "dim cyan"
})

# 이모지 세트
EMOJIS = {
    "wony": "🎀",
    "user": "💬",
    "thinking": "🤔",
    "save": "💾",
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "ℹ️",
    "sparkle": "✨",
    "pin": "📌",
    "clock": "⏰",
    "memory": "🧠",
    "new": "🆕",
    "clear": "🗑️",
    "exit": "👋",
    "help": "❓",
    "yahoo": "🎵"
}

# 프롬프트 스타일
def get_user_prompt():
    """Get styled user prompt"""
    return f"[bold cyan]재원[/bold cyan] {EMOJIS['user']}"

def get_wony_prompt():
    """Get styled Wony prompt"""
    return f"[bold magenta]워니[/bold magenta] {EMOJIS['wony']}"

# 메시지 포맷터
def format_thinking():
    """Format thinking indicator"""
    return f"[dim yellow]{EMOJIS['thinking']} 생각 중...[/dim yellow]"

def format_saving():
    """Format saving indicator"""
    return f"[dim green]{EMOJIS['save']} 메모리 저장 중...[/dim green]"

def format_session_info(session_id: str):
    """Format session info display"""
    short_id = str(session_id)[:8]
    return f"[dim cyan]{EMOJIS['pin']} 세션: {short_id}... | {EMOJIS['memory']} 자동 저장 활성화[/dim cyan]"