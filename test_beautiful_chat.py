#!/usr/bin/env python
"""Test beautiful chat interface"""

import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# Add the app directory to the path
sys.path.insert(0, '/Users/jaewonlee/wony-bot')

from app.services.chat import ChatService
from app.rag import RAGChain

console = Console()

async def test_beautiful_chat():
    """Test beautiful chat interface"""
    
    # Suppress logging for clean output
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Initialize services
    console.print(Rule("[bold cyan]🤖 WonyBot Beautiful Chat Test[/bold cyan]", style="cyan"))
    
    rag_chain = RAGChain()
    chat_service = ChatService(
        enable_memory=True,
        rag_chain=rag_chain,
        enable_agents=True,
        use_hierarchical=True
    )
    
    console.print(Panel.fit(
        "✅ Chat service initialized\n"
        "🏛️ Hierarchical agents enabled\n"
        "💾 Memory system active",
        border_style="green",
        title="System Status"
    ))
    
    # Test simple message
    console.print("\n[bold]Test 1: Simple Message[/bold]")
    console.print(Text("💬 User: 안녕하세요!", style="cyan"))
    
    response = ""
    async for chunk in chat_service.chat("안녕하세요!", stream=False):
        if not chunk.startswith("\n[session:"):
            response = chunk
    
    console.print(Panel(
        response[:200] + "..." if len(response) > 200 else response,
        title="[bold magenta]💬 워니[/bold magenta]",
        border_style="magenta",
        padding=(1, 2)
    ))
    
    # Test complex task
    console.print("\n[bold]Test 2: Complex Task (Hierarchical)[/bold]")
    task = "간단한 TODO 앱의 기능을 분석하고 개선 방안을 제시해줘"
    console.print(Text(f"💬 User: {task}", style="cyan"))
    
    if chat_service.hierarchical_manager:
        console.print(Rule("[cyan]🏛️ Hierarchical Processing[/cyan]", style="cyan"))
        
        with console.status("[yellow]Orchestrator working...[/yellow]", spinner="dots"):
            result = await chat_service.chat_with_hierarchical_agents(
                message=task,
                require_consensus=True
            )
        
        # Display result
        status_text = "✅ Success" if result.get('status') == 'success' else "⚠️ Partial"
        consensus_text = ""
        if 'consensus' in result:
            if result['consensus'].get('reached'):
                consensus_text = f"\n🗳️ Consensus: {result['consensus'].get('approval_rate', 0):.1%} approval"
            else:
                consensus_text = "\n❌ No consensus reached"
        
        console.print(Panel(
            f"{status_text}{consensus_text}\n"
            f"⏱️ Time: {result.get('execution_time', 0):.2f}s",
            title="[bold magenta]💬 워니[/bold magenta]",
            border_style="magenta",
            padding=(1, 2)
        ))
    
    console.print("\n[green]✨ Test completed![/green]")

if __name__ == "__main__":
    asyncio.run(test_beautiful_chat())