#!/usr/bin/env python3
"""WonyBot - GPT-OSS based personal assistant CLI"""

import asyncio
import sys
from typing import Optional
from uuid import UUID
from pathlib import Path
import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.syntax import Syntax

from app.config import settings
from app.core.database import init_db
from app.core.ollama import OllamaClient
from app.services.chat import ChatService
from app.services.prompt import PromptService
from app.rag import RAGChain, VectorStore, EmbeddingManager, DocumentLoader
from app.utils.logger_config import setup_logging
from app.agents.hierarchical_manager import HierarchicalAgentManager
from app.agents.consensus import VoteType

# Setup logging
setup_logging(level=logging.INFO)

# Initialize Typer app
app = typer.Typer(
    name="wony",
    help="🤖 WonyBot - Your personal AI assistant powered by gpt-oss",
    add_completion=True,
    rich_markup_mode="rich"
)

console = Console()
prompt_service = PromptService()
ollama_client = OllamaClient()

# Initialize RAG components (lazy loading)
rag_chain = None
chat_service = None  # Will be initialized with RAG chain

@app.command()
def chat(
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Continue existing session by ID"),
    prompt: str = typer.Option("default", "--prompt", "-p", help="Prompt template to use"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming output"),
    hierarchical: bool = typer.Option(True, "--hierarchical/--no-hierarchical", help="Use hierarchical agent system (default: enabled)"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug messages"),
):
    """Start an interactive chat session with AI agents"""
    
    async def run_chat():
        global chat_service, rag_chain
        
        # Configure logging based on debug flag
        import logging
        if debug:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(name)s: %(message)s')
        else:
            # Suppress debug messages when not in debug mode
            logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
            # Specifically suppress noisy loggers
            logging.getLogger('app.agents').setLevel(logging.WARNING)
            logging.getLogger('app.services').setLevel(logging.WARNING)
            logging.getLogger('app.rag').setLevel(logging.WARNING)
            logging.getLogger('httpx').setLevel(logging.WARNING)
            logging.getLogger('chromadb').setLevel(logging.WARNING)
        
        # Initialize database
        await init_db()
        
        # Initialize RAG chain if not already done
        if rag_chain is None:
            rag_chain = RAGChain()
        
        # Initialize chat service with memory and hierarchical agents support
        if chat_service is None:
            chat_service = ChatService(
                enable_memory=True, 
                rag_chain=rag_chain,
                enable_agents=True,
                use_hierarchical=hierarchical
            )
        
        # Check Ollama health (silently unless debug mode)
        if debug:
            with console.status("[yellow]Checking Ollama connection...[/yellow]"):
                if not await ollama_client.check_health():
                    console.print("[red]❌ Ollama is not running or model is not available![/red]")
                    console.print("[yellow]Please ensure Ollama is running and gpt-oss:20b is pulled.[/yellow]")
                    console.print("\n[dim]Run: ollama pull gpt-oss:20b[/dim]")
                    return
            console.print("[green]✅ Connected to Ollama[/green]")
        else:
            # Silent check
            if not await ollama_client.check_health():
                console.print(Panel.fit(
                    "[red]❌ Ollama 연결 실패[/red]\n\n"
                    "[yellow]Ollama가 실행 중인지 확인해주세요.[/yellow]\n"
                    "[dim]ollama pull gpt-oss:20b[/dim]",
                    border_style="red"
                ))
                return
        
        # Get or create session
        session_id = None
        if session:
            try:
                session_id = UUID(session)
                history = await chat_service.get_session_history(session_id)
                if history:
                    if debug:
                        console.print(f"[green]Continuing session: {session_id}[/green]")
                    # Show last few messages for context
                    if len(history) > 0 and debug:
                        console.print("\n[dim]Recent conversation:[/dim]")
                        for msg in history[-3:]:
                            role_color = "blue" if msg.role.value == "user" else "green"
                            console.print(f"[{role_color}]{msg.role.value.title()}:[/{role_color}] {msg.content[:100]}...")
                else:
                    if debug:
                        console.print(f"[yellow]Session {session_id} not found, starting new session[/yellow]")
                    session_id = None
            except ValueError:
                if debug:
                    console.print(f"[red]Invalid session ID: {session}[/red]")
                session_id = None
        
        # Load prompt template
        prompt_template = await prompt_service.get_prompt(prompt)
        if not prompt_template:
            if debug:
                console.print(f"[yellow]Prompt '{prompt}' not found, using default[/yellow]")
            prompt_template = await prompt_service.get_prompt("default")
        
        system_prompt = prompt_template.content if prompt_template else None
        
        # Show welcome message
        if prompt == "wony" or prompt == "default":
            # Wony's personalized welcome with better design
            console.print(Panel.fit(
                f"[bold magenta]💝 워니 비서 시스템 v1.0[/bold magenta]\n"
                f"[yellow]야호! 재원아~ 나는 너의 개인 비서 워니야! 🎀[/yellow]\n"
                f"[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]\n"
                f"[dim white]💬 티키타카 모드 | 짧고 빠른 대화[/dim white]\n"
                f"[dim white]💾 자동 메모리 저장 | 중요 정보 기억[/dim white]\n"
                f"[dim white]🤖 {'계층적 에이전트' if hierarchical else '기본 에이전트'} | 5명의 전문가[/dim white]\n"
                f"[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]\n"
                f"[dim]🔹 exit/ㅂㅂ - 나가기  🔹 new/새로 - 새 대화[/dim]\n"
                f"[dim]🔹 clear/지워 - 기록 삭제  🔹 help - 도움말[/dim]",
                border_style="magenta",
                title="[bold white]WonyBot[/bold white]",
                subtitle="[dim]AI Personal Assistant[/dim]"
            ))
        else:
            console.print(Panel.fit(
                f"[bold cyan]🤖 WonyBot Chat Interface[/bold cyan]\n"
                f"[dim]Model: {settings.ollama_model} | Prompt: {prompt}[/dim]\n"
                f"[dim]Type 'exit' or 'quit' to end the conversation[/dim]\n"
                f"[dim]Type 'clear' to clear the current session history[/dim]\n"
                f"[dim]Type 'new' to start a new session[/dim]",
                border_style="cyan"
            ))
        
        # Main chat loop
        while True:
            try:
                # Get user input with beautiful prompt
                console.print()  # Add spacing
                user_input = Prompt.ask(Text("💬 재원", style="bold cyan"))
                
                # Handle special commands
                if user_input.lower() in ["exit", "quit", "q", "ㅂㅂ", "바이"]:
                    console.print("[yellow]야호! 다음에 또 만나~ 👋[/yellow]")
                    break
                
                if user_input.lower() in ["clear", "클리어", "지워"]:
                    if session_id and Confirm.ask("[yellow]대화 기록 지울까?[/yellow]"):
                        await chat_service.clear_session(session_id)
                        console.print("[green]✨ 대화 기록 깨끗하게 지웠어![/green]")
                    continue
                
                if user_input.lower() in ["new", "새로", "새대화"]:
                    session_id = None
                    console.print("[green]🆕 야호! 새로운 대화 시작![/green]")
                    continue
                
                # Check if this is a complex task that needs hierarchical processing
                is_complex_task = any(keyword in user_input.lower() for keyword in [
                    "분석", "개선", "제안", "계획", "설계", "구현", "복잡한", "전체적으로",
                    "analyze", "improve", "suggest", "plan", "design", "implement", "complex"
                ])
                
                # Process chat message
                if is_complex_task and hierarchical and chat_service.hierarchical_manager:
                    # Use hierarchical agent system for complex tasks
                    console.print(Rule("[bold cyan]🏛️ 계층적 에이전트 시스템[/bold cyan]", style="cyan"))
                    
                    with console.status("[yellow]Orchestrator가 작업을 분석하고 분배 중...[/yellow]", spinner="dots"):
                        result = await chat_service.chat_with_hierarchical_agents(
                            message=user_input,
                            session_id=session_id,
                            require_consensus=True
                        )
                    
                    # Display result with pretty formatting
                    console.print()
                    # Build response text
                    status_text = "[green]✅ 작업 완료![/green]" if result.get('status') == 'success' else "[red]❌ 작업 실패[/red]"
                    consensus_text = ""
                    if 'consensus' in result and result['consensus'].get('reached'):
                        consensus_text = f"\n[cyan]🗳️ 합의: {result['consensus']['approval_rate']:.1%} 승인[/cyan]"
                    
                    response_panel = Panel.fit(
                        Text.from_markup(
                            f"[bold magenta]💬 워니의 답변[/bold magenta]\n\n"
                            f"{status_text}"
                            f"{consensus_text}"
                        ),
                        border_style="magenta",
                        padding=(1, 2)
                    )
                    console.print(response_panel)
                    
                    # Update session ID if needed
                    if not session_id and result.get('status') in ['success', 'partial']:
                        session_id = await chat_service.session_manager.create_session()
                        session_id = session_id.id if hasattr(session_id, 'id') else session_id
                        
                    continue
                
                # Regular chat response
                response_text = ""
                current_session_id = None
                
                if no_stream:
                    # Non-streaming mode with beautiful status
                    with console.status("[yellow]💭 생각 중...[/yellow]", spinner="dots"):
                        async for chunk in chat_service.chat(
                            message=user_input,
                            session_id=session_id,
                            system_prompt=system_prompt,
                            stream=False
                        ):
                            if chunk.startswith("\n[session:"):
                                current_session_id = chunk.strip()[9:-1]
                            else:
                                response_text = chunk
                    
                    # Display in beautiful panel
                    console.print()
                    response_panel = Panel(
                        Markdown(response_text),
                        title="[bold magenta]💬 워니[/bold magenta]",
                        border_style="magenta",
                        padding=(1, 2),
                        expand=False
                    )
                    console.print(response_panel)
                else:
                    # Streaming mode with live panel update
                    console.print()
                    response_parts = []
                    
                    with Live(
                        Panel(
                            Text("💭 응답 생성 중...", style="italic dim"),
                            title="[bold magenta]💬 워니[/bold magenta]",
                            border_style="magenta",
                            padding=(1, 2),
                            expand=False
                        ),
                        console=console,
                        refresh_per_second=10
                    ) as live:
                        async for chunk in chat_service.chat(
                            message=user_input,
                            session_id=session_id,
                            system_prompt=system_prompt,
                            stream=True
                        ):
                            if chunk.startswith("\n[session:"):
                                current_session_id = chunk.strip()[9:-1]
                            else:
                                response_parts.append(chunk)
                                # Update with markdown in panel
                                live.update(
                                    Panel(
                                        Markdown("".join(response_parts)),
                                        title="[bold magenta]💬 워니[/bold magenta]",
                                        border_style="magenta",
                                        padding=(1, 2),
                                        expand=False
                                    )
                                )
                    
                    response_text = "".join(response_parts)
                
                # Update session ID for next iteration
                if current_session_id:
                    session_id = UUID(current_session_id)
                    if not session:  # Only show for new sessions
                        console.print(f"\n[dim cyan]📌 세션: {str(session_id)[:8]}... | 💾 자동 저장 중[/dim cyan]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                continue
    
    asyncio.run(run_chat())

@app.command()
def history(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of sessions to show"),
):
    """Show chat history and sessions"""
    
    async def show_history():
        await init_db()
        
        # Initialize chat service if needed
        global chat_service
        if chat_service is None:
            from app.services.chat import ChatService
            chat_service = ChatService()
        
        sessions = await chat_service.list_sessions(limit=limit)
        
        if not sessions:
            console.print("[yellow]No chat sessions found[/yellow]")
            return
        
        table = Table(title="Chat Sessions", show_header=True, header_style="bold cyan")
        table.add_column("Session ID", style="dim")
        table.add_column("Name")
        table.add_column("Created", style="green")
        table.add_column("Updated", style="yellow")
        
        for session in sessions:
            table.add_row(
                str(session.id)[:8] + "...",
                session.name or "Unnamed",
                session.created_at.strftime("%Y-%m-%d %H:%M"),
                session.updated_at.strftime("%Y-%m-%d %H:%M")
            )
        
        console.print(table)
        console.print(f"\n[dim]Use 'wony chat --session <id>' to continue a session[/dim]")
    
    asyncio.run(show_history())

@app.command()
def prompts(
    action: str = typer.Argument("list", help="Action: list, show, create, delete"),
    name: Optional[str] = typer.Argument(None, help="Prompt name"),
):
    """Manage prompt templates"""
    
    async def manage_prompts():
        await init_db()
        
        if action == "list":
            prompts = await prompt_service.list_prompts()
            if prompts:
                console.print("[bold cyan]Available Prompts:[/bold cyan]")
                for p in prompts:
                    console.print(f"  • {p}")
            else:
                console.print("[yellow]No prompts available[/yellow]")
        
        elif action == "show" and name:
            prompt = await prompt_service.get_prompt(name)
            if prompt:
                console.print(Panel(
                    prompt.content,
                    title=f"Prompt: {name}",
                    border_style="cyan"
                ))
            else:
                console.print(f"[red]Prompt '{name}' not found[/red]")
        
        elif action == "create" and name:
            console.print(f"Creating prompt '{name}'")
            console.print("[dim]Enter prompt content (press Ctrl+D when done):[/dim]")
            
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            
            content = "\n".join(lines)
            prompt = await prompt_service.create_prompt(name, content)
            console.print(f"[green]✅ Prompt '{name}' created[/green]")
        
        elif action == "delete" and name:
            if Confirm.ask(f"Delete prompt '{name}'?"):
                if await prompt_service.delete_prompt(name):
                    console.print(f"[green]✅ Prompt '{name}' deleted[/green]")
                else:
                    console.print(f"[red]Failed to delete prompt '{name}'[/red]")
        
        else:
            console.print("[red]Invalid action or missing name[/red]")
            console.print("Usage: wony prompts [list|show|create|delete] [name]")
    
    asyncio.run(manage_prompts())

@app.command()
def config(
    action: str = typer.Argument("show", help="Action: show, set"),
    key: Optional[str] = typer.Argument(None, help="Configuration key"),
    value: Optional[str] = typer.Argument(None, help="Configuration value"),
):
    """Manage configuration settings"""
    
    if action == "show":
        console.print("[bold cyan]Current Configuration:[/bold cyan]")
        console.print(f"  • Ollama Host: {settings.ollama_host}")
        console.print(f"  • Model: {settings.ollama_model}")
        console.print(f"  • Database: {settings.database_url}")
        console.print(f"  • Max History: {settings.max_history_length} messages")
        console.print(f"  • Streaming: {settings.streaming_enabled}")
    
    elif action == "set" and key and value:
        # This would typically update a config file or database
        console.print(f"[yellow]Setting {key} = {value}[/yellow]")
        console.print("[dim]Note: Runtime configuration changes not yet implemented[/dim]")
    
    else:
        console.print("[red]Invalid action or missing arguments[/red]")
        console.print("Usage: wony config [show|set] [key] [value]")

@app.command()
def setup():
    """Initial setup and model download"""
    
    async def run_setup():
        console.print(Panel.fit(
            "[bold cyan]🚀 WonyBot Setup Wizard[/bold cyan]",
            border_style="cyan"
        ))
        
        # Check Ollama
        with console.status("[yellow]Checking Ollama...[/yellow]"):
            if not await ollama_client.check_health():
                console.print("[red]❌ Ollama is not running![/red]")
                console.print("[yellow]Please install and start Ollama first:[/yellow]")
                console.print("  Visit: https://ollama.ai")
                return
        
        console.print("[green]✅ Ollama is running[/green]")
        
        # Check model
        models = await ollama_client.list_models()
        if settings.ollama_model in models:
            console.print(f"[green]✅ Model {settings.ollama_model} is already installed[/green]")
        else:
            console.print(f"[yellow]Model {settings.ollama_model} not found[/yellow]")
            if Confirm.ask(f"Download {settings.ollama_model}? (This may take a while)"):
                success = await ollama_client.pull_model()
                if success:
                    console.print(f"[green]✅ Model downloaded successfully[/green]")
                else:
                    console.print(f"[red]❌ Failed to download model[/red]")
                    return
        
        # Initialize database
        console.print("[yellow]Initializing database...[/yellow]")
        await init_db()
        console.print("[green]✅ Database initialized[/green]")
        
        console.print("\n[bold green]🎉 Setup complete! You can now use 'wony chat' to start chatting.[/bold green]")
    
    asyncio.run(run_setup())

@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory path to index"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Recursively index directories"),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection name"),
    glob: str = typer.Option("**/*", "--glob", "-g", help="Glob pattern for files"),
):
    """Index documents for RAG search"""
    
    async def run_index():
        global rag_chain
        
        # Initialize RAG chain if needed
        if rag_chain is None:
            console.print("[yellow]Initializing RAG system...[/yellow]")
            rag_chain = RAGChain(collection_name=collection)
            console.print("[green]✅ RAG system initialized[/green]")
        
        path_obj = Path(path)
        
        with console.status(f"[yellow]Indexing {path}...[/yellow]"):
            try:
                if path_obj.is_file():
                    # Index single file
                    count = rag_chain.index_document(str(path_obj))
                    console.print(f"[green]✅ Indexed {count} chunks from {path_obj.name}[/green]")
                elif path_obj.is_dir():
                    # Index directory
                    count = rag_chain.index_directory(
                        directory=str(path_obj),
                        glob_pattern=glob,
                        recursive=recursive
                    )
                    console.print(f"[green]✅ Indexed {count} chunks from directory[/green]")
                else:
                    console.print(f"[red]❌ Path not found: {path}[/red]")
                    return
                
                # Show stats
                stats = rag_chain.get_stats()
                console.print(f"\n[dim]Collection '{collection}' now has {stats['vector_store']['count']} documents[/dim]")
                
            except Exception as e:
                console.print(f"[red]❌ Indexing failed: {e}[/red]")
    
    asyncio.run(run_index())

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of results"),
    search_type: str = typer.Option("hybrid", "--type", "-t", help="Search type: hybrid, vector, keyword"),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection name"),
):
    """Search indexed documents"""
    
    async def run_search():
        global rag_chain
        
        # Initialize RAG chain if needed
        if rag_chain is None:
            rag_chain = RAGChain(collection_name=collection)
        
        # Perform search
        with console.status(f"[yellow]Searching for: {query}[/yellow]"):
            results = rag_chain.search(
                query=query,
                top_k=top_k,
                search_type=search_type
            )
        
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Display results
        console.print(f"\n[bold cyan]Found {len(results)} results:[/bold cyan]\n")
        
        for i, result in enumerate(results, 1):
            # Extract source info
            source = result['metadata'].get('source', 'Unknown')
            score = result['score']
            content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            
            # Display result
            console.print(Panel(
                f"[dim]{content}[/dim]\n\n"
                f"[yellow]Source:[/yellow] {source}\n"
                f"[green]Score:[/green] {score:.3f}",
                title=f"Result {i}",
                border_style="cyan"
            ))
    
    asyncio.run(run_search())

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of documents to use"),
    search_type: str = typer.Option("hybrid", "--type", "-t", help="Search type: hybrid, vector, keyword"),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection name"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming output"),
):
    """Ask questions using RAG"""
    
    async def run_ask():
        global rag_chain
        
        # Initialize database
        await init_db()
        
        # Check Ollama
        if not await ollama_client.check_health():
            console.print("[red]❌ Ollama is not running![/red]")
            return
        
        # Initialize RAG chain if needed
        if rag_chain is None:
            console.print("[yellow]Initializing RAG system...[/yellow]")
            rag_chain = RAGChain(collection_name=collection)
            console.print("[green]✅ RAG system initialized[/green]")
        
        # Check if collection has documents
        stats = rag_chain.get_stats()
        if stats['vector_store']['count'] == 0:
            console.print(f"[yellow]⚠️  No documents indexed in collection '{collection}'[/yellow]")
            console.print("[dim]Use 'wony index <path>' to index documents first[/dim]")
            return
        
        console.print(f"\n[bold blue]Question:[/bold blue] {question}")
        console.print("\n[bold green]Answer:[/bold green] ", end="")
        
        # Get answer
        if no_stream:
            # Non-streaming mode
            with console.status("[yellow]Thinking...[/yellow]"):
                response_text = ""
                async for chunk in rag_chain.ask(
                    question=question,
                    top_k=top_k,
                    search_type=search_type,
                    stream=False
                ):
                    response_text = chunk
            
            console.print(Markdown(response_text))
        else:
            # Streaming mode
            response_parts = []
            with Live("", console=console, refresh_per_second=10) as live:
                async for chunk in rag_chain.ask(
                    question=question,
                    top_k=top_k,
                    search_type=search_type,
                    stream=True
                ):
                    response_parts.append(chunk)
                    live.update(Text("".join(response_parts)))
    
    asyncio.run(run_ask())

@app.command()
def rag_stats(
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection name"),
):
    """Show RAG system statistics"""
    
    async def show_stats():
        global rag_chain
        
        # Initialize RAG chain if needed
        if rag_chain is None:
            rag_chain = RAGChain(collection_name=collection)
        
        stats = rag_chain.get_stats()
        
        # Display stats
        console.print(Panel.fit(
            "[bold cyan]RAG System Statistics[/bold cyan]",
            border_style="cyan"
        ))
        
        # Vector store stats
        vs_stats = stats['vector_store']
        console.print(f"\n[yellow]Vector Store:[/yellow]")
        console.print(f"  • Collection: {vs_stats['name']}")
        console.print(f"  • Documents: {vs_stats['count']}")
        console.print(f"  • Storage: {vs_stats['persist_directory']}")
        
        # Embedding cache stats
        cache_stats = stats['embedding_cache']
        console.print(f"\n[yellow]Embedding Cache:[/yellow]")
        console.print(f"  • Cached Embeddings: {cache_stats['cached_embeddings']}")
        console.print(f"  • Cache Size: {cache_stats['cache_size_mb']} MB")
        console.print(f"  • Model: {cache_stats['model_name']}")
        console.print(f"  • Dimensions: {cache_stats['embedding_dim']}")
        console.print(f"  • Device: {cache_stats['device']}")
        
        # Available collections
        console.print(f"\n[yellow]Available Collections:[/yellow]")
        for coll in stats['collections']:
            console.print(f"  • {coll}")
    
    asyncio.run(show_stats())

@app.command()
def memories(
    importance: Optional[str] = typer.Option(None, "--importance", "-i", help="Filter by importance: LOW, MEDIUM, HIGH, CRITICAL"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Filter by tags (comma-separated)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of memories to show"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Filter by session ID"),
):
    """View saved conversation memories"""
    
    async def show_memories(importance_level=importance, tags_filter=tags, memory_limit=limit, session_filter=session):
        global chat_service, rag_chain
        
        # Initialize if needed
        await init_db()
        if rag_chain is None:
            rag_chain = RAGChain()
        if chat_service is None:
            from app.services.chat import ChatService
            chat_service = ChatService(enable_memory=True, rag_chain=rag_chain)
        
        if not chat_service.memory_manager:
            console.print("[yellow]Memory system not enabled[/yellow]")
            return
        
        # Build query for memories
        query = "recent important conversations"
        
        # Parse importance filter
        importance_filter = None
        if importance_level:
            try:
                from app.services.memory_manager import ImportanceLevel
                importance_filter = ImportanceLevel[importance_level.upper()]
            except KeyError:
                console.print(f"[red]Invalid importance level: {importance_level}[/red]")
                console.print("Valid levels: LOW, MEDIUM, HIGH, CRITICAL")
                return
        
        # Parse tags filter
        tag_filter = tags_filter.split(",") if tags_filter else None
        
        # Query memories
        with console.status("[yellow]Searching memories...[/yellow]"):
            memories = await chat_service.memory_manager.query_memories(
                query=query,
                top_k=memory_limit,
                importance_filter=importance_filter,
                tag_filter=tag_filter
            )
        
        if not memories:
            console.print("[yellow]No memories found matching criteria[/yellow]")
            return
        
        # Display memories
        console.print(f"\n[bold cyan]Found {len(memories)} memories:[/bold cyan]\n")
        
        for i, memory in enumerate(memories, 1):
            metadata = memory.get('metadata', {})
            content = memory.get('content', '')
            score = memory.get('score', 0)
            
            # Extract display info
            importance_str = metadata.get('importance', 'UNKNOWN')
            tags_str = metadata.get('tags', '')
            timestamp = metadata.get('timestamp', '')
            session_id = metadata.get('session_id', '')
            
            # Format content preview
            preview = content[:200] + "..." if len(content) > 200 else content
            
            # Color code by importance
            importance_colors = {
                'CRITICAL': 'red',
                'HIGH': 'yellow',
                'MEDIUM': 'cyan',
                'LOW': 'dim'
            }
            color = importance_colors.get(importance_str, 'white')
            
            # Display memory
            console.print(Panel(
                f"[{color}]{preview}[/{color}]\n\n"
                f"[yellow]Importance:[/yellow] {importance_str}\n"
                f"[green]Tags:[/green] {tags_str}\n"
                f"[blue]Time:[/blue] {timestamp}\n"
                f"[dim]Session:[/dim] {session_id[:8] if session_id else 'N/A'}\n"
                f"[magenta]Relevance:[/magenta] {score:.3f}",
                title=f"Memory {i}",
                border_style=color
            ))
    
    asyncio.run(show_memories())

@app.command()
def memory_stats():
    """Show memory system statistics"""
    
    async def show_stats():
        global chat_service, rag_chain
        
        # Initialize if needed
        await init_db()
        if rag_chain is None:
            rag_chain = RAGChain()
        if chat_service is None:
            from app.services.chat import ChatService
            chat_service = ChatService(enable_memory=True, rag_chain=rag_chain)
        
        if not chat_service.memory_manager:
            console.print("[yellow]Memory system not enabled[/yellow]")
            return
        
        # Get stats
        stats = chat_service.memory_manager.get_memory_stats()
        
        # Display stats
        console.print(Panel.fit(
            "[bold cyan]Memory System Statistics[/bold cyan]",
            border_style="cyan"
        ))
        
        console.print(f"\n[yellow]General:[/yellow]")
        console.print(f"  • Buffer Size: {stats['buffer_size']} entries")
        console.print(f"  • Auto-Save: {stats['auto_save']}")
        console.print(f"  • Threshold: {stats['importance_threshold']}")
        console.print(f"  • Collection: {stats['collection']}")
        
        # Importance distribution
        if stats.get('importance_distribution'):
            console.print(f"\n[yellow]Importance Distribution:[/yellow]")
            for level, count in stats['importance_distribution'].items():
                console.print(f"  • {level}: {count}")
        
        # Tag distribution
        if stats.get('tag_distribution'):
            console.print(f"\n[yellow]Top Tags:[/yellow]")
            sorted_tags = sorted(stats['tag_distribution'].items(), key=lambda x: x[1], reverse=True)
            for tag, count in sorted_tags[:10]:
                console.print(f"  • {tag}: {count}")
        
        # Check RAG stats for memory collection
        if rag_chain:
            rag_stats = rag_chain.get_stats()
            # Check if chat_history collection exists
            if 'chat_history' in rag_stats.get('collections', []):
                console.print(f"\n[yellow]Indexed Memories:[/yellow]")
                # Would need to query the collection for count
                console.print(f"  • Collection 'chat_history' is active")
    
    asyncio.run(show_stats())

@app.command()
def summarize_session(
    session_id: str = typer.Argument(..., help="Session ID to summarize"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save summary to memory"),
):
    """Summarize a chat session"""
    
    async def run_summarize():
        global chat_service, rag_chain
        
        # Initialize
        await init_db()
        if rag_chain is None:
            rag_chain = RAGChain()
        if chat_service is None:
            from app.services.chat import ChatService
            chat_service = ChatService(enable_memory=True, rag_chain=rag_chain)
        
        # Parse session ID
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            console.print(f"[red]Invalid session ID: {session_id}[/red]")
            return
        
        # Get session history
        history = await chat_service.get_session_history(session_uuid)
        if not history:
            console.print(f"[red]Session not found: {session_id}[/red]")
            return
        
        console.print(f"[yellow]Summarizing session with {len(history)} messages...[/yellow]")
        
        # Create summary
        if chat_service.memory_manager:
            summary = await chat_service.memory_manager.summarize_session(
                messages=history,
                session_id=session_uuid if save else None
            )
            
            # Display summary
            console.print(Panel(
                Markdown(summary),
                title=f"Session Summary - {session_id[:8]}",
                border_style="cyan"
            ))
            
            if save:
                console.print("[green]✅ Summary saved to memory[/green]")
        else:
            console.print("[yellow]Memory system not enabled[/yellow]")
    
    asyncio.run(run_summarize())

@app.command()
def agents(
    action: str = typer.Argument("list", help="Action: list, status, assign, history"),
    agent_id: Optional[str] = typer.Argument(None, help="Agent ID or role"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Task for agent"),
):
    """Manage AI agents"""
    
    async def manage_agents():
        global chat_service, rag_chain
        
        # Initialize if needed
        await init_db()
        if rag_chain is None:
            rag_chain = RAGChain()
        if chat_service is None:
            from app.services.chat import ChatService
            chat_service = ChatService(enable_memory=True, rag_chain=rag_chain, enable_agents=True)
        
        if not chat_service.agent_manager:
            console.print("[yellow]Agent system not enabled[/yellow]")
            return
        
        agent_manager = chat_service.agent_manager
        
        if action == "list":
            # List all agents
            agents = agent_manager.list_agents()
            
            if not agents:
                console.print("[yellow]No agents available[/yellow]")
                return
            
            table = Table(title="Available Agents", show_header=True, header_style="bold cyan")
            table.add_column("ID", style="dim")
            table.add_column("Name")
            table.add_column("Role", style="yellow")
            table.add_column("Status", style="green")
            table.add_column("Tasks", style="cyan")
            
            for agent in agents:
                status_color = {
                    'idle': 'green',
                    'working': 'yellow',
                    'error': 'red',
                    'paused': 'dim'
                }.get(agent['status'], 'white')
                
                table.add_row(
                    agent['id'],
                    agent['name'],
                    agent['role'],
                    f"[{status_color}]{agent['status']}[/{status_color}]",
                    str(agent['tasks_completed'])
                )
            
            console.print(table)
            
            # Show capabilities
            console.print("\n[bold cyan]Agent Capabilities:[/bold cyan]")
            for agent in agents:
                console.print(f"\n[yellow]{agent['name']}:[/yellow]")
                for capability in agent['capabilities']:
                    console.print(f"  • {capability}")
        
        elif action == "status":
            # Get overall status
            status = agent_manager.get_all_status()
            
            console.print(Panel.fit(
                "[bold cyan]Agent System Status[/bold cyan]",
                border_style="cyan"
            ))
            
            console.print(f"\n[yellow]Overview:[/yellow]")
            console.print(f"  • Active Agents: {status['active_agents']}")
            console.print(f"  • Queue Length: {status['queue_length']}")
            console.print(f"  • Completed Tasks: {status['completed_tasks']}")
            
            console.print(f"\n[yellow]Agent Status:[/yellow]")
            for agent in status['agents']:
                status_color = 'green' if agent['status'] == 'idle' else 'yellow'
                console.print(f"  • {agent['name']}: [{status_color}]{agent['status']}[/{status_color}]")
                if agent['current_task']:
                    console.print(f"    Current: {agent['current_task']['task'][:50]}...")
        
        elif action == "assign" and task:
            # Assign task to agent
            console.print(f"[yellow]Assigning task to {agent_id or 'best available agent'}...[/yellow]")
            
            with console.status("[yellow]Processing...[/yellow]"):
                result = await agent_manager.assign_task(
                    task=task,
                    agent_id=agent_id
                )
            
            if result['status'] == 'success':
                console.print(f"[green]✅ Task completed by {result['agent']}[/green]")
                
                # Display result based on agent type
                if 'findings' in result:
                    console.print(Panel(
                        Markdown(result['findings']),
                        title="Research Findings",
                        border_style="cyan"
                    ))
                elif 'code_blocks' in result:
                    console.print("[yellow]Generated Code:[/yellow]")
                    for code in result['code_blocks']:
                        console.print(Panel(
                            code,
                            border_style="green"
                        ))
                elif 'analysis' in result:
                    console.print(Panel(
                        Markdown(result['analysis']),
                        title="Analysis Results",
                        border_style="cyan"
                    ))
                elif 'summary' in result:
                    console.print(Panel(
                        Markdown(result['summary']),
                        title="Summary",
                        border_style="cyan"
                    ))
                elif 'creative_output' in result:
                    console.print(Panel(
                        Markdown(result['creative_output']),
                        title="Creative Output",
                        border_style="magenta"
                    ))
            elif result['status'] == 'queued':
                console.print(f"[yellow]Task queued (position {result['queue_position']})[/yellow]")
            else:
                console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
        
        elif action == "history":
            # Show task history
            history = agent_manager.get_task_history(limit=10)
            
            if not history:
                console.print("[yellow]No task history available[/yellow]")
                return
            
            console.print(Panel.fit(
                "[bold cyan]Recent Task History[/bold cyan]",
                border_style="cyan"
            ))
            
            for i, task_info in enumerate(history, 1):
                console.print(f"\n[yellow]Task {i}:[/yellow]")
                console.print(f"  • Task: {task_info['task'][:100]}...")
                console.print(f"  • Agent: {task_info['agent']}")
                console.print(f"  • Status: {task_info['result']['status']}")
                console.print(f"  • Time: {task_info['completed_at'].strftime('%Y-%m-%d %H:%M')}")
        
        else:
            console.print("[red]Invalid action or missing arguments[/red]")
            console.print("Usage: wony agents [list|status|assign|history] [agent_id] [--task]")
    
    asyncio.run(manage_agents())

@app.command()
def agent_chat(
    agent: str = typer.Option("auto", "--agent", "-a", help="Agent type: researcher, coder, analyst, summarizer, creative, auto"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Continue existing session by ID"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming output"),
):
    """Chat with a specific AI agent"""
    
    async def run_agent_chat():
        global chat_service, rag_chain
        
        # Initialize
        await init_db()
        if rag_chain is None:
            rag_chain = RAGChain()
        if chat_service is None:
            from app.services.chat import ChatService
            chat_service = ChatService(enable_memory=True, rag_chain=rag_chain, enable_agents=True)
        
        if not chat_service.agent_manager:
            console.print("[red]Agent system not enabled[/red]")
            return
        
        # Check Ollama
        if not await ollama_client.check_health():
            console.print("[red]❌ Ollama is not running![/red]")
            return
        
        agent_manager = chat_service.agent_manager
        
        # Map agent type to ID
        agent_map = {
            'researcher': 'researcher',
            'coder': 'coder',
            'analyst': 'analyst',
            'summarizer': 'summarizer',
            'creative': 'creative',
            'auto': None
        }
        
        agent_id = agent_map.get(agent, None)
        
        # Get session if provided
        session_id = None
        if session:
            try:
                session_id = UUID(session)
            except ValueError:
                console.print(f"[red]Invalid session ID: {session}[/red]")
        
        # Show agent info
        if agent_id:
            agent_obj = agent_manager.get_agent(agent_id)
            if agent_obj:
                console.print(Panel.fit(
                    f"[bold cyan]🤖 {agent_obj.name}[/bold cyan]\n"
                    f"[yellow]Role: {agent_obj.role.value}[/yellow]\n"
                    f"[dim]{agent_obj.description}[/dim]\n\n"
                    f"[green]Capabilities:[/green]\n" + 
                    "\n".join(f"  • {cap}" for cap in agent_obj.capabilities),
                    border_style="cyan",
                    title="Agent Assistant"
                ))
        else:
            console.print(Panel.fit(
                "[bold cyan]🤖 Auto Agent Selection[/bold cyan]\n"
                "[yellow]The best agent will be automatically selected based on your task[/yellow]\n\n"
                "[dim]Available agents: researcher, coder, analyst, summarizer, creative[/dim]",
                border_style="cyan"
            ))
        
        console.print("[dim]Type 'exit' to quit, 'switch <agent>' to change agent[/dim]\n")
        
        # Chat loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                # Handle commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                if user_input.lower().startswith("switch "):
                    new_agent = user_input[7:].strip()
                    if new_agent in agent_map:
                        agent_id = agent_map[new_agent]
                        console.print(f"[green]Switched to {new_agent} agent[/green]")
                    else:
                        console.print(f"[red]Unknown agent: {new_agent}[/red]")
                    continue
                
                # Process with agent
                console.print(f"\n[bold magenta]Agent[/bold magenta] ", end="")
                
                with console.status("[yellow]Processing...[/yellow]"):
                    result = await chat_service.chat_with_agent(
                        message=user_input,
                        agent_id=agent_id,
                        session_id=session_id
                    )
                
                # Display result
                if result['status'] == 'success':
                    agent_name = result.get('agent', 'Unknown')
                    console.print(f"[dim]({agent_name})[/dim]")
                    
                    # Format output based on result type
                    if 'findings' in result:
                        console.print(Markdown(result['findings']))
                    elif 'response' in result:
                        console.print(Markdown(result['response']))
                    elif 'analysis' in result:
                        console.print(Markdown(result['analysis']))
                    elif 'summary' in result:
                        console.print(Markdown(result['summary']))
                    elif 'creative_output' in result:
                        console.print(Markdown(result['creative_output']))
                    
                    # Show code blocks if present
                    if 'code_blocks' in result and result['code_blocks']:
                        for code in result['code_blocks']:
                            console.print(Panel(
                                code,
                                border_style="green",
                                title="Code"
                            ))
                else:
                    console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/red]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                continue
    
    asyncio.run(run_agent_chat())

@app.command()
def hierarchical(
    action: str = typer.Argument("status", help="Action: status, process, hierarchy, consensus"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Task to process"),
    consensus: bool = typer.Option(True, "--consensus/--no-consensus", help="Require consensus"),
):
    """Hierarchical Agent System with Orchestrator and Consensus"""
    
    async def run_hierarchical():
        # Initialize
        await init_db()
        
        # Create hierarchical manager
        manager = HierarchicalAgentManager(
            ollama_client=ollama_client,
            consensus_threshold=0.6,
            vote_type=VoteType.SIMPLE_MAJORITY
        )
        
        if action == "status":
            # Show system status
            status = manager.get_system_status()
            
            console.print(Panel.fit(
                "[bold cyan]🏛️ Hierarchical Agent System Status[/bold cyan]",
                border_style="cyan"
            ))
            
            # Orchestrator status
            console.print("\n[yellow]📋 Orchestrator:[/yellow]")
            orch_status = status['orchestrator']
            console.print(f"  • Status: {orch_status['status']}")
            console.print(f"  • Active Tasks: {len(orch_status['tasks']['active'])}")
            console.print(f"  • Queued Tasks: {len(orch_status['tasks']['queue'])}")
            console.print(f"  • Completed Tasks: {len(orch_status['tasks']['completed'])}")
            
            # Worker status
            console.print("\n[yellow]👷 Workers:[/yellow]")
            worker_status = status['workers']
            console.print(f"  • Active Agents: {worker_status['active_agents']}")
            console.print(f"  • Queue Length: {worker_status['queue_length']}")
            console.print(f"  • Completed Tasks: {worker_status['completed_tasks']}")
            
            # Consensus status
            console.print("\n[yellow]🗳️ Consensus System:[/yellow]")
            consensus_status = status['consensus']
            console.print(f"  • Active Proposals: {consensus_status['active_proposals']}")
            if consensus_status['voting_history']:
                console.print("  • Recent Decisions:")
                for decision in consensus_status['voting_history'][:3]:
                    console.print(f"    - {decision['result']}: "
                                f"{decision['approval_rate']:.1%} approval")
            
            # System metrics
            console.print("\n[yellow]📊 System Metrics:[/yellow]")
            metrics = status['metrics']
            console.print(f"  • Tasks Processed: {metrics['tasks_processed']}")
            console.print(f"  • Consensus Decisions: {metrics['consensus_decisions']}")
            console.print(f"  • Avg Consensus Time: {metrics['average_consensus_time']:.2f}s")
            console.print(f"  • Task Success Rate: {metrics['task_success_rate']:.1%}")
        
        elif action == "process" and task:
            # Process a complex task
            console.print(f"[yellow]🚀 Processing complex task through hierarchical system...[/yellow]")
            console.print(f"Task: {task}")
            console.print(f"Consensus Required: {consensus}")
            
            with console.status("[yellow]Orchestrating task decomposition...[/yellow]"):
                result = await manager.process_complex_task(
                    task=task,
                    require_consensus=consensus
                )
            
            if result['status'] == 'success':
                console.print("[green]✅ Task processing completed successfully![/green]")
                
                # Show orchestration results
                orchestration = result['orchestration']
                console.print("\n[cyan]📋 Orchestration Summary:[/cyan]")
                console.print(f"  • Original Task: {orchestration['original_task']}")
                console.print(f"  • Subtasks Created: {orchestration['subtasks']}")
                
                # Show distribution
                if 'distribution' in orchestration:
                    console.print("\n[cyan]📦 Task Distribution:[/cyan]")
                    for task_id, dist in orchestration['distribution'].items():
                        console.print(f"  • {dist['task'][:50]}...")
                        console.print(f"    Assigned to: {dist.get('assigned_to', 'Unassigned')}")
                
                # Show consensus results
                if consensus and 'consensus' in result:
                    consensus_result = result['consensus']
                    console.print("\n[cyan]🗳️ Consensus Results:[/cyan]")
                    console.print(f"  • Consensus Reached: {consensus_result['reached']}")
                    console.print(f"  • Approval Rate: {consensus_result['approval_rate']:.1%}")
                    console.print(f"  • Participation: {consensus_result['participation_rate']:.1%}")
                    console.print(f"  • Decision: {consensus_result['decision']}")
                    console.print(f"  • Total Votes: {consensus_result['votes']}")
                
                console.print(f"\n[dim]Execution Time: {result['execution_time']:.2f}s[/dim]")
            else:
                console.print(f"[red]❌ Task processing failed: {result.get('error', 'Unknown error')}[/red]")
        
        elif action == "hierarchy":
            # Show agent hierarchy
            hierarchy = manager.get_agent_hierarchy()
            
            console.print(Panel.fit(
                "[bold cyan]🏛️ Agent Hierarchy Structure[/bold cyan]",
                border_style="cyan"
            ))
            
            # Orchestrator
            orch = hierarchy['orchestrator']
            console.print("\n[yellow]👑 Orchestrator (Central Coordinator):[/yellow]")
            console.print(f"  • Name: {orch['name']}")
            console.print(f"  • Status: {orch['status']}")
            console.print("  • Capabilities:")
            for cap in orch['capabilities']:
                console.print(f"    - {cap}")
            
            # Workers
            console.print("\n[yellow]👷 Worker Agents:[/yellow]")
            workers_table = Table(show_header=True, header_style="bold cyan")
            workers_table.add_column("ID", style="dim")
            workers_table.add_column("Name")
            workers_table.add_column("Role", style="yellow")
            workers_table.add_column("Status", style="green")
            
            for worker in hierarchy['workers']:
                workers_table.add_row(
                    worker['id'],
                    worker['name'],
                    worker['role'],
                    worker['status']
                )
            
            console.print(workers_table)
            
            # Consensus configuration
            consensus_cfg = hierarchy['consensus_system']
            console.print("\n[yellow]🗳️ Consensus Configuration:[/yellow]")
            console.print(f"  • Vote Type: {consensus_cfg['vote_type']}")
            console.print(f"  • Threshold: {consensus_cfg['threshold']:.1%}")
            console.print("  • Agent Weights:")
            for agent_id, weight in consensus_cfg['weights'].items():
                console.print(f"    - {agent_id}: {weight}")
        
        elif action == "consensus":
            # Show consensus history
            status = manager.get_system_status()
            voting_history = status['consensus']['voting_history']
            
            console.print(Panel.fit(
                "[bold cyan]🗳️ Consensus Voting History[/bold cyan]",
                border_style="cyan"
            ))
            
            if not voting_history:
                console.print("[yellow]No voting history available yet[/yellow]")
            else:
                for i, decision in enumerate(voting_history, 1):
                    console.print(f"\n[yellow]Decision {i}:[/yellow]")
                    console.print(f"  • Proposal: {decision['proposal_id'][:8]}...")
                    console.print(f"  • Result: {decision['result']}")
                    console.print(f"  • Consensus: {'✅' if decision['consensus_reached'] else '❌'}")
                    console.print(f"  • Approval Rate: {decision['approval_rate']:.1%}")
                    console.print(f"  • Participation: {decision['participation_rate']:.1%}")
                    console.print(f"  • Votes Cast: {len(decision['votes'])}")
                    console.print(f"  • Time: {decision['timestamp']}")
        
        else:
            console.print("[red]Invalid action or missing task[/red]")
            console.print("Usage: wony hierarchical [status|process|hierarchy|consensus] [--task]")
    
    asyncio.run(run_hierarchical())

@app.command()
def clear_index(
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection name"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clear all indexed documents"""
    
    async def run_clear():
        global rag_chain
        
        # Initialize RAG chain if needed
        if rag_chain is None:
            rag_chain = RAGChain(collection_name=collection)
        
        # Get current stats
        stats = rag_chain.get_stats()
        doc_count = stats['vector_store']['count']
        
        if doc_count == 0:
            console.print(f"[yellow]Collection '{collection}' is already empty[/yellow]")
            return
        
        # Confirm
        if not confirm:
            if not Confirm.ask(f"Clear {doc_count} documents from '{collection}'?"):
                console.print("[yellow]Cancelled[/yellow]")
                return
        
        # Clear
        rag_chain.clear_index(collection)
        console.print(f"[green]✅ Cleared collection '{collection}'[/green]")
    
    asyncio.run(run_clear())

@app.callback()
def main():
    """
    🤖 WonyBot - Your personal AI assistant powered by gpt-oss
    
    Now with RAG support! Index documents and ask questions:
    - wony index <path>    # Index documents
    - wony search <query>  # Search indexed documents  
    - wony ask <question>  # Ask questions using RAG
    
    Get started with 'wony setup' if this is your first time!
    """
    pass

if __name__ == "__main__":
    app()