"""Ollama client for interacting with gpt-oss model"""

import httpx
import json
from typing import AsyncGenerator, Optional, List, Dict, Any
from app.config import settings
from app.models.schemas import OllamaRequest, OllamaResponse, Message, MessageRole
import asyncio
from rich.console import Console

console = Console()

class OllamaClient:
    """Client for Ollama API"""
    
    def __init__(self):
        self.base_url = settings.ollama_host
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout
        
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate response from Ollama"""
        
        url = f"{self.base_url}/api/generate"
        
        # Build the full prompt with system message if provided
        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\nUser: {prompt}\n\nAssistant:"
        
        request_data = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": stream,
            "options": kwargs.get("options", {})
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if stream:
                async with client.stream("POST", url, json=request_data) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
            else:
                response = await client.post(url, json=request_data)
                response.raise_for_status()
                data = response.json()
                yield data.get("response", "")
    
    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Chat with Ollama using message history"""
        
        url = f"{self.base_url}/api/chat"
        
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        request_data = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": stream,
            "options": kwargs.get("options", {})
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if stream:
                async with client.stream("POST", url, json=request_data) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                            if data.get("done", False):
                                break
            else:
                response = await client.post(url, json=request_data)
                response.raise_for_status()
                data = response.json()
                if "message" in data:
                    yield data["message"].get("content", "")
    
    async def check_health(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False
                
                # Check if our model is available
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return self.model in models
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception:
            return []
    
    async def pull_model(self, model_name: str = None) -> bool:
        """Pull a model from Ollama registry"""
        model = model_name or self.model
        url = f"{self.base_url}/api/pull"
        
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                console.print(f"[yellow]Pulling model {model}...[/yellow]")
                async with client.stream("POST", url, json={"name": model}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "status" in data:
                                console.print(f"[dim]{data['status']}[/dim]")
                            if data.get("done", False):
                                console.print(f"[green]Model {model} pulled successfully![/green]")
                                return True
        except Exception as e:
            console.print(f"[red]Failed to pull model: {e}[/red]")
            return False
        
        return False