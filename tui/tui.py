#!/usr/bin/env python3

import os
import sys
import json
import argparse
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

try:
    import openai
except ImportError:
    print("OpenAI library not found. Please install with 'pip install openai'")
    openai = None

import requests


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = ""


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    base_url: Optional[str] = None
    default_model: str = ""


class LLMClient:
    """Base class for LLM clients"""
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
        self.current_model = config.default_model

    def chat(self, messages: List[Message], model: str = None) -> str:
        raise NotImplementedError

    def list_models(self) -> List[str]:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not openai:
            raise ImportError("OpenAI library is required for this provider.")
        self.client = openai.OpenAI(api_key=config.api_key)
        self.current_model = config.default_model or "gpt-3.5-turbo"

    def chat(self, messages: List[Message], model: str = None) -> str:
        model_to_use = model or self.current_model
        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": msg.role, "content": msg.content} for msg in messages]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            return sorted([model.id for model in models.data if 'gpt' in model.id])
        except Exception:
            return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]


class AnthropicClient(LLMClient):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.current_model = config.default_model or "claude-3-haiku-20240307"

    def chat(self, messages: List[Message], model: str = None) -> str:
        model_to_use = model or self.current_model
        anthropic_messages = [msg for msg in messages if msg.role in ["user", "assistant"]]
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = {
                "model": model_to_use,
                "max_tokens": 4096,
                "messages": [{"role": msg.role, "content": msg.content} for msg in anthropic_messages]
            }
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["content"][0]["text"]
        except Exception as e:
            return f"Error: {e}"

    def list_models(self) -> List[str]:
        return ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
        # return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]


class GeminiClient(LLMClient):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.current_model = config.default_model or "gemini-1.5-flash-latest"

    def chat(self, messages: List[Message], model: str = None) -> str:
        model_to_use = model or self.current_model
        contents = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        try:
            url = f"{self.base_url}/{model_to_use}:generateContent"
            headers = {"Content-Type": "application/json"}
            data = {"contents": contents}
            response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            if "candidates" in result:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return "Error: No response generated (check safety settings in your Google AI account)"
        except Exception as e:
            return f"Error: {e}"

    def list_models(self) -> List[str]:
        """
        Fetches the list of models from the API and filters for chat models.
        """
        try:
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            all_models = data.get("models", [])
            
            # Filter for models that support the "generateContent" method
            chat_models = [
                model["name"].split("/")[-1] for model in all_models 
                if "generateContent" in model.get("supportedGenerationMethods", [])
            ]
            return sorted(chat_models)

        except Exception as e:
            # If the API call fails, return a safe, hardcoded list
            print(f"[yellow]Warning: Could not fetch model list from Gemini API ({e}). Showing common models.[/yellow]")
            return ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]


class OllamaClient(LLMClient):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self.current_model = config.default_model or "llama3"

    def chat(self, messages: List[Message], model: str = None) -> str:
        model_to_use = model or self.current_model
        try:
            response = requests.post(f"{self.base_url}/api/chat", json={
                "model": model_to_use,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "stream": False
            })
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Error: {e}"

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return sorted([model["name"] for model in response.json().get("models", [])])
        except Exception:
            return []


class ConfigManager:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.expanduser("~/.config/llm-chat/config.json")
        self.config_dir = os.path.dirname(self.config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            config = self.get_default_config()
            self.save_config(config)
            return config
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config: {e}. Using default.")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "default_provider": "openai",
            "providers": {
                "openai": {"name": "OpenAI", "api_key": "YOUR_OPENAI_API_KEY", "default_model": "gpt-4o"},
                "anthropic": {"name": "Anthropic", "api_key": "YOUR_ANTHROPIC_API_KEY", "default_model": "claude-3-haiku-20240307"},
                "gemini": {"name": "Google Gemini", "api_key": "YOUR_GEMINI_API_KEY", "default_model": "gemini-1.5-flash-latest"},
                "ollama": {"name": "Ollama", "api_key": "", "base_url": "http://localhost:11434", "default_model": "llama3"}
            }
        }

    def save_config(self, config: Dict[str, Any]):
        os.makedirs(self.config_dir, exist_ok=True)
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except IOError as e:
            print(f"Error saving config: {e}")

    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        if provider_name not in self.config["providers"]:
            return None
        p = self.config["providers"][provider_name]
        return ProviderConfig(name=p["name"], api_key=p.get("api_key", ""), base_url=p.get("base_url"), default_model=p.get("default_model", ""))

    def get_default_provider(self) -> str:
        return self.config.get("default_provider", "openai")

    def list_providers(self) -> List[str]:
        return list(self.config["providers"].keys())


class ChatSession:
    def __init__(self, config_manager: ConfigManager, initial_provider: str = None):
        self.config_manager = config_manager
        self.console = Console()
        self.messages: List[Message] = []
        self.history = InMemoryHistory()

        self.current_provider_name = initial_provider or config_manager.get_default_provider()
        self.llm_client = self.create_client(self.current_provider_name)
        
        if self.llm_client is None:
            self.console.print(f"[red]Error: Could not initialize '{self.current_provider_name}' client. Check your config at {self.config_manager.config_path}[/red]")
            sys.exit(1)

    def create_client(self, provider_name: str) -> Optional[LLMClient]:
        config = self.config_manager.get_provider_config(provider_name)
        if not config: return None
        if provider_name != "ollama" and ("YOUR_" in config.api_key or not config.api_key):
            self.console.print(f"[yellow]Warning: No API key configured for {provider_name}[/yellow]")
            return None
        try:
            client_map = {"openai": OpenAIClient, "anthropic": AnthropicClient, "gemini": GeminiClient, "ollama": OllamaClient}
            return client_map[provider_name](config)
        except Exception as e:
            self.console.print(f"[red]Failed to create client for {provider_name}: {e}[/red]")
            return None

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def display_message(self, message: Message):
        title = "[bold blue]You[/bold blue]" if message.role == "user" else f"[bold green]{self.llm_client.name}[/bold green]"
        panel = Panel(Markdown(message.content), title=title, border_style="blue" if message.role == "user" else "green")
        self.console.print(panel)
        self.console.print()

    def get_user_input(self) -> Optional[str]:
        bindings = KeyBindings()
        @bindings.add('c-c')
        def _(event): event.app.exit(exception=KeyboardInterrupt)
        
        prompt_text = HTML(f'<ansicyan>{self.llm_client.current_model} via {self.llm_client.name}</ansicyan> <ansiblue><b>&gt;</b></ansiblue> ')
        try:
            return prompt(prompt_text, multiline=True, history=self.history, key_bindings=bindings, mouse_support=True)
        except (KeyboardInterrupt, EOFError):
            return None

    def run(self):
        self.console.print("[bold cyan]LLM Chat Tool[/bold cyan]")
        self.console.print("Type your message and press [bold]Alt+Enter[/bold] or [bold]Esc+Enter[/bold] to send.")
        self.console.print("Type [bold]/help[/bold] for commands. Use [bold]Ctrl+C[/bold] to exit.")
        self.console.print("-" * 80)
        
        while True:
            user_input = self.get_user_input()
            if user_input is None: break
            if not user_input.strip(): continue
            
            if user_input.startswith('/'):
                if not self.handle_command(user_input): break
                continue

            self.add_message("user", user_input)
            self.display_message(self.messages[-1])
            
            with self.console.status("[bold yellow]Thinking...[/bold yellow]"):
                response = self.llm_client.chat(self.messages)
            
            self.add_message("assistant", response)
            self.display_message(self.messages[-1])

    def handle_command(self, command: str) -> bool:
        parts = command.strip().split()
        cmd, args = parts[0].lower(), parts[1:]
        
        if cmd in ['/q', '/quit', '/exit']: return False
        elif cmd == '/help': self.show_help()
        elif cmd == '/clear': self.messages.clear(); self.console.clear(); self.console.print("[green]Conversation cleared![/green]")
        elif cmd == '/history': self.show_history()
        elif cmd == '/save': self.save_conversation()
        elif cmd == '/edit': self.edit_last_message()
        elif cmd == '/provider': self.handle_provider_command(args)
        elif cmd == '/models': self.handle_models_command()
        elif cmd == '/model': self.handle_model_command(args)
        else: self.console.print(f"[red]Unknown command: {cmd}[/red]. Type /help for options.")
        
        return True

    def show_help(self):
        help_text = """
[bold cyan]Available Commands:[/bold cyan]
[bold]/help[/bold]                  - Show this help message
[bold]/clear[/bold]                 - Clear conversation history
[bold]/history[/bold]               - Show conversation history
[bold]/save[/bold]                  - Save conversation to a JSON file
[bold]/edit[/bold]                  - Edit and resend your last message
[bold]/provider [name][/bold]       - Switch provider (e.g., /provider openai). No name lists providers.
[bold]/models[/bold]                - List available models for the current provider.
[bold]/model [name][/bold]          - Switch model (e.g., /model gpt-4o).
[bold]/quit, /q, /exit[/bold]      - Exit the chat
        """
        self.console.print(Panel(help_text, title="Help"))

    def show_history(self):
        if not self.messages: self.console.print("[yellow]No history yet.[/yellow]"); return
        for i, msg in enumerate(self.messages, 1):
            role_color = "blue" if msg.role == "user" else "green"
            content = (msg.content[:100] + '...') if len(msg.content) > 100 else msg.content
            self.console.print(f"[bold {role_color}]{i}. {msg.role.capitalize()}:[/bold {role_color}] {content.replace('[', '[[').strip()}")

    def save_conversation(self):
        if not self.messages: self.console.print("[yellow]No conversation to save.[/yellow]"); return
        filename = f"chat_{self.current_provider_name}_{len(self.messages)}.json"
        try:
            with open(filename, 'w') as f: json.dump([asdict(msg) for msg in self.messages], f, indent=2)
            self.console.print(f"[green]Conversation saved to {filename}[/green]")
        except Exception as e: self.console.print(f"[red]Error saving: {e}[/red]")

    def edit_last_message(self):
        last_user_idx = next((i for i in range(len(self.messages) - 1, -1, -1) if self.messages[i].role == 'user'), None)
        if last_user_idx is None: self.console.print("[yellow]No user message to edit.[/yellow]"); return
        
        original_content = self.messages[last_user_idx].content
        self.messages = self.messages[:last_user_idx]
        
        edited_input = prompt(HTML('<ansiblue><b>Edit:</b></ansiblue> '), default=original_content, multiline=True)
        if edited_input and edited_input.strip():
            self.add_message("user", edited_input.strip())
            self.display_message(self.messages[-1])
            with self.console.status("[bold yellow]Thinking...[/bold yellow]"):
                response = self.llm_client.chat(self.messages)
            self.add_message("assistant", response)
            self.display_message(self.messages[-1])
        else:
            self.console.print("[yellow]Edit cancelled.[/yellow]")

    def handle_provider_command(self, args: List[str]):
        providers = self.config_manager.list_providers()
        if not args:
            table = Table(title="Available Providers")
            table.add_column("Name", style="cyan")
            table.add_column("Default Model", style="magenta")
            for p_name in providers:
                p_conf = self.config_manager.get_provider_config(p_name)
                is_active = "*" if p_name == self.current_provider_name else ""
                table.add_row(f"{is_active}{p_name}", p_conf.default_model)
            self.console.print(table)
            return

        provider_name = args[0].lower()
        if provider_name not in providers:
            self.console.print(f"[red]Provider '{provider_name}' not found.[/red]"); return
        
        new_client = self.create_client(provider_name)
        if new_client:
            self.llm_client = new_client
            self.current_provider_name = provider_name
            self.console.print(f"[green]Switched to provider: {self.llm_client.name}[/green]")
            self.console.print(f"[green]Current model: {self.llm_client.current_model}[/green]")
        else:
            self.console.print(f"[red]Failed to switch to '{provider_name}'. Check config.[/red]")

    def handle_models_command(self):
        with self.console.status("[yellow]Fetching models...[/yellow]"):
            models = self.llm_client.list_models()
        if not models:
            self.console.print(f"[yellow]Could not fetch models for {self.llm_client.name}.[/yellow]"); return
        
        table = Table(title=f"Available Models for {self.llm_client.name}")
        table.add_column("Model Name", style="cyan")
        for model in models:
            is_active = "*" if model == self.llm_client.current_model else ""
            table.add_row(f"{is_active}{model}")
        self.console.print(table)

    def handle_model_command(self, args: List[str]):
        if not args:
            self.console.print("[red]Usage: /model <model_name>[/red]"); return
        
        model_name = args[0]
        self.llm_client.current_model = model_name
        self.console.print(f"[green]Switched model to: {model_name}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Command-line LLM chat tool with multiple providers.")
    parser.add_argument("--provider", help="Override the default provider from config.")
    parser.add_argument("--config", help="Path to a custom config file.")
    args = parser.parse_args()

    config_manager = ConfigManager(config_path=args.config)
    
    config_str = json.dumps(config_manager.config)
    if "YOUR_" in config_str:
        rprint(f"[bold yellow]Welcome! It looks like your config is not yet set up.[/bold yellow]")
        rprint(f"Please edit the configuration file with your API keys at: [cyan]{config_manager.config_path}[/cyan]")
        if not confirm("Continue anyway (some providers may not work)?"):
            sys.exit(0)

    try:
        session = ChatSession(config_manager, initial_provider=args.provider)
        session.run()
    except KeyboardInterrupt:
        rprint("\n[yellow]Chat session ended.[/yellow]")
    except Exception as e:
        rprint(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()