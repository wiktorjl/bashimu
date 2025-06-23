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

# --- Data Classes ---

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

@dataclass
class Persona:
    name: str
    user_identity: str
    ai_identity: str
    conversation_goal: str
    response_style: str
    system_prompt: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self):
        """Build a detailed system prompt if one isn't provided explicitly."""
        if not self.system_prompt:
            self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return f"""**Persona and Guidelines**

**Your Identity (AI):** {self.ai_identity}

**User's Identity:** {self.user_identity}

**Conversation Goal:** {self.conversation_goal}

**Required Response Style:** {self.response_style}

---
Adhere strictly to these persona guidelines for all your responses."""

# --- Core Managers ---

class PersonaManager:
    def __init__(self, personas_dir: str = None):
        self.personas_dir = personas_dir or os.path.expanduser("~/.config/llm-chat/personas")
        self.ensure_personas_dir()
        self.create_default_personas()

    def ensure_personas_dir(self):
        os.makedirs(self.personas_dir, exist_ok=True)

    def create_default_personas(self):
        """Create some default persona files if the directory is empty."""
        default_personas = {
            "default.json": {
                "name": "Default Assistant",
                "user_identity": "A user seeking helpful information.",
                "ai_identity": "A helpful, knowledgeable, and friendly AI assistant.",
                "conversation_goal": "To provide accurate, clear, and concise answers to the user's questions.",
                "response_style": "Friendly, approachable, and direct. Use Markdown for formatting when it improves clarity."
            },
            "coding_mentor.json": {
                "name": "Coding Mentor",
                "provider": "openai",
                "model": "gpt-4o",
                "user_identity": "A junior developer seeking to improve their code and learn best practices.",
                "ai_identity": "An expert senior software engineer with a talent for mentoring.",
                "conversation_goal": "To review code, explain complex concepts, and teach software engineering principles.",
                "response_style": "Provide detailed, technical explanations. Offer code examples and alternatives. Always explain the 'why' behind a suggestion. Be encouraging but rigorous."
            }
        }
        if not os.listdir(self.personas_dir):
            for filename, data in default_personas.items():
                filepath = os.path.join(self.personas_dir, filename)
                try:
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                except IOError as e:
                    print(f"Warning: Could not create default persona {filename}: {e}")

    def load_persona(self, filename: str) -> Optional[Persona]:
        if not filename.endswith('.json'):
            filename += '.json'
        filepath = os.path.join(self.personas_dir, filename)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return Persona(
                name=data.get('name', 'Unknown Persona'),
                user_identity=data.get('user_identity', 'A user'),
                ai_identity=data.get('ai_identity', 'An AI'),
                conversation_goal=data.get('conversation_goal', 'A conversation'),
                response_style=data.get('response_style', 'A standard response'),
                system_prompt=data.get('system_prompt', ''),
                provider=data.get('provider'),
                model=data.get('model')
            )
        except (json.JSONDecodeError, IOError, KeyError) as e:
            rprint(f"[red]Error loading persona from {filepath}: {e}[/red]")
            return None

    def list_personas(self) -> List[str]:
        try:
            return sorted([f[:-5] for f in os.listdir(self.personas_dir) if f.endswith('.json')])
        except OSError:
            return []

# --- LLM Clients (with system prompt handling) ---

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
        if not openai: raise ImportError("OpenAI library is required for this provider.")
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
        except Exception as e: return f"Error: {str(e)}"
    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            return sorted([model.id for model in models.data if 'gpt' in model.id])
        except Exception: return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

class AnthropicClient(LLMClient):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.current_model = config.default_model or "claude-3-haiku-20240307"
    def chat(self, messages: List[Message], model: str = None) -> str:
        model_to_use = model or self.current_model
        system_prompt = ""
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            chat_messages = [msg for msg in messages[1:] if msg.role in ["user", "assistant"]]
        else:
            chat_messages = [msg for msg in messages if msg.role in ["user", "assistant"]]
        try:
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            data = {"model": model_to_use, "max_tokens": 4096, "messages": [{"role": msg.role, "content": msg.content} for msg in chat_messages]}
            if system_prompt: data["system"] = system_prompt
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["content"][0]["text"]
        except Exception as e: return f"Error: {e}"
    def list_models(self) -> List[str]:
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]

class GeminiClient(LLMClient):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.current_model = config.default_model or "gemini-1.5-flash-latest"
    def chat(self, messages: List[Message], model: str = None) -> str:
        model_to_use = model or self.current_model
        system_instruction = None
        if messages and messages[0].role == "system":
            system_instruction = {"parts": [{"text": messages[0].content}]}
            chat_messages = messages[1:]
        else:
            chat_messages = messages
        contents = [{"role": "user" if msg.role == "user" else "model", "parts": [{"text": msg.content}]} for msg in chat_messages]
        try:
            url = f"{self.base_url}/{model_to_use}:generateContent"
            headers = {"Content-Type": "application/json"}
            data = {"contents": contents}
            if system_instruction: data["system_instruction"] = system_instruction
            response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            if "candidates" in result: return result["candidates"][0]["content"]["parts"][0]["text"]
            return "Error: No response generated (check safety settings in your Google AI account)"
        except Exception as e: return f"Error: {e}"
    def list_models(self) -> List[str]:
        try:
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            models = response.json().get("models", [])
            return sorted([m["name"].split("/")[-1] for m in models if "generateContent" in m.get("supportedGenerationMethods", [])])
        except Exception as e:
            rprint(f"[yellow]Warning: Could not fetch Gemini models ({e}). Using common models.[/yellow]")
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
        except Exception as e: return f"Error: {e}"
    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return sorted([model["name"] for model in response.json().get("models", [])])
        except Exception: return []

# --- Config and Session Management ---

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
            with open(self.config_path, 'r') as f: return json.load(f)
        except (json.JSONDecodeError, IOError): return self.get_default_config()
    def get_default_config(self) -> Dict[str, Any]:
        return {
            "default_provider": "openai",
            "default_persona": "default",
            "providers": {
                "openai": {"name": "OpenAI", "api_key": "YOUR_OPENAI_API_KEY", "default_model": "gpt-4o"},
                "anthropic": {"name": "Anthropic", "api_key": "YOUR_ANTHROPIC_API_KEY", "default_model": "claude-3-haiku-20240307"},
                "gemini": {"name": "Google Gemini", "api_key": "YOUR_GEMINI_API_KEY", "default_model": "gemini-1.5-flash-latest"},
                "ollama": {"name": "Ollama", "api_key": "", "base_url": "http://localhost:11434", "default_model": "llama3"}
            }
        }
    def save_config(self, config: Dict[str, Any]):
        os.makedirs(self.config_dir, exist_ok=True)
        with open(self.config_path, 'w') as f: json.dump(config, f, indent=2)
    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        p = self.config["providers"].get(provider_name)
        if not p: return None
        return ProviderConfig(name=p["name"], api_key=p.get("api_key", ""), base_url=p.get("base_url"), default_model=p.get("default_model", ""))
    def get_default_provider(self) -> str: return self.config.get("default_provider", "openai")
    def get_default_persona(self) -> str: return self.config.get("default_persona", "default")
    def list_providers(self) -> List[str]: return list(self.config["providers"].keys())

class ChatSession:
    def __init__(self, config_manager: ConfigManager, persona_manager: PersonaManager, initial_provider: str = None, initial_persona: str = None):
        self.config_manager = config_manager
        self.persona_manager = persona_manager
        self.console = Console()
        self.messages: List[Message] = []
        self.history = InMemoryHistory()
        
        self.current_provider_name = initial_provider or config_manager.get_default_provider()
        self.llm_client = self.create_client(self.current_provider_name)
        if self.llm_client is None:
            rprint(f"[red]Error: Could not initialize '{self.current_provider_name}' client. Check config: {self.config_manager.config_path}[/red]")
            sys.exit(1)
            
        self.current_persona_name = initial_persona or config_manager.get_default_persona()
        self.current_persona = self.persona_manager.load_persona(self.current_persona_name)
        if self.current_persona is None:
            rprint(f"[yellow]Warning: Could not load persona '{self.current_persona_name}'. Falling back to 'default'.[/yellow]")
            self.current_persona_name = "default"
            self.current_persona = self.persona_manager.load_persona(self.current_persona_name)
        if self.current_persona is None:
             rprint("[red]Fatal: Could not load default persona. Exiting.[/red]")
             sys.exit(1)
        self.apply_persona_settings(is_initial_load=True)
        self.set_system_prompt()

    def create_client(self, provider_name: str) -> Optional[LLMClient]:
        config = self.config_manager.get_provider_config(provider_name)
        if not config: return None
        if provider_name != "ollama" and ("YOUR_" in config.api_key or not config.api_key):
            self.console.print(f"[yellow]Warning: No API key set for {provider_name}[/yellow]")
        client_map = {"openai": OpenAIClient, "anthropic": AnthropicClient, "gemini": GeminiClient, "ollama": OllamaClient}
        try:
            return client_map[provider_name](config)
        except Exception as e:
            self.console.print(f"[red]Failed to create client for {provider_name}: {e}[/red]")
            return None
    
    def apply_persona_settings(self, is_initial_load: bool = False):
        """Applies provider and model settings from the current persona."""
        persona_provider = self.current_persona.provider
        if persona_provider and persona_provider != self.current_provider_name:
            new_client = self.create_client(persona_provider)
            if new_client:
                self.llm_client = new_client
                self.current_provider_name = persona_provider
                if not is_initial_load:
                    rprint(f"[green]Persona switched provider to [bold]{new_client.name}[/bold].[/green]")
            else:
                rprint(f"[red]Persona specified provider '{persona_provider}', but it failed to load. Staying with {self.llm_client.name}.[/red]")
        
        persona_model = self.current_persona.model
        if persona_model:
            if self.llm_client.current_model != persona_model and not is_initial_load:
                 rprint(f"[green]Persona set model to [bold]{persona_model}[/bold].[/green]")
            self.llm_client.current_model = persona_model
            
    def set_system_prompt(self):
        """Sets or replaces the system prompt as the first message."""
        if self.messages and self.messages[0].role == "system":
            self.messages.pop(0)
        if self.current_persona and self.current_persona.system_prompt:
            self.messages.insert(0, Message(role="system", content=self.current_persona.system_prompt))
            
    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
    
    def display_message(self, message: Message):
        title = "[bold blue]You[/bold blue]" if message.role == "user" else f"[bold green]{self.llm_client.name} ({self.current_persona.name})[/bold green]"
        panel = Panel(Markdown(message.content, code_theme="monokai"), title=title, border_style="blue" if message.role == "user" else "green")
        self.console.print(panel)
        self.console.print()

    def get_user_input(self) -> Optional[str]:
        bindings = KeyBindings()
        @bindings.add('c-c')
        def _(event): event.app.exit(exception=KeyboardInterrupt)
        
        # This is the corrected key binding for Alt+Enter
        @bindings.add('escape', 'enter')
        def _(event):
            event.current_buffer.insert_text('\n')
            
        prompt_text = HTML(f'<ansigray>({self.current_persona.name})</ansigray> <ansicyan>{self.llm_client.current_model}@{self.llm_client.name}</ansicyan> <ansiblue><b>&gt;</b></ansiblue> ')
        try:
            return prompt(prompt_text, multiline=False, history=self.history, key_bindings=bindings, mouse_support=True)
        except (KeyboardInterrupt, EOFError):
            return None

    def run(self):
        self.console.print("[bold cyan]LLM Chat Tool[/bold cyan]")
        self.console.print(f"Persona: [bold magenta]{self.current_persona.name}[/bold magenta]. Type [bold]/help[/bold] for commands.")
        # Updated help text to reflect the correct key combination
        self.console.print("Press [bold]Enter[/bold] to send, [bold]Alt+Enter[/bold] for a new line.")
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
        if not parts: return True
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ['/q', '/quit', '/exit']: return False
        elif cmd == '/help': self.show_help()
        elif cmd == '/clear': self.messages.clear(); self.set_system_prompt(); self.console.clear(); self.console.print("[green]Conversation cleared![/green]")
        elif cmd == '/history': self.show_history()
        elif cmd == '/save': self.save_conversation()
        elif cmd == '/edit': self.edit_last_message()
        elif cmd == '/provider': self.handle_provider_command(args)
        elif cmd == '/models': self.handle_models_command()
        elif cmd == '/model': self.handle_model_command(args)
        elif cmd == '/personas': self.handle_personas_command()
        elif cmd == '/persona': self.handle_persona_command(args)
        else: self.console.print(f"[red]Unknown command: {cmd}[/red]. Type /help.")
        return True

    def show_help(self):
        help_text = """
[bold cyan]Available Commands:[/bold cyan]
[bold]/help[/bold]                  - Show this help message
[bold]/clear[/bold]                 - Clear conversation history (keeps persona)
[bold]/history[/bold]               - Show conversation history
[bold]/save[/bold]                  - Save conversation to a JSON file
[bold]/edit[/bold]                  - Edit and resend your last message
[bold]/provider [name][/bold]       - Switch provider (e.g., /provider openai). No name lists available.
[bold]/models[/bold]                - List available models for the current provider.
[bold]/model [name][/bold]          - Switch model (e.g., /model gpt-4o).
[bold]/personas[/bold]              - List available personas.
[bold]/persona [name][/bold]        - Switch persona (e.g., /persona coding_mentor). This clears the chat.
[bold]/quit, /q, /exit[/bold]      - Exit the chat
        """
        self.console.print(Panel(help_text, title="Help"))

    def show_history(self):
        if not self.messages or all(m.role == 'system' for m in self.messages): self.console.print("[yellow]No history yet.[/yellow]"); return
        for i, msg in enumerate(self.messages, 1):
            if msg.role == 'system': continue
            role_color = "blue" if msg.role == "user" else "green"
            content = (msg.content[:100] + '...') if len(msg.content) > 100 else msg.content
            self.console.print(f"[bold {role_color}]{i}. {msg.role.capitalize()}:[/bold {role_color}] {content.replace('[', '[[').strip()}")

    def save_conversation(self):
        if not self.messages: self.console.print("[yellow]No conversation to save.[/yellow]"); return
        filename = f"chat_{self.current_provider_name}_{self.current_persona_name}_{len(self.messages)}.json"
        try:
            with open(filename, 'w') as f: json.dump([asdict(msg) for msg in self.messages], f, indent=2)
            self.console.print(f"[green]Conversation saved to {filename}[/green]")
        except Exception as e: self.console.print(f"[red]Error saving: {e}[/red]")

    def edit_last_message(self):
        last_user_idx = next((i for i in range(len(self.messages) - 1, -1, -1) if self.messages[i].role == 'user'), None)
        if last_user_idx is None: self.console.print("[yellow]No user message to edit.[/yellow]"); return
        original_content = self.messages[last_user_idx].content
        self.messages = self.messages[:last_user_idx]
        edited_input = prompt(HTML('<ansiblue><b>Edit:</b></ansiblue> '), default=original_content, multiline=False)
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
            table.add_column("Name", style="cyan"); table.add_column("Default Model", style="magenta")
            for p_name in providers:
                p_conf = self.config_manager.get_provider_config(p_name)
                table.add_row(f"{'*' if p_name == self.current_provider_name else ''}{p_name}", p_conf.default_model)
            self.console.print(table); return
        provider_name = args[0].lower()
        if provider_name not in providers: self.console.print(f"[red]Provider '{provider_name}' not found.[/red]"); return
        new_client = self.create_client(provider_name)
        if new_client:
            self.llm_client, self.current_provider_name = new_client, provider_name
            self.console.print(f"[green]Switched to provider: {self.llm_client.name} with model: {self.llm_client.current_model}[/green]")
        else: self.console.print(f"[red]Failed to switch to '{provider_name}'. Check config.[/red]")

    def handle_models_command(self):
        with self.console.status("[yellow]Fetching models...[/yellow]"): models = self.llm_client.list_models()
        if not models: self.console.print(f"[yellow]Could not fetch models for {self.llm_client.name}.[/yellow]"); return
        table = Table(title=f"Available Models for {self.llm_client.name}")
        table.add_column("Model Name", style="cyan")
        for model in models: table.add_row(f"{'*' if model == self.llm_client.current_model else ''}{model}")
        self.console.print(table)

    def handle_model_command(self, args: List[str]):
        if not args: self.console.print("[red]Usage: /model <model_name>[/red]"); return
        model_name = args[0]
        self.llm_client.current_model = model_name
        self.console.print(f"[green]Switched model to: {model_name}[/green]")
        
    def handle_personas_command(self):
        personas = self.persona_manager.list_personas()
        if not personas: self.console.print(f"[yellow]No personas found in {self.persona_manager.personas_dir}[/yellow]"); return
        table = Table(title="Available Personas"); table.add_column("Filename", style="cyan"); table.add_column("Name", style="magenta"); table.add_column("Provider", style="yellow"); table.add_column("Model", style="yellow")
        for p_filename in personas:
            p = self.persona_manager.load_persona(p_filename)
            if p: table.add_row(
                f"{'*' if p_filename == self.current_persona_name else ''}{p_filename}", 
                p.name, p.provider or "-", p.model or "-")
        self.console.print(table)
        
    def handle_persona_command(self, args: List[str]):
        if not args: self.console.print("[red]Usage: /persona <persona_name>[/red]"); return
        persona_name = args[0].lower()
        new_persona = self.persona_manager.load_persona(persona_name)
        if not new_persona: self.console.print(f"[red]Persona '{persona_name}' not found.[/red]"); return
        
        # Immediately apply the changes
        self.messages.clear(); self.console.clear()
        self.current_persona, self.current_persona_name = new_persona, persona_name
        self.apply_persona_settings()
        self.set_system_prompt()
        self.console.print(f"[green]Conversation cleared. Switched to persona: [bold]{self.current_persona.name}[/bold][/green]")

def main():
    parser = argparse.ArgumentParser(description="Command-line LLM chat tool with multiple providers and personas.")
    parser.add_argument("--provider", help="Override the default provider from config.")
    parser.add_argument("--persona", help="Override the default persona from config.")
    parser.add_argument("--config", help="Path to a custom config file.")
    args = parser.parse_args()
    
    config_manager = ConfigManager(config_path=args.config)
    persona_manager = PersonaManager()
    
    config_str = json.dumps(config_manager.config)
    if "YOUR_" in config_str:
        rprint(f"[bold yellow]Welcome! It looks like your config is not yet set up.[/bold yellow]")
        rprint(f"Please edit the configuration file with your API keys at: [cyan]{config_manager.config_path}[/cyan]")
        rprint(f"You can also create or edit chat personas in: [cyan]{persona_manager.personas_dir}[/cyan]")
        if not confirm("Continue anyway (some providers may not work)?"): sys.exit(0)
    
    try:
        session = ChatSession(config_manager, persona_manager, initial_provider=args.provider, initial_persona=args.persona)
        session.run()
    except KeyboardInterrupt:
        rprint("\n[yellow]Chat session ended.[/yellow]")
    except Exception as e:
        rprint(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()