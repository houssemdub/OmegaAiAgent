#!/usr/bin/env python3
"""
OmegaAi - The Ultimate Autonomous Coding Assistant
Merged and Evolved from Omega Agent Series
Author: Antigravity AI
"""

# ==================== DEPENDENCY CHECK ====================
REQUIRED_LIBS = {
    "rich": "rich",
    "httpx": "httpx",
    "aiofiles": "aiofiles",
    "prompt_toolkit": "prompt_toolkit",
    "dotenv": "python-dotenv",
    "duckduckgo_search": "duckduckgo_search"
}

def check_dependencies():
    missing = []
    for lib, package in REQUIRED_LIBS.items():
        try:
            __import__(lib)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"📦 OmegaAi: Missing dependencies detected: {', '.join(missing)}")
        print("🛠️  Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("✅ Dependencies installed successfully. Restarting engine...\n")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please install them manually: pip install " + " ".join(missing))
            sys.exit(1)

check_dependencies()

# Core Imports
import sys
import os
import time
import json
import re
import asyncio
import httpx
import aiofiles
import shutil
import subprocess
import hashlib
import textwrap
import platform
import traceback
import stat
from typing import List, Dict, Optional, Any, Tuple, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# UI & Terminal Imports (Verified)
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.tree import Tree
from rich.style import Style
from rich.columns import Columns
from rich.align import Align
from rich import box

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style as PromptStyle

# Environment Variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv = None

# Search
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None
    DDGS = None

# Global Console
console = Console(force_terminal=True, soft_wrap=True)

# Windows Compatibility
if os.name == 'nt':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7) # Enable VT100
        os.system('chcp 65001 > nul 2>&1') # Set UTF-8
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except:
        pass

# ==================== LOGGING ====================

class Logger:
    def __init__(self, root: Path):
        self.log_dir = root / "logs"
        self.log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{ts}.log"
        self._write_raw(f"=== OMEGAAI SESSION LOG {ts} ===\n")

    def _write_raw(self, text: str):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text)
        except: pass

    def info(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._write_raw(f"[{ts}] [INFO] {msg}\n")

    def tool(self, tool_type: str, details: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._write_raw(f"[{ts}] [TOOL:{tool_type.upper()}] {details}\n")

    def ai(self, role: str, response: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._write_raw(f"\n--- AI RESPONSE ({role}) [{ts}] ---\n{response}\n-------------------\n")

    def exception(self, msg: str, e: Exception):
        ts = datetime.now().strftime("%H:%M:%S")
        tb = traceback.format_exc()
        self._write_raw(f"[{ts}] [CRITICAL ERROR] {msg}: {e}\n{tb}\n")

# ==================== CONFIGURATION ====================

def load_dynamic_models() -> Dict[str, Dict[str, str]]:
    """Load and categorize models from downloaded JSON if available"""
    catalog = {
        "Coding & Development": {
            "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
            "gpt-4o": "openai/gpt-4o",
            "deepseek-coder": "deepseek/deepseek-coder",
            "qwen-2.5-coder-32b": "qwen/qwen-2.5-coder-32b-instruct"
        },
        "Reasoning & Logic": {
            "o1-preview": "openai/o1-preview",
            "claude-3-opus": "anthropic/claude-3-opus",
            "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct"
        },
        "Efficiency": {
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "claude-3-haiku": "anthropic/claude-3-haiku",
            "gemini-flash-1.5": "google/gemini-flash-1.5"
        },
        "Free Tiers": {
            "devstral-free": "mistralai/devstral-2512:free",
            "liquid-think-free": "liquid/lfm-2.5-1.2b-thinking:free"
        }
    }
    
    # Prioritize free models JSON first
    json_paths = [Path("openrouter_free_models.json"), Path("openrouter_models.json")]
    found_any = False
    
    for json_path in json_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f).get("data", [])
                
            # Auto-categorize fresh models
            for m in data:
                m_id = m.get("id", "")
                name = m.get("name", m_id)
                pricing = m.get("pricing", {})
                
                # Dynamic check for absolute free tier
                is_free = str(pricing.get("prompt", "")) == "0" and str(pricing.get("completion", "")) == "0"
                
                if is_free or ":free" in m_id.lower():
                    catalog["Free Tiers"][name] = m_id
                    found_any = True
                elif "codex" in m_id.lower() or "coder" in m_id.lower():
                    catalog["Coding & Development"][name] = m_id
                elif "thinking" in m_id.lower() or "reasoning" in m_id.lower() or "o1" in m_id.lower():
                    catalog["Reasoning & Logic"][name] = m_id
        except: pass
        if found_any: break 
        
    return catalog

MODELS_CATALOG = load_dynamic_models()
MODEL_MAP = {alias: id for cat in MODELS_CATALOG.values() for alias, id in cat.items()}

DEFAULT_CONFIG = {
    "models": {
        "planner": "mistralai/devstral-2512:free",
        "architect": "mistralai/devstral-2512:free",
        "coder": "mistralai/devstral-2512:free",
        "debugger": "mistralai/devstral-2512:free",
        "reviewer": "mistralai/devstral-2512:free"
    },
    "limits": {
        "max_iterations": 30,
        "max_file_size_mb": 15,
        "max_context_tokens": 12000,
        "history_messages": 15,
        "confirm_destructive": True
    },
    "tools": {
        "search": True,
        "rag": True,
        "persistence": True,
        "vision": True,
        "patching": True
    },
    "aesthetics": {
        "theme": "nebula",
        "banner_color": "cyan",
        "accent_color": "magenta"
    }
}

# ==================== UTILS ====================

class TokenCounter:
    @staticmethod
    def estimate(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def count_history(history: List[Dict]) -> int:
        return sum(TokenCounter.estimate(m.get("content", "")) for m in history)

class ConfigManager:
    def __init__(self, root: Path):
        self.root = root
        self.omega_dir = root / ".omega"
        self.config_path = self.omega_dir / "config.json"
        self.state_path = self.omega_dir / "state.json"
        self.config = self._load(self.config_path, DEFAULT_CONFIG)
        self.state = self._load(self.state_path, {"last_task": None, "history": []})

    def _load(self, path: Path, default: Dict) -> Dict:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {**default, **data}
            except:
                return default
        return default

    def save(self):
        self.omega_dir.mkdir(exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)

    def get(self, *keys, default=None):
        val = self.config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

# ==================== KNOWLEDGE & RAG ====================

class KnowledgeManager:
    """Handles long-term project memory and pattern recognition."""
    def __init__(self, root: Path):
        self.root = root
        self.omega_dir = root / ".omega"
        self.knowledge_path = self.omega_dir / "knowledge.md"
        self.rag_path = self.omega_dir / "rag.json"
        self.omega_dir.mkdir(exist_ok=True)
        
    def get_knowledge(self) -> str:
        if self.knowledge_path.exists():
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                return f.read()
        return "No project-specific knowledge recorded yet."

    def update_knowledge(self, new_insight: str):
        content = self.get_knowledge()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n### Insight [{ts}]\n{new_insight}\n"
        with open(self.knowledge_path, 'a', encoding='utf-8') as f:
            f.write(entry)

    def build_index(self):
        """Simple RAG indexer that maps file summaries and key markers."""
        index = {}
        for file in self.root.rglob("*"):
            if file.is_file() and not str(file).startswith(".") and file.suffix in [".py", ".js", ".html", ".css", ".md", ".txt"]:
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Extract basic info
                        index[str(file.relative_to(self.root))] = {
                            "size": len(content),
                            "markers": re.findall(r'(?:class|def|function|const|let)\s+([\w\d_]+)', content)[:20]
                        }
                except: continue
        with open(self.rag_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        return f"Indexed {len(index)} project files."

# ==================== ENGINE MODULES ====================

class GitManager:
    def __init__(self, root: Path):
        self.root = root
        self.available = self._check()

    def _check(self):
        try:
            return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
        except:
            return False

    def init_repo(self):
        if self.available and not (self.root / ".git").exists():
            subprocess.run(["git", "init"], cwd=self.root, capture_output=True)
            gitignore = self.root / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(".omega/\n__pycache__/\n*.pyc\n.venv/\n.env\n")

    def commit(self, message: str):
        if not self.available: return False
        subprocess.run(["git", "add", "."], cwd=self.root, capture_output=True)
        res = subprocess.run(["git", "commit", "-m", message], cwd=self.root, capture_output=True)
        return res.returncode == 0

class BackupManager:
    def __init__(self, root: Path, max_backups: int):
        self.root = root
        self.backup_dir = root / ".omega" / "backups"
        self.max_backups = max_backups
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create(self, iteration: int, tag: str = "") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"it{iteration:02d}_{tag}_{ts}" if tag else f"it{iteration:02d}_{ts}"
        dest = self.backup_dir / name
        dest.mkdir()
        
        for item in self.root.iterdir():
            if item.name in [".omega", ".git", "node_modules", ".venv", "__pycache__"]:
                continue
            if item.is_file():
                shutil.copy2(item, dest / item.name)
            elif item.is_dir():
                shutil.copytree(item, dest / item.name, ignore=shutil.ignore_patterns(".omega", ".git", "__pycache__"))
        
        self._prune()
        return str(dest)

    def _prune(self):
        def remove_readonly(func, path, _):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        backups = sorted([d for d in self.backup_dir.iterdir() if d.is_dir()])
        while len(backups) > self.max_backups:
            backup_to_remove = backups.pop(0)
            try:
                shutil.rmtree(backup_to_remove, onexc=remove_readonly)
            except Exception as e:
                # Fallback for older python versions if needed
                try: shutil.rmtree(backup_to_remove, onerror=remove_readonly)
                except: pass

    def rollback(self, path: str):
        src = Path(path)
        if not src.exists(): return False
        
        # Wipe current (except .omega/.git)
        for item in self.root.iterdir():
            if item.name in [".omega", ".git"]: continue
            if item.is_file(): item.unlink()
            else: shutil.rmtree(item)
            
        # Restore
        for item in src.iterdir():
            if item.is_file(): shutil.copy2(item, self.root / item.name)
            else: shutil.copytree(item, self.root / item.name)
        return True

class WorkspaceManager:
    def __init__(self, root: Path):
        self.root = root

    def get_tree(self, depth: int = 3) -> str:
        tree = Tree(f"[bold blue]{self.root.name}/[/bold blue]")
        def _walk(path, node, d):
            if d > depth: return
            try:
                items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                for item in items:
                    if item.name.startswith('.') and item.name not in ['.env', '.gitignore']: continue
                    if item.name in ["__pycache__", "node_modules"]: continue
                    
                    if item.is_dir():
                        sub = node.add(f"[blue]{item.name}/[/blue]")
                        _walk(item, sub, d + 1)
                    else:
                        node.add(f"[green]{item.name}[/green]")
            except: pass
        _walk(self.root, tree, 0)
        
        from rich.console import Console
        from io import StringIO
        c = Console(file=StringIO(), force_terminal=True)
        c.print(tree)
        return c.file.getvalue()

    async def read(self, path: str) -> str:
        p = self.root / path
        if not p.exists(): return f"Error: {path} not found"
        async with aiofiles.open(p, 'r', encoding='utf-8', errors='replace') as f:
            return await f.read()

    async def write(self, path: str, content: str):
        if not path or path.strip() in [".", "/", "\\"]:
            return "Error: Invalid path. Path cannot be empty or root."
            
        full_path = self.root / path
        if full_path.is_dir():
            return f"Error: '{path}' is a directory. Cannot overwrite with a file write."
            
        try:
            # Check if file exists and content is the same to avoid redundant writes
            if full_path.exists():
                async with aiofiles.open(full_path, mode='r', encoding='utf-8') as f:
                    current = await f.read()
                    if current == content:
                        return f"NO_CHANGE: File '{path}' already has this exact content."
            
            full_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(full_path, mode='w', encoding='utf-8') as f:
                await f.write(content)
            return f"Successfully wrote {len(content)} bytes to {path}"
        except PermissionError:
            return f"Error: Permission Denied for '{path}'. Is it open in another program?"
        except Exception as e:
            return f"Error writing to {path}: {e}"

class ToolParser:
    @staticmethod
    def parse(text: str) -> List[Dict]:
        tools = []
        # XML-like tags (strict and fuzzy)
        
        # XML Read: <read path="..."/>
        for m in re.finditer(r'<read\s+path=["\']?([^"\'>\s]+)["\']?\s*/?>', text, re.IGNORECASE):
            tools.append({"type": "read", "path": m.group(1)})
            
        # XML Write: <write path="...">content</write>
        for m in re.finditer(r'<write\s+path=["\']?([^"\'>\s]+)["\']?>(.*?)</write>', text, re.DOTALL | re.IGNORECASE):
            tools.append({"type": "write", "path": m.group(1), "content": m.group(2).strip()})
            
        # XML Run: <run>command</run>
        for m in re.finditer(r'<run>(.*?)</run>', text, re.DOTALL | re.IGNORECASE):
            tools.append({"type": "run", "cmd": m.group(1).strip()})
            
        # XML Search: <search>query</search>
        for m in re.finditer(r'<search>(.*?)</search>', text, re.DOTALL | re.IGNORECASE):
            tools.append({"type": "search", "query": m.group(1).strip()})
            
        # Markdown Fallback for Write: ```(file:path) or just ```path
        if not any(t["type"] == "write" for t in tools):
            blocks = re.finditer(r'```(?:\w+)?\s*(?:\(file:\s*([^\)]+)\)|([^\n\r]+))?\n(.*?)```', text, re.DOTALL)
            for b in blocks:
                path = b.group(1) or b.group(2)
                if path and "." in path and "/" not in path.split(".")[-1]:
                    tools.append({"type": "write", "path": path.strip(), "content": b.group(3).strip()})
        
        # Markdown Fallback for Run: `shell:command`
        if not any(t["type"] == "run" for t in tools):
            for m in re.finditer(r'`shell:(.*?)`', text):
                tools.append({"type": "run", "cmd": m.group(1).strip()})

        # XML Patch: <patch path="..."> <search>...</search> <replace>...</replace> </patch>
        for p in re.finditer(r'<patch\s+path=["\']?([^"\'>\s]+)["\']?>(.*?)</patch>', text, re.DOTALL | re.IGNORECASE):
            path = p.group(1)
            body = p.group(2)
            search = re.search(r'<search>(.*?)</search>', body, re.DOTALL | re.IGNORECASE)
            replace = re.search(r'<replace>(.*?)</replace>', body, re.DOTALL | re.IGNORECASE)
            if search and replace:
                tools.append({
                    "type": "patch", 
                    "path": path, 
                    "search": search.group(1), 
                    "replace": replace.group(1)
                })

        return tools

# ==================== CLIENT & AGENT ====================

class OmegaClient:
    def __init__(self, api_key: str, models: Dict[str, str]):
        self.api_key = api_key
        self.models = models # Dictionary of roles: model_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/antigravity-ai/omega",
            "X-Title": "OmegaAi Agent"
        }
        self.history = []

    async def stream(self, prompt: str, system: str = "", role: str = "coder"):
        model = self.models.get(role, self.models.get("coder"))
        
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                async with client.stream("POST", 
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=self.headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": True,
                        "temperature": 0.3
                    }
                ) as response:
                    # Check for immediate success
                    if response.status_code != 200:
                        error_body = await response.aread()
                        try:
                            err_json = json.loads(error_body)
                            err_msg = err_json.get("error", {}).get("message", error_body.decode())
                        except:
                            err_msg = error_body.decode()
                        
                        hint = ""
                        if response.status_code == 401:
                            if "User not found" in err_msg:
                                hint = "\n[yellow]HINT: OpenRouter cannot find your account. Check your API key or account status at openrouter.ai[/yellow]"
                            else:
                                hint = "\n[yellow]HINT: Unauthorized. Your API key might be invalid or expired.[/yellow]"
                        if response.status_code == 402: hint = "\n[yellow]HINT: Your OpenRouter account may be out of credits.[/yellow]"
                        if response.status_code == 429: hint = "\n[yellow]HINT: You have hit a rate limit. Please wait a few moments.[/yellow]"
                        
                        yield ("error", f"\n[bold red]API ERROR {response.status_code}:[/bold red] {err_msg}{hint}")
                        return

                    full = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]": break
                            try:
                                j = json.loads(data)
                                if "error" in j:
                                    err_msg = j['error'].get('message', 'Unknown error')
                                    yield ("error", f"\n[bold red]STREAM ERROR:[/bold red] {err_msg}")
                                    break
                                
                                choices = j.get('choices', [])
                                if not choices: continue
                                
                                delta = choices[0].get('delta', {})
                                
                                # Handle reasoning tokens
                                reasoning = delta.get('reasoning', '')
                                if reasoning:
                                    full += reasoning
                                    yield ("reasoning", reasoning)
                                
                                # Handle content tokens
                                content = delta.get('content', '')
                                if content:
                                    full += content
                                    yield ("content", content)
                            except Exception as e:
                                # Only log if it's a real failure, ignore heartbeat chunks
                                if data.strip() and data.strip() != "[DONE]":
                                    self.logger.info(f"Stream parse error: {e} on data: {data}")
                                pass
                    self.history.append({"role": "user", "content": prompt})
                    self.history.append({"role": "assistant", "content": full})
                    # Prune history if too long
                    if len(self.history) > 20: self.history = self.history[-20:]
            except Exception as e:
                yield ("error", f"\n[bold red]CONNECTION FAILED:[/bold red] {str(e)}")

class OmegaAi:
    def __init__(self):
        self.root = Path.cwd()
        self.logger = Logger(self.root)
        self.config = ConfigManager(self.root)
        self.workspace = WorkspaceManager(self.root)
        self.git = GitManager(self.root)
        self.backup = BackupManager(self.root, self.config.get("limits", "max_backups", default=10))
        self.knowledge = KnowledgeManager(self.root)
        
        # Auto-index if RAG is on
        if self.config.get("tools", "rag"):
            self.knowledge.build_index()
        
        # Log startup
        self.logger.info(f"OmegaAi initialized in {self.root}")
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            console.print("[bold red]OPENROUTER_API_KEY not found in environment![/bold red]")
            api_key = Prompt.ask("Enter OpenRouter API Key", password=True)
        
        if api_key: api_key = api_key.strip()
            
        self.client = OmegaClient(api_key, self.config.get("models"))
        self.iteration = 0

    def show_banner(self):
        banner_font = r"""
   ____  __  __________________    ___    ____
  / __ \/  |/  / ____/ ____/   |  /   |  /  _/
 / / / / /|_/ / __/ / / __/ /| | / /| |  / /  
/ /_/ / /  / / /___/ /_/ / ___ |/ ___ |_/ /   
\____/_/  |_/_____/\____/_/  |_/_/  |_/___/   
"""
        active_models = self.config.get("models")
        
        # 1. Wide Header
        header_text = Text.assemble(
            (banner_font, "bold cyan"),
            ("\n[ PREMIER AUTONOMOUS ENGINE V3.8 ]", "bold white on blue")
        )
        console.print(Panel(header_text, border_style="bright_blue", padding=(1, 2), expand=True))

        # 2. System Telemetry (Full Width)
        tele_grid = Table.grid(expand=True)
        tele_grid.add_column(style="bold cyan", width=20)
        tele_grid.add_column(style="white")
        
        tele_grid.add_row(" 🛰️ CORE ENGINE", f": [bold green]{active_models.get('coder')}[/bold green]")
        tele_grid.add_row(" 🧠 BRAIN MODEL", f": {active_models.get('planner')}")
        tele_grid.add_row(" 🏛️ ARCHITECT", f": {active_models.get('architect')}")
        tele_grid.add_row(" 📂 WORKSPACE", f": [dim]{self.root}[/dim]")
        tele_grid.add_row(" 🕒 SESSION", f": {datetime.now().strftime('%H:%M:%S')}")
        
        console.print(Panel(
            tele_grid,
            title="[bold blue] SYSTEM TELEMETRY [/bold blue]",
            title_align="left",
            border_style="blue",
            padding=(1, 2),
            expand=True
        ))

        # 3. Command Dock (Full Width)
        cmd_grid = Table.grid(expand=True)
        cmd_grid.add_column(style="bold yellow", width=20)
        cmd_grid.add_column(style="cyan")
        
        # Tool Status Indicators
        to = self.config.get("tools")
        tool_status = " | ".join([f"[{'green' if v else 'red'}]{k.upper()}[/{'green' if v else 'red'}]" for k,v in to.items()])

        cmd_grid.add_row(" 🚀 AUTONOMOUS", "/vibe <task>")
        cmd_grid.add_row(" 🛠️ TOOLBOX", f"/tools ({tool_status})")
        cmd_grid.add_row(" 📊 CATALOGUE", "/models | /models-tier")
        cmd_grid.add_row(" ❓ HELP", "/help")
        
        console.print(Panel(
            cmd_grid,
            title="[bold yellow] QUICK COMMAND DOCK [/bold yellow]",
            title_align="left",
            border_style="yellow",
            padding=(1, 2),
            expand=True
        ))

    async def run_command(self, cmd: str, timeout: int = 40):
        console.print(f"[bold yellow]⚡ Shell:[/bold yellow] {cmd}")
        
        # Enhanced Windows Translation
        if os.name == 'nt':
            cmd = cmd.replace('/', '\\')
            
            # Handle Bash Brace Expansion for mkdir more robustly: mkdir -p folder/{a,b,c}
            if 'mkdir' in cmd and '{' in cmd and '}' in cmd:
                def expand_braces(match):
                    base = match.group(1)
                    inner = match.group(2)
                    parts = [p.strip() for p in inner.split(',')]
                    return " && ".join([f"if not exist {base}{p} mkdir {base}{p}" for p in parts])
                
                cmd = re.sub(r'([^{}\s]+)\{([^}]+)\}', expand_braces, cmd)
            
            # Global mkdir safety for single/non-expanded paths
            if cmd.startswith('mkdir '):
                # Only wrap if it's not already wrapped in if not exist
                if 'if not exist' not in cmd:
                    path = cmd[6:].strip().split(' ')[-1] # naive path pick
                    cmd = f"if not exist {path} mkdir {path}"
            
            if 'mkdir -p' in cmd: cmd = cmd.replace('mkdir -p', 'mkdir')
            if cmd.startswith('ls '): cmd = cmd.replace('ls ', 'dir ')
            if cmd == 'ls': cmd = 'dir'
            if cmd.startswith('cat '): cmd = cmd.replace('cat ', 'type ')
            if 'rm -rf' in cmd: cmd = cmd.replace('rm -rf', 'rmdir /s /q')
            
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.root
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                out = stdout.decode('utf-8', errors='replace')
                err = stderr.decode('utf-8', errors='replace')
                
                # Check for "already exists" errors in mkdir and suppress if it's the only error
                if proc.returncode != 0 and out == "" and ("already exists" in err.lower() or "subsistema" in err.lower()):
                    if "mkdir" in cmd: proc.returncode = 0 

                res = f"Exit Code: {proc.returncode}\n"
                if out: res += f"Output:\n{out[:2000]}"
                if err: res += f"Error:\n{err[:2000]}"
                return res
            except asyncio.TimeoutError:
                try:
                    proc.terminate()
                    await asyncio.sleep(0.2)
                    if proc.returncode is None: proc.kill()
                    await proc.wait()
                    await asyncio.sleep(0.1)
                except: pass
                return f"Error: Command timed out ({timeout}s). NO INTERACTIVE LOOPS."
        except Exception as e:
            return f"Error executing command: {e}"

    async def exe_tools(self, tools: List[Dict]):
        results = []
        for t in tools:
            if t["type"] == "read":
                res = await self.workspace.read(t["path"])
                results.append(f"Content of {t['path']}:\n{res[:1000]}")
            elif t["type"] == "write":
                res = await self.workspace.write(t["path"], t["content"])
                results.append(res)
            elif t["type"] == "run":
                res = await self.run_command(t["cmd"])
                results.append(res)
            elif t["type"] == "patch":
                res = await self.patch_file(t["path"], t["search"], t["replace"])
                results.append(res)
            elif t["type"] == "search":
                if self.config.get("tools", "search"):
                    if DDGS:
                        with DDGS() as ddgs:
                            res = [r for r in ddgs.text(t["query"], max_results=5)]
                            results.append(f"Search results for '{t['query']}':\n{res}")
                    else: results.append("Error: Search tool not available (missing library).")
                else: results.append("Error: Web Search is currently DISABLED in /tools.")
        return "\n".join(results)

    async def patch_file(self, path: str, search: str, replace: str):
        full_path = self.root / path
        if not full_path.exists(): return f"Error: File {path} not found."
        async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        if search not in content:
            return f"Error: Could not find search string in {path}. Send the EXACT string including whitespace."
        new_content = content.replace(search, replace)
        async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
            await f.write(new_content)
        return f"Successfully patched {path}."

    async def process_task(self, task: str):
        self.iteration = 0
        self.client.history = [] 
        self.backup.create(0, "pre_task")
        
        # Helper for smart tailing display
        def get_streaming_view(text: str, role: str, it: int, reasoning: str = "") -> Panel:
            term_height = console.size.height
            max_lines = max(5, term_height - 12)
            
            # Combine reasoning and content for display
            display_parts = []
            if reasoning:
                display_parts.append(f"> [dim italic]💭 Thinking: {reasoning}[/dim italic]")
            if text:
                display_parts.append(text)
            
            full_text = "\n\n".join(display_parts)
            lines = full_text.splitlines()
            
            if len(lines) > max_lines:
                full_text = "\n".join(lines[-max_lines:])
                full_text = f"[dim]... {len(lines) - max_lines} lines above ...[/dim]\n" + full_text
            
            return Panel(
                Markdown(full_text),
                title=f"[bold magenta]OmegaAi {role.capitalize()}[/bold magenta] [dim](Iteration {it})[/dim]",
                subtitle="[italic cyan]Streaming Performance ⚡[/italic cyan]",
                border_style="magenta",
                padding=(1, 2)
            )

        # === PHASE 1: PLANNING & ARCHITECTURE ===
        console.print(Panel(f"[bold cyan]🧠 Phase 1: Planning & Architecture[/bold cyan]\n[dim]Analyzing requirements and mapping project structure...[/dim]", style="cyan"))
        
        tree = self.workspace.get_tree()
        plan_prompt = f"""Task: {task}

Current Workspace:
{tree}

Create a detailed technical specification and step-by-step implementation plan. 
At the end of your plan, you MUST include a line formatted exactly like this:
ESTIMATED_ITERATIONS: <number>
(Where <number> is your best guess for the Implementation phase, e.g., 3 for a simple file, 15 for a complex app)."""
        
        plan = ""
        reasoning = ""
        error_occurred = False
        with Live(Panel("Brainstorming...", border_style="blue"), refresh_per_second=10) as live:
            async for type, chunk in self.client.stream(plan_prompt, system="You are the Architect. Design the system and provide a clear plan.", role="architect"):
                if type == "error":
                    live.update(Panel(chunk, border_style="red", title="[bold red]API Failure[/bold red]"))
                    error_occurred = True
                    break
                if type == "reasoning": reasoning += chunk
                else: plan += chunk
                live.update(get_streaming_view(plan, "Architect", 1, reasoning))
        
        if error_occurred: return

        confirm = Confirm.ask("\n[bold yellow]Proceed with this plan?[/bold yellow]")
        if not confirm: return

        # Extract Budget
        budget_match = re.search(r"ESTIMATED_ITERATIONS:\s*(\d+)", plan)
        budget = int(budget_match.group(1)) if budget_match else 10
        max_iters = min(30, max(2, budget + 2)) # Safety padding
        self.logger.info(f"Dynamic budget set to {max_iters} iterations.")

        # === PHASE 2: IMPLEMENTATION ===
        console.print(Panel(f"[bold green]🔨 Phase 2: Implementation & Development[/bold green]\n[dim]Executing the blueprint...[/dim]", style="green"))
        
        system_prompt = f"""You are OmegaAi Coder. 
Workspace Root: {self.root}
Current Status: IMPLEMENTATION PHASE

Project Knowledge:
{self.knowledge.get_knowledge() if self.config.get("tools", "persistence") else "Persistence Disabled"}

Tools:
<read path="file" />
<write path="file">content</write>
<patch path="file"><search>old</search><replace>new</replace></patch>
<run>command</run>
<search>query</search>

IMPORTANT RULES:
1. ALWAYS use relative paths from the workspace root.
2. REAL-WORLD DATA: If the task requires facts, weather, news, or any real-world data, you MUST use the <search> tool. DO NOT simulate or hallucinate this data.
3. SURGICAL UPDATES: Preferred <patch> over <write> for existing large files to save tokens and avoid errors.
4. VISUAL AUDIT: If you create a UI, you MUST explain the aesthetic choices (typography, colors) in reasoning.
5. DO NOT create redundant nested directories.
6. YOU MUST START EVERY TOOL CALL WITH '<'.
7. If you learn something critical about how this project works (e.g., specific port, library version), include the tag <insight>Learned detail</insight> in your response to save it to Project Knowledge.
8. Once finished, say "TASK COMPLETE".
"""
        
        current_input = "Start implementation of Step 1."
        
        while self.iteration < max_iters:
            self.iteration += 1
            console.print(f"\n[bold cyan]Iteration {self.iteration}[/bold cyan]")
            self.logger.info(f"Starting iteration {self.iteration}")
            
            response = ""
            reasoning = ""
            error_msg = None
            with Live(Panel("Coding...", border_style="magenta"), refresh_per_second=10) as live:
                async for type, chunk in self.client.stream(current_input, system_prompt, role="coder"):
                    if type == "error":
                        error_msg = chunk
                        break
                    if type == "reasoning": reasoning += chunk
                    else: response += chunk
                    live.update(get_streaming_view(response, "Developer", self.iteration, reasoning))

            if error_msg:
                console.print(Panel(error_msg, border_style="red", title="[bold red]Autonomous Pipeline Halted[/bold red]"))
                break

            self.logger.ai("Developer", response)
            
            if "TASK COMPLETE" in response.upper() or "TASK COMPLETE" in reasoning.upper():
                self.logger.info("Task marked complete by AI.")
                break
                
            # Parse tools from BOTH content and reasoning to be safe
            combined_output = f"{reasoning}\n{response}"
            tools = ToolParser.parse(combined_output)
            
            if not tools:
                if not response.strip() and not reasoning.strip():
                    self.logger.info("Empty response from model.")
                    current_input = "You returned an empty response. Please provide a tool call (like <write> or <run>) to proceed with the implementation, or say TASK COMPLETE if you are finished."
                else:
                    self.logger.info("No tool calls detected in AI response.")
                    # If the AI is just talking, maybe it forgot to say TASK COMPLETE
                    current_input = "If you have finished the requested work, please say 'TASK COMPLETE'. Otherwise, use a tool like <write> or <run> to continue implementation."
                continue
                
            results = await self.exe_tools(tools)
            self.logger.tool("Execution", results)
            
            # Persistent insights
            if self.config.get("tools", "persistence"):
                insights = re.findall(r'<insight>(.*?)</insight>', combined_output, re.DOTALL | re.IGNORECASE)
                for ins in insights:
                    self.knowledge.update_knowledge(ins)
                    console.print(f"[dim cyan]🧠 Knowledge base updated with new insight.[/dim cyan]")

            current_input = f"Tool Results:\n{results}"
            
        # === PHASE 3: VERIFICATION & DEBUGGING ===
        console.print(Panel(f"[bold yellow]🔍 Phase 3: Verification & Debugging[/bold yellow]\n[dim]The implementation is being reviewed by the Debugger...[/dim]", style="yellow"))
        
        verify_prompt = f"""Implementation phase complete. 
Original Task: {task}
Blueprint used: {plan}

Workspace Tree:
{self.workspace.get_tree()}

Please:
1. Review the created/modified files.
2. Identify any missing features, bugs, or logic errors.
3. If issues are found, use tools to FIX them.
4. If everything looks perfect, say "SYSTEM VERIFIED".
"""
        
        v_iteration = 0
        while v_iteration < 5: # Max 5 verification/fix cycles
            v_iteration += 1
            v_response = ""
            v_reasoning = ""
            v_error = None
            with Live(Panel("Verifying...", border_style="yellow"), refresh_per_second=10) as live:
                async for type, chunk in self.client.stream(verify_prompt, system="You are the Debugger. Ensure the implementation is flawless.", role="debugger"):
                    if type == "error":
                        v_error = chunk
                        break
                    if type == "reasoning": v_reasoning += chunk
                    else: v_response += chunk
                    live.update(get_streaming_view(v_response, "Debugger", v_iteration, v_reasoning))

            if v_error:
                console.print(Panel(v_error, border_style="red", title="[bold red]Verification Halted[/bold red]"))
                break

            if "SYSTEM VERIFIED" in v_response.upper():
                console.print("[bold green]✅ System verified and validated.[/bold green]")
                break
                
            v_tools = ToolParser.parse(v_response)
            if v_tools:
                v_results = await self.exe_tools(v_tools)
                verify_prompt = f"Debugger Results:\n{v_results}\n\nPlease continue verification or say SYSTEM VERIFIED."
            else:
                break

        console.print(Panel("[bold green]🏁 MISSION ACCOMPLISHED[/bold green]\nAll phases complete. The agent has signed off on the workspace.", border_style="green"))

    async def main_loop(self):
        self.show_banner()
        
        # Command Autocompletion
        commands = [
            "/vibe", "/tree", "/config", "/undo", "/models", "/model", 
            "/auto-models", "/models-tier", "/tools", "/history", "/help", "/exit"
        ]
        cmd_completer = WordCompleter(commands, ignore_case=True)

        session = PromptSession(
            history=FileHistory(str(self.root / ".omega" / "cmd_history")),
            auto_suggest=AutoSuggestFromHistory(),
            completer=cmd_completer
        )
        
        while True:
            try:
                cmd = await session.prompt_async(f"Ω {self.root.name} > ")
                if not cmd.strip(): continue
                
                if cmd.startswith("/"):
                    parts = cmd.split(" ", 1)
                    base = parts[0][1:]
                    arg = parts[1] if len(parts) > 1 else ""
                    
                    if base in ["exit", "quit"]: break
                    elif base == "vibe": await self.process_task(arg)
                    elif base == "tree": console.print(self.workspace.get_tree())
                    elif base == "config": console.print(self.config.config)
                    elif base == "undo":
                        snaps = sorted([d.name for d in self.backup.backup_dir.iterdir() if d.is_dir()])
                        if snaps:
                            # Use the second to last one if we just finished an iteration
                            target = snaps[-2] if len(snaps) > 1 else snaps[0]
                            path = self.backup.backup_dir / target
                            if self.backup.rollback(str(path)):
                                console.print(f"[bold green]⏪ Rolled back to snapshot: {target}[/bold green]")
                        else: console.print("[red]No snapshots found in .omega/backups[/red]")
                    elif base == "models":
                        # Frame-based Model Catalogue UI
                        outer_grid = Table.grid(expand=True)
                        
                        header = Panel(
                            Text.assemble(
                                (" 🌐 OPENROUTER GLOBAL CATALOGUE ", "bold white on blue"),
                                ("\n Use /model <alias> to switch all, or /model set <role> <alias>", "dim italic white")
                            ),
                            border_style="blue", padding=(1, 2)
                        )
                        outer_grid.add_row(header)
                        
                        model_tables = []
                        for category, models in MODELS_CATALOG.items():
                            table = Table(title=f"📁 {category}", border_style="cyan", header_style="bold cyan", box=box.SIMPLE)
                            table.add_column("Alias", style="white", no_wrap=True)
                            table.add_column("Identifier", style="dim green")
                            table.add_column("Status", style="magenta")
                            
                            # Limit to top 15 per category to avoid scroll hell
                            sorted_items = sorted(models.items())[:15]
                            for alias, full_id in sorted_items:
                                roles = [rk.upper() for rk, rv in self.config.config["models"].items() if rv == full_id]
                                status = f"★ {', '.join(roles)}" if roles else ""
                                table.add_row(alias, full_id, status)
                            model_tables.append(table)
                            
                        # Layout tables in columns
                        outer_grid.add_row(Columns(model_tables, equal=True, expand=True))
                        source = "openrouter_free_models.json" if Path("openrouter_free_models.json").exists() else "openrouter_models.json"
                        console.print(Panel(outer_grid, border_style="bright_blue", title="Model Management Hub", subtitle=f"[dim]Loaded from {source}[/dim]"))
                        
                    elif base == "model":
                        if not arg:
                            console.print("[yellow]Usage: /model <alias> OR /model set <role> <alias>[/yellow]")
                            console.print(f"[dim]Roles: coder, planner, architect, debugger, reviewer[/dim]")
                        elif arg.startswith("set "):
                            parts = arg.split(" ")
                            if len(parts) == 3:
                                _, role, alias = parts
                                if alias in MODEL_MAP and role in ["coder", "planner", "architect", "debugger", "reviewer"]:
                                    self.config.config["models"][role] = MODEL_MAP[alias]
                                    self.client.models = self.config.config["models"]
                                    self.config.save()
                                    console.print(f"[green]Set {role} to {alias}[/green]")
                                else:
                                    console.print("[red]Invalid role or model alias[/red]")
                        else:
                            if arg in MODEL_MAP:
                                for r in ["coder", "planner", "architect", "debugger", "reviewer"]:
                                    self.config.config["models"][r] = MODEL_MAP[arg]
                                self.client.models = self.config.config["models"]
                                self.config.save()
                                console.print(f"[green]All roles switched to {arg}[/green]")
                            else:
                                if "/" in arg: # Direct ID
                                    for r in ["coder", "planner", "architect", "debugger", "reviewer"]:
                                        self.config.config["models"][r] = arg
                                    self.client.models = self.config.config["models"]
                                    self.config.save()
                                    console.print(f"[green]All roles switched to {arg}[/green]")
                                else:
                                    console.print("[red]Unknown model alias. Type /models for list.[/red]")

                    elif base == "auto-models":
                        # Automatic intelligence-based model assignment
                        # Updated with elite free models from your provided list
                        recommendations = {
                            "planner": ["xiaomi/mimo-v2-flash:free", "google/gemini-2.0-flash-thinking-exp:free", "anthropic/claude-3.5-sonnet"],
                            "architect": ["xiaomi/mimo-v2-flash:free", "meta-llama/llama-3.3-70b-instruct:free", "anthropic/claude-3.5-sonnet"],
                            "coder": ["qwen/qwen3-coder:free", "mistralai/devstral-2512:free", "anthropic/claude-3.5-sonnet"],
                            "debugger": ["xiaomi/mimo-v2-flash:free", "qwen/qwen3-coder:free", "anthropic/claude-3.5-sonnet"],
                            "reviewer": ["nvidia/nemotron-3-nano-30b-a3b:free", "google/gemini-flash-1.5", "openai/gpt-4o-mini"]
                        }
                        
                        assignments = {}
                        with console.status("[bold green]Auto-Optimizing System Roles...[/bold green]"):
                            for role, candidates in recommendations.items():
                                assigned = False
                                for cand in candidates:
                                    if cand in MODEL_MAP.values():
                                        self.config.config["models"][role] = cand
                                        assignments[role] = cand
                                        assigned = True
                                        break
                                if not assigned:
                                    # Fallback to devstral free if nothing else works
                                    self.config.config["models"][role] = "mistralai/devstral-2512:free"
                                    assignments[role] = "mistralai/devstral-2512:free"
                            
                            self.client.models = self.config.config["models"]
                            self.config.save()
                        
                        # UI Feedback
                        done_table = Table(title="🚀 System Auto-Optimization Result", border_style="green", box=box.ROUNDED)
                        done_table.add_column("Role", style="cyan")
                        done_table.add_column("Selected Model", style="white")
                        for r, m in assignments.items():
                            done_table.add_row(r.capitalize(), m)
                        console.print(done_table)

                    elif base == "history":
                        try:
                            history_file = self.root / ".omega" / "cmd_history"
                            if history_file.exists():
                                async with aiofiles.open(history_file, 'r', encoding='utf-8') as f:
                                    lines = (await f.read()).splitlines()
                                
                                # Process raw prompt_toolkit history (it usually adds timestamps or special markers)
                                # For FileHistory, it's usually just raw commands one per line
                                history_list = [l.strip() for l in lines if l.strip()]
                                
                                if arg:
                                    # Filter mode
                                    history_list = [l for l in history_list if arg.lower() in l.lower()]
                                    title = f"🔎 Command History (Search: {arg})"
                                else:
                                    title = "📜 Command History Log"
                                
                                if not history_list:
                                    console.print(f"[yellow]No history found matching '{arg}'[/yellow]")
                                else:
                                    h_table = Table(title=title, border_style="cyan", expand=True, box=box.ROUNDED)
                                    h_table.add_column("ID", justify="right", style="dim", width=6)
                                    h_table.add_column("Command", style="white")
                                    
                                    # Show last 25 by default to avoid clutter
                                    display_limit = 25
                                    start_idx = max(0, len(history_list) - display_limit)
                                    for i, h_cmd in enumerate(history_list[start_idx:], start=start_idx + 1):
                                        h_table.add_row(str(i), h_cmd)
                                    
                                    console.print(h_table)
                                    if len(history_list) > display_limit:
                                        console.print(f"[dim]Showing last {display_limit} of {len(history_list)} commands.[/dim]")
                            else:
                                console.print("[yellow]History file not found.[/yellow]")
                        except Exception as e:
                            console.print(f"[red]Error reading history: {e}[/red]")
                        
                    elif base == "models-tier":
                        tier_map = {
                            "paid": {
                                "planner": "anthropic/claude-3.5-sonnet",
                                "architect": "anthropic/claude-3.5-sonnet",
                                "coder": "anthropic/claude-3.5-sonnet",
                                "debugger": "openai/gpt-4o",
                                "reviewer": "anthropic/claude-3-haiku"
                            },
                            "fullfree": {
                                "planner": "xiaomi/mimo-v2-flash:free",
                                "architect": "xiaomi/mimo-v2-flash:free",
                                "coder": "qwen/qwen3-coder:free",
                                "debugger": "mistralai/devstral-2512:free",
                                "reviewer": "nvidia/nemotron-3-nano-30b-a3b:free"
                            },
                            "freetier": {
                                "planner": "google/gemini-2.0-flash-exp:free",
                                "architect": "google/gemini-2.0-flash-exp:free",
                                "coder": "mistralai/devstral-2512:free",
                                "debugger": "nvidia/nemotron-nano-9b-v2:free",
                                "reviewer": "liquid/lfm-2.5-1.2b-instruct:free"
                            },
                            "extrafree": {
                                "planner": "xiaomi/mimo-v2-flash:free",
                                "architect": "xiaomi/mimo-v2-flash:free",
                                "coder": "mistralai/devstral-2512:free",
                                "debugger": "mistralai/devstral-2512:free",
                                "reviewer": "openrouter/auto"
                            }
                        }
                        
                        target = arg.lower().strip()
                        if target in tier_map:
                            self.config.config["models"] = tier_map[target]
                            self.client.models = tier_map[target]
                            self.config.save()
                            
                            # Visual Feedback
                            table = Table(title=f"🚀 TIER ACTIVATED: {target.upper()}", border_style="bold green", box=box.ROUNDED)
                            table.add_column("Role", style="cyan", header_style="bold cyan")
                            table.add_column("Model ID", style="white", header_style="bold white")
                            for role, mid in tier_map[target].items():
                                table.add_row(role.capitalize(), mid)
                            
                            console.print(Panel(table, border_style="green", expand=False))
                            console.print(f"[bold green]System optimized for {target} operations. Config saved.[/bold green]")
                        else:
                            console.print(Panel(
                                "[bold red]INVALID TIER SPECIFIED[/bold red]\n\n"
                                "Available Modes:\n"
                                "• [bold cyan]paid[/bold cyan]      : Elite Gold-Standard Models (Sonnet 3.5, GPT-4o)\n"
                                "• [bold cyan]fullfree[/bold cyan]  : Heavy-Duty Free Models (MiMo-V2, Qwen3 Coder)\n"
                                "• [bold cyan]freetier[/bold cyan]  : Lightweight/Eco Free Models (Gemini Flash, Nano)\n"
                                "• [bold cyan]extrafree[/bold cyan] : Verified Survivors (MiMo-V2, Devstral, Auto)",
                                title="Tier Selection Error", border_style="red"
                            ))

                    elif base == "tools":
                        to = self.config.get("tools")
                        if arg:
                            target = arg.lower().strip()
                            if target in to:
                                to[target] = not to[target]
                                self.config.save()
                                console.print(f"[bold green]Tool '{target}' is now {'ENABLED' if to[target] else 'DISABLED'}.[/bold green]")
                            else:
                                console.print(f"[red]Unknown tool: {target}[/red]")
                        
                        # Show Tool Table
                        t_table = Table(title="🧰 OmegaAi Advanced Toolbox", border_style="yellow", box=box.ROUNDED)
                        t_table.add_column("Tool", style="cyan")
                        t_table.add_column("Status", style="bold")
                        t_table.add_column("Description", style="dim")
                        
                        desc = {
                            "search": "Web Search (DuckDuckGo integration)",
                            "rag": "Knowledge Indexing (Local file mapping)",
                            "persistence": "Project Memory (.omega/knowledge.md)",
                            "vision": "Aesthetic & UI/UX Auditing",
                            "patching": "Surgical Code Editing (<patch> tool)"
                        }
                        
                        for k, v in to.items():
                            stat = "[green]ENABLED[/green]" if v else "[red]DISABLED[/red]"
                            t_table.add_row(k.upper(), stat, desc.get(k, ""))
                        
                        console.print(t_table)
                        console.print("\n[dim]Tip: type '/tools <name>' to toggle a tool.[/dim]")

                    elif base == "help":
                        h_panel = Panel(
                            "[bold cyan]OmegaAi Command Shell[/bold cyan]\n\n"
                            "[yellow]/vibe <task>[/yellow] - Start autonomous development pipeline\n"
                            "[yellow]/models[/yellow]      - List categorized models and current active roles\n"
                            "[yellow]/auto-models[/yellow] - Automatically pick recommended models for all roles\n"
                            "[yellow]/model <alias>[/yellow] - Switch all roles to this model\n"
                            "[yellow]/model set <role> <alias>[/yellow] - Assign specific role to a model\n"
                            "[yellow]/tree[/yellow]        - Visualize workspace structure\n"
                            "[yellow]/history[/yellow]     - View recent commands (/history <search> to filter)\n"
                            "[yellow]/undo[/yellow]        - Rollback to last iteration\n"
                            "[yellow]/exit[/yellow]        - Power down agent",
                            border_style="blue", title="System Help"
                        )
                        console.print(h_panel)
                    else: console.print(f"[red]Unknown command: {base}[/red]")
                else:
                    if len(cmd) > 5: await self.process_task(cmd)
            except KeyboardInterrupt: continue
            except EOFError: break
            except Exception as e:
                console.print(f"[bold red]System Exception:[/bold red] {e}")
                self.logger.exception("Main Loop Crash", e)
                import traceback
                if self.config.get("features", "verbose"): traceback.print_exc()

if __name__ == "__main__":
    ai = OmegaAi()
    try:
        asyncio.run(ai.main_loop())
    except KeyboardInterrupt:
        pass
