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
    "ddgs": "ddgs",
    "google.genai": "google-genai"
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
from prompt_toolkit.completion import WordCompleter, NestedCompleter
from prompt_toolkit.styles import Style as PromptStyle

# Environment Variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv = None

# Search
import warnings
# Specifically ignore the renaming warning from duckduckgo_search
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed to.*ddgs.*")
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

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

# ==================== PERSISTENT REPL ====================

class REPLManager:
    """Maintains a persistent interactive shell (Python/Node) across iterations."""
    def __init__(self, root: Path, lang: str = "python"):
        self.root = root
        self.lang = lang
        self.process = None
        self._start()

    def _start(self):
        cmd = [sys.executable, "-i", "-q"] if self.lang == "python" else ["node", "-i"]
        try:
            self.process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, cwd=self.root
            )
        except: self.process = None

    async def execute(self, code: str) -> str:
        if not self.process or self.process.poll() is not None:
            self._start()
        
        if not self.process: return "REPL Error: Process failed to start."
        
        try:
            # Wrap code to ensure it prints output if it's an expression (Python)
            clean_code = textwrap.dedent(code).strip()
            self.process.stdin.write(clean_code + "\nprint('---OMEGA_EOF---')\n")
            self.process.stdin.flush()
            
            output = ""
            while True:
                line = await asyncio.to_thread(self.process.stdout.readline)
                if '---OMEGA_EOF---' in line: break
                output += line
            return output.strip()
        except Exception as e:
            return str(e)

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
            if json_path.exists():
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

    # Add Google AI Studio Models (Dynamic + Core Fallback)
    catalog["Google AI Studio"] = {
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash"
    }
    
    google_json_path = Path("google_free_models.json")
    if google_json_path.exists():
        try:
            with open(google_json_path, 'r', encoding='utf-8') as f:
                google_data = json.load(f).get("data", [])
                for m in google_data:
                    m_id = m.get("model", "").replace("models/", "")
                    if not m_id: continue
                    # Use a clean display name
                    display_name = m.get("display_name", m_id)
                    catalog["Google AI Studio"][display_name] = m_id
        except: pass
    
    # Add Groq Models (Lightning-Fast LPU Inference)
    groq_json_path = Path("groq_models.json")
    if groq_json_path.exists():
        try:
            with open(groq_json_path, 'r', encoding='utf-8') as f:
                groq_data = json.load(f).get("data", [])
                catalog["Groq (Verified Core)"] = {}
                for m in groq_data:
                    m_id = m.get("id", "")
                    # Skip Whisper models (audio-only)
                    if "whisper" in m_id.lower():
                        continue
                    # Create friendly names
                    name = m_id.split("/")[-1] if "/" in m_id else m_id
                    catalog["Groq (Verified Core)"][name] = m_id
        except: pass
    
    return catalog

MODELS_CATALOG = load_dynamic_models()
MODEL_MAP = {alias: id for cat in MODELS_CATALOG.values() for alias, id in cat.items()}


PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "env_key": "OPENROUTER_API_KEY",
        "refer": "https://openrouter.ai/keys"
    },
    "google": {
        "name": "Google AI Studio",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "env_key": "GOOGLE_API_KEY",
        "refer": "https://aistudio.google.com/app/apikey"
    },
    "groq": {
        "name": "Groq",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "env_key": "GROQ_API_KEY",
        "refer": "https://console.groq.com/keys"
    }
}

PROVIDER_PRESETS = {
    "openrouter": {
        "Ultra": {
            "planner": "anthropic/claude-3.5-sonnet",
            "architect": "openai/gpt-4o",
            "coder": "anthropic/claude-3.5-sonnet",
            "debugger": "openai/gpt-4o",
            "reviewer": "anthropic/claude-3.5-haiku"
        },
        "Core": {
            "planner": "google/gemini-2.0-flash-001",
            "architect": "meta-llama/llama-3.3-70b-instruct",
            "coder": "qwen/qwen-2.5-coder-32b-instruct",
            "debugger": "google/gemini-2.0-flash-001",
            "reviewer": "mistralai/mistral-small-24b-instruct-2501"
        },
        "Pulse": {
            "planner": "mistralai/devstral-2512:free",
            "architect": "mistralai/devstral-2512:free",
            "coder": "mistralai/devstral-2512:free",
            "debugger": "liquid/lfm-2.5-1.2b-thinking:free",
            "reviewer": "liquid/lfm-2.5-1.2b-instruct:free"
        }
    },
    "google": {
        "Ultra": {
            "planner": "gemini-2.0-flash",
            "architect": "gemini-1.5-pro",
            "coder": "gemini-2.0-flash",
            "debugger": "gemini-2.0-flash",
            "reviewer": "gemini-1.5-flash"
        },
        "Core": {
            "planner": "gemini-2.0-flash",
            "architect": "gemini-2.0-flash",
            "coder": "gemini-2.0-flash",
            "debugger": "gemini-2.0-flash",
            "reviewer": "gemini-1.5-flash"
        },
        "Pulse": {
            "planner": "gemini-2.0-flash-lite-preview-0205",
            "architect": "gemini-2.0-flash-lite-preview-0205",
            "coder": "gemini-2.0-flash-lite-preview-0205",
            "debugger": "gemini-2.0-flash-lite-preview-0205",
            "reviewer": "gemini-2.0-flash-lite-preview-0205"
        }
    },
    "groq": {
        "Ultra": {
            "planner": "llama-3.3-70b-versatile",
            "architect": "llama-3.3-70b-versatile",
            "coder": "llama-3.3-70b-versatile",
            "debugger": "llama-3.3-70b-versatile",
            "reviewer": "llama-3.3-70b-specdec"
        },
        "Core": {
            "planner": "llama-3.3-70b-versatile",
            "architect": "llama-3.1-70b-versatile",
            "coder": "llama-3.3-70b-versatile",
            "debugger": "llama-3.1-8b-instant",
            "reviewer": "llama-3.1-8b-instant"
        },
        "Pulse": {
            "planner": "llama-3.1-8b-instant",
            "architect": "llama-3.1-8b-instant",
            "coder": "llama-3.1-8b-instant",
            "debugger": "llama-3.1-8b-instant",
            "reviewer": "llama-3.1-8b-instant"
        }
    }
}

DEFAULT_CONFIG = {
    "active_provider": "openrouter",
    "active_preset": "Core",
    "models": PROVIDER_PRESETS["openrouter"]["Core"].copy(),
    "provider_models": {
        "openrouter": PROVIDER_PRESETS["openrouter"]["Core"].copy(),
        "google": PROVIDER_PRESETS["google"]["Core"].copy(),
        "groq": PROVIDER_PRESETS["groq"]["Core"].copy()
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

COMMAND_HELP = {
    "vibe": {
        "desc": "Initiate the autonomous coding pipeline for a specific task.",
        "usage": "/vibe <detailed task description>",
        "example": "/vibe create a snake game in python using pygame"
    },
    "tree": {
        "desc": "Display a visual directory tree of the current workspace.",
        "usage": "/tree",
        "example": "/tree"
    },
    "undo": {
        "desc": "Rollback the workspace and config to the previous iteration.",
        "usage": "/undo",
        "example": "/undo"
    },
    "models": {
        "desc": "List all available models categorized by provider and capability.",
        "usage": "/models",
        "example": "/models"
    },
    "model": {
        "desc": "Switch roles or the entire system to a specific model.",
        "usage": "/model <alias> OR /model set <role> <alias>",
        "example": "/model gpt-4o | /model set coder claude-3.5-sonnet",
        "subs": ["set"]
    },
    "auto-models": {
        "desc": "Automatically assign the best available models based on your provider.",
        "usage": "/auto-models",
        "example": "/auto-models"
    },
    "models-tier": {
        "desc": "Switch between pre-defined model tiers (Paid, Free, ExtraFree).",
        "usage": "/models-tier <paid|fullfree|freetier|extrafree>",
        "example": "/models-tier extrafree",
        "subs": ["paid", "fullfree", "freetier", "extrafree"]
    },
    "tools": {
        "desc": "Toggle specific autonomous capabilities (RAG, Search, Vision).",
        "usage": "/tools <name>",
        "example": "/tools search | /tools rag",
        "subs": ["search", "rag", "persistence", "vision", "patching"]
    },
    "provider": {
        "desc": "Switch the active API provider (OpenRouter/Google/Groq).",
        "usage": "/provider <id>",
        "example": "/provider google | /provider groq",
        "subs": ["google", "openrouter", "groq"]
    },
    "history": {
        "desc": "Display command history with optional filtering.",
        "usage": "/history <search_term>",
        "example": "/history python"
    },
    "config": {
        "desc": "View the current JSON configuration of the engine.",
        "usage": "/config",
        "example": "/config"
    },
    "presets": {
        "desc": "View the Global Catalogue of pre-calibrated mission profiles across all providers.",
        "usage": "/presets",
        "example": "/presets"
    },
    "preset": {
        "desc": "Instantly apply a pre-calibrated mission profile (Ultra, Core, Pulse) to your active provider.",
        "usage": "/preset <Ultra|Core|Pulse>",
        "example": "/preset Ultra",
        "subs": ["Ultra", "Core", "Pulse"]
    },
    "help": {
        "desc": "Show the dashboard or specific command documentation.",
        "usage": "/help <command>",
        "example": "/help models-tier"
    },
    "calibrate": {
        "desc": "Run a neural link performance audit and auto-optimize model assignments.",
        "usage": "/calibrate",
        "example": "/calibrate"
    },
    "menu": {
        "desc": "Access the primary command dashboard.",
        "usage": "/menu",
        "example": "/menu"
    },
    "exit": {
        "desc": "Power down the agent and save session state.",
        "usage": "/exit",
        "example": "/exit"
    },
    "stats": {
        "desc": "View real-time neural resource telemetry (Tokens, Costs, Logic Load).",
        "usage": "/stats",
        "example": "/stats"
    },
    "sandbox": {
        "desc": "Temporal Isolation Mode. Creates a safe 'Draft' workspace for non-destructive coding.",
        "usage": "/sandbox",
        "example": "/sandbox"
    },
    "commit": {
        "desc": "Permanently merge sandbox changes into the main project workspace.",
        "usage": "/commit",
        "example": "/commit"
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
        # Synchronize active models with provider_models before saving
        ap = self.config.get("active_provider")
        if ap and "models" in self.config:
            if "provider_models" not in self.config:
                self.config["provider_models"] = {}
            self.config["provider_models"][ap] = self.config["models"].copy()
            
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
        """Advanced project indexer with keyword mapping and signature extraction."""
        index = {}
        for file in self.root.rglob("*"):
            if file.is_file() and not any(part.startswith(".") for part in file.parts) and file.suffix in [".py", ".js", ".ts", ".html", ".css", ".md", ".json"]:
                # Ignore common junk
                if any(x in str(file) for x in ["node_modules", "venv", "__pycache__", "logs"]): continue
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        content = "".join(lines)
                        
                        # Extract signatures (classes/defs)
                        sigs = re.findall(r'(?:class|def|function|async\s+function)\s+([a-zA-Z_][\w\d_]*)', content)
                        
                        # Identify dominant tech/purpose
                        purpose = "content"
                        if file.suffix == ".py": purpose = "python_logic"
                        elif file.suffix in [".js", ".ts"]: purpose = "frontend_logic"
                        elif file.suffix == ".json": purpose = "configuration"
                        
                        index[str(file.relative_to(self.root))] = {
                            "size": len(content),
                            "signatures": sigs[:15],
                            "purpose": purpose,
                            "top_lines": [l.strip() for l in lines[:5] if l.strip()]
                        }
                except: continue
        with open(self.rag_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        return f"Neural Index Built: {len(index)} files mapped."

    async def build_semantic_index(self, client: 'OmegaClient'):
        """Asynchronously adds vectors to the existing index using Gemini."""
        if not self.rag_path.exists(): return
        
        with open(self.rag_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        updated = 0
        for path, data in index.items():
            if "vector" not in data:
                # Chunk top lines for embedding
                text = f"File: {path}\nSignatures: {','.join(data['signatures'])}\nContent: {','.join(data['top_lines'])}"
                data["vector"] = await client.embed_text(text)
                updated += 1
                if updated % 5 == 0: # Small delay to avoid rate limits
                    await asyncio.sleep(0.5)
        
        with open(self.rag_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        return f"Semantic Update: {updated} vectors generated."

    async def semantic_search(self, query: str, client: 'OmegaClient', top_k: int = 5) -> List[str]:
        """Perform vector-based similarity search against the project index."""
        if not self.rag_path.exists(): return []
        
        query_vec = await client.embed_text(query)
        if not query_vec: return []

        with open(self.rag_path, 'r', encoding='utf-8') as f:
            index = json.load(f)

        results = []
        for path, data in index.items():
            vec = data.get("vector")
            if not vec: continue
            
            # Manual cosine similarity Calculation
            dot = sum(a*b for a, b in zip(query_vec, vec))
            norm_a = sum(a*a for a in query_vec) ** 0.5
            norm_b = sum(b*b for b in vec) ** 0.5
            score = dot / (norm_a * norm_b) if norm_a and norm_b else 0
            results.append((path, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:top_k]]

# ==================== NEURAL CALIBRATION ====================

class NeuralCalibrator:
    """Unifies multi-provider performance auditing and auto-optimization."""
    def __init__(self, ai_client: 'OmegaClient', workspace: 'WorkspaceManager'):
        self.ai = ai_client
        self.workspace = workspace
        self.results = {}

    async def audit_provider(self, provider_name: str, models_file: str) -> Dict:
        """Perform a live performance audit on all models in the provider catalogue."""
        if not Path(models_file).exists():
            return {"error": f"Catalogue {models_file} not found."}

        with open(models_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            # Handle both OpenRouter/Groq (data list) and Google (data list of model objects)
            models = []
            if provider_name == "google":
                models = [m["model"] for m in raw_data.get("data", [])]
            else:
                models = [m.get("id") or m.get("model") for m in raw_data.get("data", []) if m]
        
        models = [m for m in models if m] # Clean up
        results = []
        semaphore = asyncio.Semaphore(3) # Respect rate limits
        
        async def test_one(m_id):
            async with semaphore:
                start = time.time()
                try:
                    # Simple prompt to measure TTFT and speed
                    # We use a direct stream call to bypass the full agent logic
                    headers = {"Content-Type": "application/json"}
                    if provider_name == "openrouter":
                        headers["Authorization"] = f"Bearer {self.ai.api_key}"
                    elif provider_name == "google":
                        headers["Authorization"] = f"Bearer {self.ai.api_key}"
                    elif provider_name == "groq":
                        headers["Authorization"] = f"Bearer {self.ai.api_key}"

                    payload = {
                        "model": m_id.replace("models/", "") if provider_name == "google" else m_id,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 10
                    }
                    
                    # Log attempt for debugging if needed (internal)
                    # print(f"Testing {m_id} on {provider_name}...") 
                    
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(PROVIDERS[provider_name]["endpoint"], headers=headers, json=payload, timeout=10)
                        latency = time.time() - start
                        if resp.status_code == 200:
                            return {"id": m_id, "status": "PASS", "latency": latency}
                        return {"id": m_id, "status": "FAIL", "error": resp.status_code}
                except Exception as e:
                    return {"id": m_id, "status": "FAIL", "error": str(e)}

        tasks = [test_one(m) for m in models[:15]] # Test top 15 for speed
        results = await asyncio.gather(*tasks)
        return {"provider": provider_name, "results": results, "timestamp": time.time()}

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
        # REFINED: Use non-greedy but ensure we don't swallow other <run> tags
        for m in re.finditer(r'<run>((?:(?!<run>).)*?)</run>', text, re.DOTALL | re.IGNORECASE):
            tools.append({"type": "run", "cmd": m.group(1).strip()})
            
        # XML Search: <search>query</search>
        # REFINED: Ensure it's not a mention and not inside patch
        search_matches = re.finditer(r'<search>((?:(?!<search>).)*?)</search>', text, re.DOTALL | re.IGNORECASE)
        for m in search_matches:
            query = m.group(1).strip()
            # COLLISION FIX: If it's inside a <patch> tool, it will be handled there.
            # We can check if it looks like a regex search/replace part or if it's already
            # inside a <patch> block by doing a broader check.
            is_patch_part = False
            # If the response contains <patch and the query is followed by <replace, it's a patch
            if '<patch' in text.lower() and '<replace>' in text.lower():
                # Naive but effective check: is there a <replace> tag later in the text?
                # Actually, ToolParser.parse for patch handles it specifically.
                # To be 100% safe, we only accept <search> as a web tool if it's NOT inside <patch> tags.
                patch_blocks = re.findall(r'<patch.*?>(.*?)</patch>', text, re.DOTALL | re.IGNORECASE)
                for block in patch_blocks:
                    if query in block:
                        is_patch_part = True
                        break
            
            if not is_patch_part:
                tools.append({"type": "search", "query": query})

        # XML Vision: <read_vision path="image" prompt="query" />
        for m in re.finditer(r'<read_vision\s+path=["\']?([^"\'>\s]+)["\']?\s+prompt=["\']?([^"\'>]+)["\']?\s*/>', text, re.IGNORECASE):
            tools.append({"type": "vision", "path": m.group(1), "prompt": m.group(2)})
            
        # Markdown Fallback for Write: ```(file:path) or just ```path
        # REFINED: Do not fallback if we already have explicit XML tools (high precision mode)
        # OR if the "path candidate" contains suspicious characters like < > [ ] or looks like a tool tag.
        if not tools:
            # Known language identifiers to ignore if they appear alone
            LANG_TAGS = {"python", "py", "javascript", "js", "typescript", "ts", "html", "css", "json", "markdown", "md", "sql", "bash", "sh", "yaml", "yml", "xml", "txt"}
            blocks = re.finditer(r'```(?:\w+)?\s*(?:\(file:\s*([^\)]+)\)|([^\n\r]+))?\n(.*?)```', text, re.DOTALL)
            for b in blocks:
                path = (b.group(1) or b.group(2) or "").strip()
                if path:
                    # Sanity checks for a path
                    is_lang_tag = path.lower() in LANG_TAGS
                    has_path_features = "/" in path or "\\" in path or "." in path
                    has_invalid_chars = any(c in path for c in '<>[]{}|*?')
                    
                    if has_path_features and not is_lang_tag and not has_invalid_chars:
                        # Extra check: ensure it's not starting with a tool-like prefix
                        if not path.lstrip().startswith(('<', 'shell:')):
                            # Extra check: if it ends with a slash, it's a folder, skip
                            if not path.endswith(('/', '\\')):
                                tools.append({"type": "write", "path": path, "content": b.group(3).strip()})
        
        # Markdown Fallback for Run: `shell:command`
        if not any(t["type"] == "run" for t in tools):
            for m in re.finditer(r'`shell:(.*?)`', text):
                tools.append({"type": "run", "cmd": m.group(1).strip()})

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

        # XML REPL: <repl>code</repl>
        for m in re.finditer(r'<repl>(.*?)</repl>', text, re.DOTALL | re.IGNORECASE):
            tools.append({"type": "repl", "code": m.group(1).strip()})

        return tools

# ==================== CLIENT & AGENT ====================

class OmegaClient:
    def __init__(self, api_key: str, models: Dict[str, str], provider: str = "openrouter"):
        self.api_key = api_key
        self.models = models # Dictionary of roles: model_id
        self.provider = provider
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/antigravity-ai/omega",
            "X-Title": "OmegaAi Agent"
        }
        self.history = []
        self.logger = Logger(Path.cwd()) # Fallback logger access
        
        # Telemetry
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}
        self.prices = {
            "anthropic/claude-3.5-sonnet": (3.0, 15.0),
            "openai/gpt-4o": (2.5, 10.0),
            "openai/gpt-4o-mini": (0.15, 0.6),
            "google/gemini-2.0-flash": (0.1, 0.4),
            "meta-llama/llama-3.3-70b": (0.7, 0.9)
        }

    async def stream(self, prompt: str, system: str = "", role: str = "coder"):
        model = self.models.get(role, self.models.get("coder"))
        provider_config = PROVIDERS.get(self.provider, PROVIDERS["openrouter"])
        
        # USE SDK FOR GOOGLE PROVIDER
        if self.provider == "google" and genai:
            try:
                client = genai.Client(api_key=self.api_key)
                # Normalize model ID early
                clean_model = model.replace("models/", "")
                
                # Prepare history for Google SDK
                contents = []
                for msg in self.history:
                    role_map = {"user": "user", "assistant": "model", "system": "system"}
                    contents.append(genai_types.Content(
                        role=role_map.get(msg["role"], "user"),
                        parts=[genai_types.Part(text=msg["content"])]
                    ))
                
                # Check for instruction support (Gemma 3 bypass)
                can_instruct = "gemma" not in clean_model.lower()
                
                if system and not can_instruct:
                    # Prepend system context to the user message
                    prompt = f"[SYSTEM CONTEXT]\n{system}\n\n[USER REQUEST]\n{prompt}"
                    system = None # Clear so it doesn't go to config

                # Add current prompt
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)]
                ))
                
                full = ""
                # Use generate_content_stream for true streaming
                config = genai_types.GenerateContentConfig(
                    system_instruction=system if system else None,
                    temperature=0.3
                )
                
                response = client.models.generate_content_stream(
                    model=clean_model,
                    contents=contents,
                    config=config
                )
                
                for chunk in response:
                    if chunk.text:
                        full += chunk.text
                        yield ("content", chunk.text)
                
                if full:
                    self.history.append({"role": "user", "content": prompt})
                    self.history.append({"role": "assistant", "content": full})
                    if len(self.history) > 20: self.history = self.history[-20:]
                return
            except Exception as e:
                yield ("error", f"\n[bold red]GOOGLE SDK ERROR:[/bold red] {str(e)}")
                return

        # FALLBACK TO OPENROUTER/OPENAI COMPATIBILITY
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.3
                }
                
                async with client.stream("POST", 
                    provider_config["endpoint"],
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        try:
                            err_json = json.loads(error_body)
                            err_msg = err_json.get("error", {}).get("message", error_body.decode())
                        except:
                            err_msg = error_body.decode()
                        
                        hint = ""
                        if response.status_code == 401:
                            hint = f"\n[yellow]HINT: Unauthorized. Your {provider_config['name']} API key might be invalid.[/yellow]"
                        if response.status_code == 404:
                            hint = f"\n[yellow]HINT: Model '{model}' not found on {provider_config['name']}.[/yellow]"
                        
                        yield ("error", f"\n[bold red]{provider_config['name']} API ERROR {response.status_code}:[/bold red] {err_msg}{hint}")
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
                                
                                reasoning = delta.get('reasoning', '')
                                if reasoning:
                                    full += reasoning
                                    yield ("reasoning", reasoning)
                                
                                content = delta.get('content', '')
                                if content:
                                    full += content
                                    yield ("content", content)
                            except Exception as e:
                                if data.strip() and data.strip() != "[DONE]":
                                    pass
                    
                    if full:
                        self.history.append({"role": "user", "content": prompt})
                        self.history.append({"role": "assistant", "content": full})
                        if len(self.history) > 20: self.history = self.history[-20:]
                        
                        # Update Telemetry (Est.)
                        p_tok = TokenCounter.estimate(prompt + system)
                        c_tok = TokenCounter.estimate(full)
                        self.usage["prompt_tokens"] += p_tok
                        self.usage["completion_tokens"] += c_tok
                        
                        # Calculate Cost
                        rate = self.prices.get(model, (0.5, 1.5))
                        cost = (p_tok * rate[0] / 1_000_000) + (c_tok * rate[1] / 1_000_000)
                        self.usage["total_cost"] += cost
            except Exception as e:
                yield ("error", f"\n[bold red]CONNECTION FAILED ({provider_config['name']}):[/bold red] {str(e)}")

    async def embed_text(self, text: str) -> List[float]:
        """Generate high-fidelity embeddings for semantic search."""
        if self.provider == "google" and genai:
            try:
                # Use sync client in thread for embedding call as it's typically faster/simpler for single calls
                client = genai.Client(api_key=self.api_key)
                res = client.models.embed_content(model="text-embedding-004", contents=text)
                return res.embeddings[0].values
            except: return []
        return []

    async def stream_vision(self, prompt: str, image_path: Path):
        """Streaming vision support for multi-modal models."""
        if self.provider == "google" and genai:
            try:
                client = genai.Client(api_key=self.api_key)
                with open(image_path, "rb") as f:
                    img_data = f.read()
                
                response = client.models.generate_content(
                    model=self.models.get("coder", "gemini-2.0-flash").replace("models/", ""),
                    contents=[
                        genai_types.Part.from_bytes(data=img_data, mime_type="image/jpeg"),
                        genai_types.Part.from_text(text=prompt)
                    ]
                )
                return response.text
            except Exception as e:
                return f"Vision Error: {e}"
        return "Vision support is currently optimized for Google provider."

class SelfHealingManager:
    """Intelligent Error Analysis and Recommendation System."""
    def __init__(self, ai_client: OmegaClient):
        self.ai = ai_client

    async def heal(self, command: str, error: str):
        prompt = f"""FAILURE ANALYSIS REQUEST
Failed Command: {command}
Error Output: {error}

Task: Identify the root cause and provide a specific, one-line shell command to fix it or an explanation of why it failed. 
Keep it technical and concise."""
        heal_res = ""
        async for type, chunk in self.ai.stream(prompt, system="You are the Neural Healer. Fix system errors.", role="debugger"):
            if type == "content": heal_res += chunk
        return heal_res.strip()

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
        if self.config.get("tools", "rag", default=True):
            self.knowledge.build_index()
        
        # Log startup
        self.logger.info(f"OmegaAi initialized in {self.root}")
        
        # Initialize Client
        if not self.setup_client():
            console.print("[bold red]Critical Error: No API Key provided. Agent cannot function.[/bold red]")
            sys.exit(1)
            
        self.iteration = 0
        self.self_healing = SelfHealingManager(self.client)
        self.calibrator = NeuralCalibrator(self.client, self.workspace)
        self.repl = REPLManager(self.root)
        self.sandbox_orig = None

    def setup_client(self, provider_name: str = None) -> bool:
        """Configures the current AI provider and initializes the client with stateful model memory."""
        # 1. Save current models state to the old provider's preferences before switching
        if hasattr(self, 'client') and self.client.provider:
            current_p = self.client.provider
            if "provider_models" not in self.config.config:
                self.config.config["provider_models"] = {}
            self.config.config["provider_models"][current_p] = self.config.config.get("models", {}).copy()

        if not provider_name:
            provider_name = self.config.get("active_provider", default="openrouter")

        if provider_name not in PROVIDERS:
            provider_name = "openrouter"

        info = PROVIDERS[provider_name]
        api_key = os.getenv(info["env_key"])

        if not api_key:
            console.print(Panel(
                f"[bold yellow]⚠️  {info['name']} API Key Missing [/bold yellow]\n\n"
                f"Provider: [bold cyan]{info['name']}[/bold cyan]\n"
                f"Required Key: [dim]{info['env_key']}[/dim]\n"
                f"Get it here: [blue underline]{info['refer']}[/blue underline]\n\n"
                f"OmegaAi needs this key to power its neural core.",
                title="Neural Link Required", border_style="yellow"
            ))
            api_key = Prompt.ask(f"Enter {info['name']} API Key", password=True)
            if api_key:
                os.environ[info["env_key"]] = api_key.strip()
            else:
                return False

        # 2. Load the saved models for the NEW provider if they exist
        if "provider_models" in self.config.config and provider_name in self.config.config["provider_models"]:
            self.config.config["models"] = self.config.config["provider_models"][provider_name].copy()
            console.print(f"[dim italic]📂 Restored previous model configuration for {info['name']}...[/dim italic]")

        self.config.config["active_provider"] = provider_name
        self.config.save()
        
        self.client = OmegaClient(api_key.strip(), self.config.get("models"), provider=provider_name)
        
        # Sync all neural sub-managers with the new client context
        if hasattr(self, 'calibrator'): self.calibrator.ai = self.client
        if hasattr(self, 'self_healing'): self.self_healing.ai = self.client
        if hasattr(self, 'repl'): self.repl.root = self.root # Ensure REPL stays in sync with workspace root
        
        return True

    def show_banner(self):
        banner_font = r"""
 ██████╗ ███╗   ███╗███████╗ ██████╗  █████╗     █████╗ ██╗
██╔═══██╗████╗ ████║██╔════╝██╔════╝ ██╔══██╗   ██╔══██╗██║
██║   ██║██╔████╔██║█████╗  ██║  ███╗███████║   ███████║██║
██║   ██║██║╚██╔╝██║██╔══╝  ██║   ██║██╔══██║   ██╔══██║██║
╚██████╔╝██║ ╚═╝ ██║███████╗╚██████╔╝██║  ██║██╗██║  ██║██║
 ╚═════╝ ╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝
         [ NEURAL EVOLUTION : STATE-OF-THE-ART AUTONOMY ]
"""
        active_models = self.config.get("models")
        current_p = PROVIDERS.get(self.client.provider, {"name": "Unknown"})
        tools_cfg = self.config.get("tools")
        
        # 1. Header Area
        header_text = Text.assemble(
            (banner_font, "bold cyan"),
            ("\n[ PREMIER AUTONOMOUS ENGINE V5.0 | MULTI-PROVIDER ARCHITECTURE ]", "bold white on blue")
        )
        console.print(Panel(header_text, border_style="bright_blue", padding=(1, 2), expand=True))

        # 2. Key Telemetry Dashboard (3 Columns)
        tele_table = Table(box=box.ROUNDED, border_style="blue", expand=True, show_header=True, header_style="bold blue")
        tele_table.add_column("🧠 AI NEURAL LINKS", ratio=1)
        tele_table.add_column("🌐 INFRASTRUCTURE", ratio=1)
        tele_table.add_column("📋 PROJECT STATE", ratio=1)

        # -- Column 1: AI Roles & Models --
        roles_grid = Table.grid(padding=(0, 1))
        roles_grid.add_column(style="bold cyan", width=12)
        roles_grid.add_column(style="white")
        roles_grid.add_row("CODER", f": [bold green]{active_models.get('coder', 'N/A')}[/bold green]")
        roles_grid.add_row("PLANNER", f": {active_models.get('planner', 'N/A')}")
        roles_grid.add_row("ARCHITECT", f": {active_models.get('architect', 'N/A')}")
        roles_grid.add_row("DEBUGGER", f": {active_models.get('debugger', 'N/A')}")

        # -- Column 2: Provider & System Meta --
        infra_grid = Table.grid(padding=(0, 1))
        infra_grid.add_column(style="bold magenta", width=12)
        infra_grid.add_column(style="white")
        infra_grid.add_row("PROVIDER", f": [bold magenta]{current_p['name']}[/bold magenta]")
        infra_grid.add_row("GATEWAY", f": [dim]{self.client.provider.upper()}[/dim]")
        infra_grid.add_row("OS ARCH", f": [dim]{platform.system()} {platform.machine()}[/dim]")
        infra_grid.add_row("SESSION", f": [dim]{datetime.now().strftime('%H:%M:%S')}[/dim]")

        # -- Column 3: Workspace & Activity --
        state_grid = Table.grid(padding=(0, 1))
        state_grid.add_column(style="bold yellow", width=12)
        state_grid.add_column(style="white")
        state_grid.add_row("WORKSPACE", f": [dim]{self.root.name}/[/dim]")
        state_grid.add_row("PRESET", f": [bold cyan]{self.config.get('active_preset', default='Core')}[/bold cyan]")
        state_grid.add_row("ITERATION", f": {self.iteration}")
        state_grid.add_row("KNOWLEDGE", f": {'[green]INDEXED[/green]' if tools_cfg.get('rag') else '[red]OFF[/red]'}")
        backup_count = len(list(self.backup.backup_dir.iterdir())) if self.backup.backup_dir.exists() else 0
        state_grid.add_row("BACKUPS", f": [dim]{backup_count} units[/dim]")

        tele_table.add_row(roles_grid, infra_grid, state_grid)
        console.print(tele_table)

        # 3. Toolbox Status Bar
        tool_badges = []
        for k, v in tools_cfg.items():
            icon = "●" if v else "○"
            color = "green" if v else "red"
            tool_badges.append(f"[{color}]{icon} {k.upper()}[/{color}]")
        
        # Add Healing Feature Badge
        healing_on = self.config.get("features", "self_healing")
        h_icon = "🧠" if healing_on else "○"
        h_color = "bold cyan" if healing_on else "red"
        tool_badges.append(f"[{h_color}]{h_icon} HEALER[/{h_color}]")
        
        console.print(Panel(Align.center("  |  ".join(tool_badges)), title="[bold green] HARDWARE TOOLBOX STATUS [/bold green]", border_style="green"))

        # 4. Command Dock (NEURAL EVOLUTION)
        cmd_grid = Table.grid(expand=True, padding=(0, 2))
        cmd_grid.add_column(style="bold yellow", ratio=1)
        cmd_grid.add_column(style="dim white", ratio=2)
        cmd_grid.add_column(style="bold yellow", ratio=1)
        cmd_grid.add_column(style="dim white", ratio=2)

        cmd_grid.add_row("🚀 /vibe <task>", "Autonomous pipeline", "🔬 /calibrate", "Neural performance audit")
        cmd_grid.add_row("🧠 /preset", "Swap intelligence", "📊 /stats", "Telemetry & Costs")
        cmd_grid.add_row("🛡️ /sandbox", "Temporal isolation", "❓ /help", "Summon dashboard")
        
        console.print(Panel(cmd_grid, title="[bold yellow] NEURAL QUICK ACCESS [/bold yellow]", border_style="yellow", padding=(1, 2)))

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
            if cmd.strip() == 'pwd': cmd = 'echo %cd%'
            if cmd.startswith('pwd '): cmd = cmd.replace('pwd ', 'echo %cd%')
            if cmd.startswith('cat '): cmd = cmd.replace('cat ', 'type ')
            if 'rm -rf' in cmd: cmd = cmd.replace('rm -rf', 'rmdir /s /q')
        
        # STATEFUL CD SUPPORT: If command is 'cd path', update internal root
        if cmd.lower().startswith('cd '):
            target = cmd[3:].strip().strip('"').strip("'")
            new_path = (self.root / target).resolve()
            if new_path.exists() and new_path.is_dir():
                self.root = new_path
                self.workspace.root = new_path
                return {"success": True, "exit_code": 0, "stdout": f"Successfully changed directory to {new_path}", "stderr": ""}
            else:
                return {"success": False, "exit_code": 1, "stdout": "", "stderr": f"Error: Directory '{target}' not found."}

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

                return {
                    "success": proc.returncode == 0,
                    "exit_code": proc.returncode,
                    "stdout": out[:3000],
                    "stderr": err[:3000]
                }
            except asyncio.TimeoutError:
                try:
                    proc.terminate()
                    await asyncio.sleep(0.2)
                    if proc.returncode is None: proc.kill()
                    await proc.wait()
                except: pass
                return {"success": False, "exit_code": -1, "stdout": "", "stderr": "Command timed out."}
        except Exception as e:
            return {"success": False, "exit_code": -1, "stdout": "", "stderr": str(e)}

    async def exe_tools(self, tools: List[Dict]):
        results = []
        healer = SelfHealingManager(self.client)
        
        # 1. Parallelize independent tasks (Reads and Searches)
        indep_tools = [t for t in tools if t["type"] in ["read", "search"]]
        if indep_tools:
            async def run_indep(t):
                if t["type"] == "read":
                    res = await self.workspace.read(t["path"])
                    return f"Content of {t['path']}:\n{res[:1000]}"
                elif t["type"] == "search":
                    if self.config.get("tools", "search") and DDGS:
                        try:
                            with DDGS() as ddgs:
                                res = [r for r in ddgs.text(t["query"], max_results=5)]
                                return f"Search results for '{t['query']}':\n{res}"
                        except: return "Search failed."
                return ""
            c_res = await asyncio.gather(*(run_indep(t) for t in indep_tools))
            results.extend([r for r in c_res if r])

        # 2. Sequential tasks for stateful operations
        for t in tools:
            if t["type"] == "write":
                res = await self.workspace.write(t["path"], t["content"])
                results.append(res)
            elif t["type"] == "run":
                res_obj = await self.run_command(t["cmd"])
                if not res_obj["success"] and self.config.get("features", "self_healing"):
                    console.print(f"[bold red]🧠 Healing Trip:[/bold red] Analyzing failure: {t['cmd']}")
                    heal_fix = await healer.heal(t['cmd'], res_obj['stderr'])
                    results.append(f"FAILED: {t['cmd']}\nERROR: {res_obj['stderr']}\nHEALER SUGGESTED FIX: {heal_fix}")
                else:
                    results.append(f"Exit Code: {res_obj['exit_code']}\nOutput: {res_obj['stdout']}\nError: {res_obj['stderr']}")
            elif t["type"] == "patch":
                res = await self.patch_file(t["path"], t["search"], t["replace"])
                results.append(res)
            elif t["type"] == "vision":
                img_path = self.root / t["path"]
                if img_path.exists():
                    console.print(f"[bold cyan]👁️ Vision Analysis:[/bold cyan] Scanning {t['path']}...")
                    res = await self.client.stream_vision(t["prompt"], img_path)
                    results.append(f"Vision Analysis for {t['path']}:\n{res}")
                else:
                    results.append(f"Error: Image {t['path']} not found.")
            elif t["type"] == "repl":
                if not hasattr(self, 'repl'):
                    self.repl = REPLManager(self.root)
                console.print(f"[bold magenta]⚡ Executing in REPL...[/bold magenta]")
                res = await self.repl.execute(t["code"])
                results.append(f"REPL Output:\n{res}")
        
        return "\n".join(results)

    async def patch_file(self, path: str, search: str, replace: str):
        full_path = self.root / path
        if not full_path.exists(): return f"Error: File {path} not found."
        async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        if search in content:
            new_content = content.replace(search, replace)
        else:
            # Fuzzy Matching Logic
            import difflib
            lines = content.splitlines(keepends=True)
            search_lines = search.splitlines(keepends=True)
            n = len(search_lines)
            
            best_ratio = 0
            best_block = None
            
            # Sliding window fuzzy match
            for i in range(len(lines) - n + 1):
                window = "".join(lines[i:i+n])
                ratio = difflib.SequenceMatcher(None, window, search).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_block = window
            
            if best_ratio > 0.85: # High confidence threshold
                console.print(f"[bold yellow]⚠️ Fuzzy Match Found ({int(best_ratio*100)}%):[/bold yellow] Applied patch to a similar block.")
                new_content = content.replace(best_block, replace)
            else:
                return f"Error: Could not find search block in {path}. Exact match failed and fuzzy match confidence too low ({int(best_ratio*100)}%)."

        async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
            await f.write(new_content)
        return f"Successfully patched {path}."

    def detect_project_folder(self, task: str) -> Optional[str]:
        """Analyzes prompt to see if a new project folder is needed."""
        triggers = ['create', 'new', 'start', 'build', 'initialize', 'make', 'generate', 'setup', 'set up', 'scaffold', 'bootstrap']
        targets = ['project', 'app', 'application', 'game', 'tool', 'script', 'website', 'utility', 'dashboard', 'system', 'api', 'bot', 'engine', 'module', 'library', 'wrapper']
        
        task_lower = task.lower()
        
        # Priority 1: Direct "in folder X" or "in a directory named X"
        match_folder = re.search(r'in (?:a\s+)?(?:folder|directory|dir)\s*(?:named|called)?\s*["\']?([\w\-_]+)["\']?', task_lower)
        if match_folder:
            return match_folder.group(1)

        # Priority 2: "for a X project" or "build a X app"
        has_trigger = any(t in task_lower for t in triggers)
        has_target = any(t in task_lower for t in targets)
        
        if (has_trigger and has_target) or ("new" in task_lower and has_target):
            # Extract name: filter out stops and triggers/targets
            clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', task)
            words = clean.split()
            stop = set(triggers + targets + ['a', 'an', 'the', 'of', 'in', 'with', 'named', 'called', 'for', 'simple', 'basic', 'advanced', 'my', 'is', 'it', 'to', 'using', 'from', 'at'])
            relevant_words = [w.lower() for w in words if w.lower() not in stop]
            if relevant_words:
                # Join up to 3 words
                return "_".join(relevant_words[:3]) or "new_project"
            elif has_target:
                for t in targets:
                    if t in task_lower: return f"new_{t}"
        return None

    async def process_task(self, task: str):
        self.iteration = 0
        self.client.history = [] 
        
        # Smart Auto-Naming and Project Folder Detection
        project_name = self.detect_project_folder(task)
        if project_name:
            project_dir = self.root / project_name
            if not project_dir.exists():
                console.print(f"[bold yellow]📂 Smart Detection:[/bold yellow] Project request detected. Creating folder: [bold cyan]{project_name}[/bold cyan]")
                project_dir.mkdir(parents=True, exist_ok=True)
                # Save internal root before switching for undo/rollback capability
                self.old_root = self.root
                self.root = project_dir
                self.workspace.root = project_dir
                self.backup.root = project_dir
                self.knowledge.root = project_dir
            else:
                console.print(f"[bold blue]📂 Smart Detection:[/bold blue] Using existing project folder: [bold cyan]{project_name}[/bold cyan]")
                self.root = project_dir
                self.workspace.root = project_dir
                self.backup.root = project_dir
                self.knowledge.root = project_dir

        self.backup.create(0, "pre_task")
        
        # Project context for the Architect to prevent redundant mkdirs
        project_context = ""
        if project_name:
            project_context = f"\n[IMPORTANT] You are already initialized inside the project folder: '{project_name}'.\nThis is your WORKSPACE ROOT. Do NOT create another '{project_name}' folder or navigate 'up' to create siblings.\nBuild the project files DIRECTLY in the current directory."
        
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
            
            # Map display role to config role
            role_map = {"Developer": "coder", "Architect": "architect", "Debugger": "debugger", "Planner": "planner"}
            role_key = role_map.get(role, role.lower())
            model_id = self.client.models.get(role_key, "Unknown")
            # Clean model ID for display
            model_display = model_id.split("/")[-1]

            return Panel(
                Markdown(full_text),
                title=f"[bold magenta]OmegaAi {role.capitalize()}[/bold magenta] [dim](Iteration {it}) Model: {model_display}[/dim]",
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
{project_context}

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
<read_vision path="image" prompt="query" />
<insight>Learned detail</insight>

IMPORTANT RULES:
1. ALWAYS use relative paths from the workspace root.
2. REAL-WORLD DATA: If the task requires facts, weather, news, or any real-world data, you MUST use the <search> tool. DO NOT simulate or hallucinate this data.
3. SURGICAL UPDATES: Preferred <patch> over <write> for existing large files to save tokens and avoid errors.
4. VISUAL AUDIT: If you create a UI, you MUST explain the aesthetic choices (typography, colors) in reasoning.
5. DO NOT create redundant nested directories.
6. YOU MUST START EVERY TOOL CALL WITH '<'.
7. If you learn something critical about how this project works (e.g., specific port, library version), include the tag <insight>Learned detail</insight> in your response to save it to Project Knowledge.
9. SMART NAMING: Choose descriptive, professional, and industry-standard names for all new files and folders (e.g., snake_case for Python/Ruby, kebab-case for HTML/CSS, PascalCase for React components). Avoid generic names like 'file1.txt' or 'code.py'.
11. STATEFUL NAVIGATION: Virtual directory changes using 'cd' are supported and persistent across tool calls within the same session.
12. VISION: Use <read_vision> for UI inspection or image analysis.
13. SELF-HEALING: Shell failures are automatically analyzed by the Neural Healer.
14. Once finished, say "TASK COMPLETE".
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
            
            # Parse tools from BOTH content and reasoning to be safe
            combined_output = f"{reasoning}\n{response}"
            tools = ToolParser.parse(combined_output)
            
            if tools:
                results = await self.exe_tools(tools)
                self.logger.tool("Execution", results)
                current_input = f"Tool Results:\n{results}"
            
            if "TASK COMPLETE" in response.upper() or "TASK COMPLETE" in reasoning.upper():
                self.logger.info("Task marked complete by AI.")
                break
                
            if not tools:
                if not response.strip() and not reasoning.strip():
                    self.logger.info("Empty response from model.")
                    current_input = "You returned an empty response. Please provide a tool call (like <write> or <run>) to proceed with the implementation, or say TASK COMPLETE if you are finished."
                else:
                    self.logger.info("No tool calls detected in AI response.")
                    # If the AI is just talking, maybe it forgot to say TASK COMPLETE
                    current_input = "If you have finished the requested work, please say 'TASK COMPLETE'. Otherwise, use a tool like <write> or <run> to continue implementation."
                continue
            
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
4. Use <run> to execute tests or linters if needed.
5. If issues are found, use tools to FIX them.
6. If everything looks perfect, say "SYSTEM VERIFIED".
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
                
        console.print(Panel("[bold green]🏁 MISSION ACCOMPLISHED[/bold green]\nAll phases complete. The agent has signed off on the workspace.", border_style="green"))

        # 4. Status Bar
        console.print(Panel(f" [bold white]READY[/bold white] | [cyan]ITERATIONS:[/cyan] {self.iteration} | [magenta]MODE:[/magenta] Autonomous", style="bright_blue"))

    def show_dashboard_menu(self):
        """Displays a premium dashboard-style help menu"""
        # 1. Dashboard Header
        title = Text.assemble(
            ("     Ω ", "bold cyan"),
            ("OMEGAAI COMMAND DASHBOARD", "bold white"),
            (" v5.0     ", "dim cyan")
        )
        
        # 2. Command Categories (3-Column Layout)
        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        # --- Column 1: NEURAL OPERATIONS ---
        neural_table = Table(title="⚡ NEURAL OPERATIONS", box=box.SIMPLE, header_style="bold cyan", border_style="cyan")
        neural_table.add_column("Command", style="yellow")
        neural_table.add_column("Description", style="white")
        neural_table.add_row("/vibe <task>", "Initiate autonomous pipeline")
        neural_table.add_row("/tree", "Visualize workspace architecture")
        neural_table.add_row("/presets", "View global mission profiles")
        neural_table.add_row("/preset <tier>", "Switch to Ultra/Core/Pulse")
        neural_table.add_row("/undo", "Rewind to previous temporal state")
        
        # --- Column 2: AI HUB CONTROL ---
        ai_table = Table(title="🧠 AI HUB CONTROL", box=box.SIMPLE, header_style="bold magenta", border_style="magenta")
        ai_table.add_column("Command", style="yellow")
        ai_table.add_column("Description", style="white")
        ai_table.add_row("/models", "Query available intelligence units")
        ai_table.add_row("/model <id>", "Hot-swap primary logic unit")
        ai_table.add_row("/models-tier", "Switch model pricing/power tiers")
        ai_table.add_row("/auto-models", "Calibrate optimal role mapping")
        
        # --- Column 3: SYSTEM CORE ---
        system_table = Table(title="⚙️ SYSTEM CORE", box=box.SIMPLE, header_style="bold blue", border_style="blue")
        system_table.add_column("Command", style="yellow")
        system_table.add_column("Description", style="white")
        system_table.add_row("/provider", "Gateways (OpenRouter/Google/Groq)")
        system_table.add_row("/tools", "Toggle hardware capabilities")
        system_table.add_row("/history", "Probe neural command logs")
        system_table.add_row("/exit", "Secure system shutdown")

        # --- Column 4: ADVANCED OPS ---
        adv_table = Table(title="💎 ADVANCED OPS", box=box.SIMPLE, header_style="bold green", border_style="green")
        adv_table.add_column("Command", style="yellow")
        adv_table.add_column("Description", style="white")
        adv_table.add_row("/calibrate", "Optimize Neural Performance")
        adv_table.add_row("/stats", "Live Telemetry Dashboard")
        adv_table.add_row("/sandbox", "Toggle Temporal Isolation")
        adv_table.add_row("/commit", "Finalize Sandbox Changes")

        grid.add_row(neural_table, ai_table, system_table)
        grid.add_row(adv_table, "", "") # Aligning the grid

        # 3. Telemetry Footer
        active_provider = PROVIDERS.get(self.client.provider, {"name": "Unknown"})["name"]
        footer = Panel(
            grid,
            title=title,
            subtitle=f"[dim]Active Provider:[/dim] [bold green]{active_provider}[/bold green] | [dim]Press TAB for autocompletion[/dim]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        console.print(footer)

    def show_command_help(self, command_name: str):
        """Shows detailed intelligence for a specific command"""
        cmd_id = command_name.lower().replace("/", "")
        if cmd_id in COMMAND_HELP:
            info = COMMAND_HELP[cmd_id]
            h_panel = Panel(
                f"[bold yellow]Description:[/bold yellow] {info['desc']}\n"
                f"[bold cyan]Usage:[/bold cyan] [white]{info['usage']}[/white]\n"
                f"[bold magenta]Example:[/bold magenta] [dim]{info['example']}[/dim]",
                title=f"HELP: /{cmd_id.upper()}",
                border_style="cyan",
                expand=False,
                padding=(1, 2)
            )
            console.print(h_panel)
        else:
            console.print(f"[red]No special documentation for '{command_name}'. Try /menu for general list.[/red]")

    async def main_loop(self):
        self.show_banner()
        
        session = PromptSession(
            history=FileHistory(str(self.root / ".omega" / "cmd_history")),
            auto_suggest=AutoSuggestFromHistory()
        )
        
        while True:
            # DYNAMIC NEURAL COMPLETER: Refreshed every iteration to react to provider/model changes
            m_catalog = load_dynamic_models()
            p_name = self.client.provider
            
            # Filter intelligence units based on active gateway
            filtered_models = {}
            for cat, models in m_catalog.items():
                if p_name == "google" and cat == "Google AI Studio": filtered_models.update(models)
                elif p_name == "groq" and cat == "Groq (Verified Core)": filtered_models.update(models)
                elif p_name == "openrouter" and cat not in ["Google AI Studio", "Groq (Verified Core)"]: filtered_models.update(models)
            
            model_options = {alias: None for alias in filtered_models.keys()}
            model_options.update({m_id: None for m_id in filtered_models.values()})
            
            role_options = {r: model_options for r in ["planner", "architect", "coder", "debugger", "reviewer"]}
            
            completer_dict = {
                "/vibe": None,
                "/tree": None,
                "/config": None,
                "/undo": None,
                "/models": None,
                "/model": {
                    "set": role_options,
                    **role_options # Roles suggested first, models only after role
                },
                "/provider": {k: None for k in PROVIDERS.keys()},
                "/auto-models": None,
                "/presets": None,
                "/preset": {"Ultra": None, "Core": None, "Pulse": None},
                "/models-tier": {"paid": None, "fullfree": None, "freetier": None, "extrafree": None},
                "/tools": {k: None for k in DEFAULT_CONFIG["tools"].keys()},
                "/history": None,
                "/help": {f"{k}": None for k in COMMAND_HELP.keys()},
                "/menu": None,
                "/exit": None
            }
            session.completer = NestedCompleter.from_nested_dict(completer_dict)

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
                        
                        provider_name = PROVIDERS.get(self.client.provider, {}).get("name", "Unknown")
                        header = Panel(
                            Text.assemble(
                                (f" 🌐 {provider_name.upper()} CATALOGUE ", "bold white on blue"),
                                ("\n Use /model <alias> to switch all, or /model set <role> <alias>", "dim italic white")
                            ),
                            border_style="blue", padding=(1, 2)
                        )
                        outer_grid.add_row(header)
                        
                        model_tables = []
                        for category, models in MODELS_CATALOG.items():
                            # Filtering logic for cleaner view
                            if self.client.provider == "google" and category != "Google AI Studio": continue
                            if self.client.provider == "openrouter" and category == "Google AI Studio": continue
                            if self.client.provider == "groq" and category not in ["Groq (Verified Core)"]: continue
                            if self.client.provider != "groq" and category == "Groq (Verified Core)": continue

                            table = Table(title=f"📁 {category}", border_style="cyan", header_style="bold cyan", box=box.SIMPLE)
                            table.add_column("Alias", style="white", no_wrap=True)
                            table.add_column("Identifier", style="dim green")
                            table.add_column("Status", style="magenta")
                            
                            # Limit to top 20 per category for a better overview
                            sorted_items = sorted(models.items())[:20]
                            for alias, full_id in sorted_items:
                                # Normalize IDs for reliable role matching
                                clean_full_id = full_id.replace("models/", "")
                                roles = []
                                for rk, rv in self.config.config["models"].items():
                                    if rv.replace("models/", "") == clean_full_id:
                                        roles.append(rk.upper())
                                
                                status = f"★ {', '.join(roles)}" if roles else ""
                                table.add_row(alias, full_id, status)
                            model_tables.append(table)
                            
                        # Layout tables in columns
                        outer_grid.add_row(Columns(model_tables, equal=True, expand=True))
                        console.print(Panel(outer_grid, border_style="bright_blue", title="Model Management Hub"))
                        
                    elif base == "model":
                        if not arg:
                            console.print("[yellow]Usage: /model <alias> OR /model <role> <alias>[/yellow]")
                            console.print(f"[dim]Roles: coder, planner, architect, debugger, reviewer[/dim]")
                        else:
                            # Robust parsing for model setting
                            # Patterns: set <role> <alias>, <role> <alias>, <alias>
                            match_set = re.match(r'^(?:set\s+)?(coder|planner|architect|debugger|reviewer)\s+(.+)$', arg, re.IGNORECASE)
                            
                            # Update MODEL_MAP just in case dynamic models were loaded
                            m_catalog = load_dynamic_models()
                            m_map = {alias: id for cat in m_catalog.values() for alias, id in cat.items()}
                            m_ids = set(m_map.values())
                            
                            if match_set:
                                role = match_set.group(1).lower()
                                input_val = match_set.group(2).strip().strip('"').strip("'")
                                
                                target_id = m_map.get(input_val, input_val)
                                
                                # Validate: It's either a known alias, a known ID, or looks like a direct ID (has slash or dash)
                                if input_val in m_map or input_val in m_ids or "/" in input_val or "-" in input_val:
                                    self.config.config["models"][role] = target_id
                                    self.client.models = self.config.config["models"]
                                    self.config.save()
                                    display_name = input_val if input_val in m_map else target_id
                                    console.print(f"[bold green]✓[/bold green] Set [bold cyan]{role}[/bold cyan] role to [bold white]{display_name}[/bold white]")
                                else:
                                    console.print(f"[bold red]✗[/bold red] Unknown model alias or ID: [yellow]{input_val}[/yellow]")
                            else:
                                # Global switch
                                input_val = arg.strip().strip('"').strip("'")
                                target_id = m_map.get(input_val, input_val)
                                
                                if input_val in m_map or input_val in m_ids or "/" in input_val or "-" in input_val:
                                    for r in ["coder", "planner", "architect", "debugger", "reviewer"]:
                                        self.config.config["models"][r] = target_id
                                    self.client.models = self.config.config["models"]
                                    self.config.save()
                                    display_name = input_val if input_val in m_map else target_id
                                    console.print(f"[bold green]🚀 All roles switched to {display_name}[/bold green]")
                                else:
                                    console.print(f"[bold red]✗[/bold red] Unknown model alias or ID: [yellow]{input_val}[/yellow]")

                    elif base == "presets":
                        # Display Global Catalogue of all Mission Profiles
                        table = Table(title="🧠 Neural Mission Profiles (Global Catalogue)", border_style="bright_blue", box=box.ROUNDED)
                        table.add_column("Provider Hub", style="bold cyan", ratio=1)
                        table.add_column("ULTRA (Logic Max)", style="bold magenta", ratio=1)
                        table.add_column("CORE (Symmetrical)", style="bold white", ratio=1)
                        table.add_column("PULSE (Performance)", style="bold green", ratio=1)
                        
                        for p_id, presets in PROVIDER_PRESETS.items():
                            p_disp = PROVIDERS.get(p_id, {"name": p_id})["name"]
                            table.add_row(
                                p_disp,
                                f"[dim]{presets['Ultra']['coder'][:25]}...[/dim]",
                                f"[dim]{presets['Core']['coder'][:25]}...[/dim]",
                                f"[dim]{presets['Pulse']['coder'][:25]}...[/dim]"
                            )
                        console.print(table)
                        console.print("[dim]Use /preset <tier> to apply a profile to your active provider.[/dim]")

                    elif base == "preset":
                        valid_presets = ["Ultra", "Core", "Pulse"]
                        if not arg or arg.capitalize() not in valid_presets:
                            console.print(f"[yellow]Usage: /preset <Ultra|Core|Pulse>[/yellow]")
                        else:
                            preset = arg.capitalize()
                            p_name = self.client.provider
                            if p_name in PROVIDER_PRESETS and preset in PROVIDER_PRESETS[p_name]:
                                with console.status(f"[bold cyan]⚖️ Re-tuning mission profile to {preset}...[/bold cyan]"):
                                    new_models = PROVIDER_PRESETS[p_name][preset].copy()
                                    self.config.config["models"] = new_models
                                    self.config.config["active_preset"] = preset
                                    self.client.models = new_models
                                    
                                    # Update provider memory
                                    if "provider_models" not in self.config.config:
                                        self.config.config["provider_models"] = {}
                                    self.config.config["provider_models"][p_name] = new_models.copy()
                                    
                                    self.config.save()
                                    time.sleep(0.5)

                                res_table = Table(title=f"🎯 Neural Preset Applied: {preset}", border_style="bright_blue", box=box.ROUNDED)
                                res_table.add_column("Role", style="bold magenta")
                                res_table.add_column("Intelligence Unit (ID)", style="white")
                                for role, model in new_models.items():
                                    res_table.add_row(role.upper(), model)
                                console.print(res_table)
                                console.print(f"[bold green]✓ Mission profile synchronized for {p_name.upper()}.[/bold green]")
                            else:
                                console.print(f"[red]Preset {preset} not found for provider {p_name}[/red]")

                    elif base == "auto-models":
                        # Automatic intelligence-based model assignment
                        if self.client.provider == "google":
                            recommendations = {
                                "planner": ["gemini-2.5-flash", "gemini-3-flash", "gemini-2.0-thinking"],
                                "architect": ["gemini-2.5-flash", "gemini-3-flash", "gemini-1.5-pro"],
                                "coder": ["gemini-3-flash", "gemini-2.5-flash", "gemini-flash-latest"],
                                "debugger": ["gemini-2.5-flash", "gemini-3-flash", "gemini-1.5-pro"],
                                "reviewer": ["gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-1.5-flash"]
                            }
                        elif self.client.provider == "groq":

                            recommendations = {

                                "planner": ["llama-3.3-70b-versatile", "groq/compound", "meta-llama/llama-4-maverick-17b-128e-instruct"],

                                "architect": ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "groq/compound"],

                                "coder": ["llama-3.3-70b-versatile", "meta-llama/llama-4-maverick-17b-128e-instruct", "llama-3.1-8b-instant"],

                                "debugger": ["llama-3.1-8b-instant", "groq/compound-mini", "llama-3.3-70b-versatile"],

                                "reviewer": ["llama-3.1-8b-instant", "groq/compound-mini", "llama-3.3-70b-versatile"]

                            }

                        else:
                            recommendations = {
                                "planner": ["xiaomi/mimo-v2-flash:free", "google/gemini-2.0-flash-thinking-exp:free", "anthropic/claude-3.5-sonnet"],
                                "architect": ["xiaomi/mimo-v2-flash:free", "meta-llama/llama-3.3-70b-instruct:free", "anthropic/claude-3.5-sonnet"],
                                "coder": ["qwen/qwen3-coder:free", "mistralai/devstral-2512:free", "anthropic/claude-3.5-sonnet"],
                                "debugger": ["xiaomi/mimo-v2-flash:free", "qwen/qwen3-coder:free", "anthropic/claude-3.5-sonnet"],
                                "reviewer": ["nvidia/nemotron-3-nano-30b-a3b:free", "google/gemini-flash-1.5", "openai/gpt-4o-mini"]
                            }
                        
                        assignments = {}
                        with console.status("[bold magenta]🧠 Re-calibrating Neural Links...[/bold magenta]"):
                            for role, candidates in recommendations.items():
                                assigned = False
                                for cand in candidates:
                                    # Translate alias to full ID if needed
                                    full_id = MODEL_MAP.get(cand, cand)
                                    if self.client.provider == "google":
                                        # Simple check for google IDs
                                        self.config.config["models"][role] = full_id
                                        assignments[role] = full_id
                                        assigned = True
                                        break
                                    elif full_id in MODEL_MAP.values():
                                        self.config.config["models"][role] = full_id
                                        assignments[role] = full_id
                                        assigned = True
                                        break
                                if not assigned:
                                    fallback = "gemini-1.5-flash-8b" if self.client.provider == "google" else "mistralai/devstral-2512:free"
                                    self.config.config["models"][role] = fallback
                                    assignments[role] = fallback
                            
                            self.client.models = self.config.config["models"]
                            self.config.save()
                            time.sleep(1) # For that "busy" AI feel
                        
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
                        if self.client.provider == "google":
                            tier_map = {
                                "paid": {
                                    "planner": "gemini-2.0-thinking",
                                    "architect": "gemini-1.5-pro",
                                    "coder": "gemini-1.5-pro",
                                    "debugger": "gemini-1.5-pro",
                                    "reviewer": "gemini-2.5-flash"
                                },
                                "fullfree": {
                                    "planner": "gemini-2.5-flash",
                                    "architect": "gemini-2.5-flash",
                                    "coder": "gemini-2.0-flash",
                                    "debugger": "gemini-2.5-flash",
                                    "reviewer": "gemini-1.5-flash"
                                },
                                "freetier": {
                                    "planner": "gemini-flash-latest",
                                    "architect": "gemini-flash-latest",
                                    "coder": "gemini-2.5-flash-lite",
                                    "debugger": "gemini-2.5-flash-lite",
                                    "reviewer": "gemini-flash-lite-latest"
                                },
                                "extrafree": {
                                    "planner": "gemini-3-flash-preview",
                                    "architect": "gemini-2.5-flash",
                                    "coder": "gemini-3-flash-preview",
                                    "debugger": "gemini-2.5-flash",
                                    "reviewer": "gemini-2.5-flash-lite"
                                }
                            }
                        else:
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
                        to = self.config.get("tools", default={})
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

                    elif base == "calibrate":
                        console.print(Panel("[bold yellow]🚀 Initiating Neural Link Calibration...[/bold yellow]\nTesting active provider models for latency.", border_style="yellow"))
                        provider = self.config.get("active_provider")
                        catalog_file = f"{provider}_free_models.json" if provider != "groq" else "groq_models.json"
                        if not Path(catalog_file).exists():
                            catalog_file = f"{provider}_models.json"
                        
                        with console.status(f"[bold cyan]Auditing {provider} neural pathways...[/bold cyan]"):
                            report = await self.calibrator.audit_provider(provider, catalog_file)
                        
                        if "error" in report:
                            console.print(f"[bold red]Calibration Failed:[/bold red] {report['error']}")
                        else:
                            passes = [r for r in report["results"] if r["status"] == "PASS"]
                            passes.sort(key=lambda x: x["latency"])
                            
                            if passes:
                                best = passes[0]
                                console.print(f"✅ [bold green]Calibration Complete![/bold green] Fastest Neural Link: [cyan]{best['id']}[/cyan] ({best['latency']:.2f}s)")
                                if Confirm.ask("Do you want to auto-assign the fastest models to your current preset?"):
                                    for role in self.config.get("models", default={}):
                                        self.config.config["models"][role] = best['id']
                                    self.config.save()
                                    console.print("[bold cyan]🧠 Synchronizing Semantic Index...[/bold cyan]")
                                    await self.knowledge.build_semantic_index(self.client)
                                    console.print("[bold magenta]✨ Preset Optimized and Semantic Index Synced.[/bold magenta]")
                            else:
                                console.print("[bold red]Calibration Warning:[/bold red] No models passed the latency audit.")

                    elif base == "stats":
                        u = self.client.usage
                        s_table = Table(title="📊 Neural Resource Telemetry", border_style="cyan", box=box.ROUNDED)
                        s_table.add_column("Metric", style="cyan")
                        s_table.add_column("Value", style="bold white")
                        s_table.add_row("Prompt Tokens", f"{u['prompt_tokens']:,}")
                        s_table.add_row("Completion Tokens", f"{u['completion_tokens']:,}")
                        s_table.add_row("Total Context", f"{u['prompt_tokens'] + u['completion_tokens']:,}")
                        s_table.add_row("Est. Cost (USD)", f"${u['total_cost']:.4f}")
                        console.print(Panel(s_table, border_style="cyan", expand=False))

                    elif base == "sandbox":
                        if not self.sandbox_orig:
                            import tempfile
                            self.sandbox_orig = self.root
                            temp_dir = Path(tempfile.mkdtemp(prefix="omega_sandbox_"))
                            shutil.copytree(self.root, temp_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".omega", ".git", "logs"))
                            self.root = temp_dir
                            self.workspace.root = temp_dir
                            console.print(f"[bold yellow]🛡️ Sandbox Mode Active:[/bold yellow] Workspace isolated in {self.root}")
                        else:
                            self.root = self.sandbox_orig
                            self.workspace.root = self.sandbox_orig
                            self.sandbox_orig = None
                            console.print("[bold green]🛡️ Sandbox Discarded. Back to original workspace.[/bold green]")

                    elif base == "commit":
                        if self.sandbox_orig:
                            console.print("[bold yellow]🚀 Committing Sandbox changes to Main Workspace...[/bold yellow]")
                            shutil.copytree(self.root, self.sandbox_orig, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".omega", ".git", "logs"))
                            console.print("[bold green]✅ Commit Successful.[/bold green]")
                        else:
                            console.print("[red]Error: Not in a Sandbox session.[/red]")

                    elif base == "provider":
                        if not arg:
                            p_table = Table(title="🔌 Available AI Providers", border_style="magenta", box=box.ROUNDED)
                            p_table.add_column("ID", style="cyan")
                            p_table.add_column("Name", style="bold white")
                            p_table.add_column("Status", style="green")
                            p_table.add_column("Key Env", style="dim")

                            for pid, info in PROVIDERS.items():
                                status = "ACTIVE" if pid == self.client.provider else ""
                                has_key = "✓" if os.getenv(info["env_key"]) else "MISSING"
                                p_table.add_row(pid, info["name"], f"[bold green]{status}[/bold green]" if status else f"[dim]{has_key}[/dim]", info["env_key"])
                            
                            console.print(p_table)
                            console.print("\n[dim]Usage: /provider <id> (e.g., /provider google)[/dim]")
                        else:
                            target = arg.lower().strip()
                            if target in PROVIDERS:
                                if self.setup_client(target):
                                    console.print(f"[bold green]Successfully switched to {PROVIDERS[target]['name']}.[/bold green]")
                                else:
                                    console.print(f"[red]Failed to switch to {target}. API Key required.[/red]")
                            else:
                                console.print(f"[red]Unknown provider: {target}[/red]")

                    elif base in ["help", "menu"]:
                        if arg:
                            self.show_command_help(arg)
                        else:
                            self.show_dashboard_menu()
                    else: console.print(f"[red]Unknown command: {base}[/red]")
                else:
                    if len(cmd) > 5: await self.process_task(cmd)
            except KeyboardInterrupt: continue
            except EOFError: break
            except Exception as e:
                console.print(f"[bold red]System Exception:[/bold red] {e}")
                self.logger.exception("Main Loop Crash", e)
                import traceback
                if self.config.get("features", "verbose", default=False): traceback.print_exc()

if __name__ == "__main__":
    ai = OmegaAi()
    try:
        asyncio.run(ai.main_loop())
    except KeyboardInterrupt:
        pass
