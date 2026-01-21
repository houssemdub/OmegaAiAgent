# 🌌 OmegaAi: The Ultimate Autonomous Coding Assistant

**OmegaAi** is a high-fidelity, autonomous development engine engineered to transform high-level "vibes" into production-grade code. By combining a tiered multi-agent architecture with long-term project persistence and surgical code editing, OmegaAi operates as a senior engineer in your terminal.

---

## ✨ Key Features

### 🔌 Multi-Provider "Neural" Core
OmegaAi is now a multi-gateway engine, allowing you to hot-swap between different AI backends seamlessly:
*   **OpenRouter Integration**: Access to 100+ models via a single API (Claude, GPT, Llama, DeepSeek).
*   **Native Google AI Studio (Gemini)**: Direct integration via the **`google-genai` SDK**, supporting the latest Gemini 2.0 Flash, 2.5 Flash, and "Thinking" models with industry-leading speed.
*   **Hot-Switching**: Use `/provider` to jump between backends without session loss.

### 🧠 Triple-Phase Autonomous Pipeline
Every task follows a rigorous engineering lifecycle:
1.  **Thinking (Architect)**: Analyzes the workspace, builds a RAG index, and produces a technical blueprint with estimated iteration budgets.
2.  **Building (Developer)**: Implements the plan using a specialized Coder model, leveraging `<patch>` tools for surgical updates and `<search>` for real-world data.
3.  **Optimizing (Debugger)**: Conducts a final audit, runs the code, and verifies the implementation against the original intent.

### 🍱 Premium Dashboard UI (V4.0)
*   **High-Density Telemetry**: A 3-column real-time dashboard tracking Role Assignments, Infrastructure Meta (OS/Arch/Provider), and Project State (Iterations/Backups).
*   **Illuminated Toolbox**: Dedicated status indicators for hardware tools (Search, RAG, Persistence, Vision).
*   **Quick Command Dock**: Instant access to primary engine operations.

### 🧰 The Advanced Toolbox (`/tools`)
*   **Surgical Patching**: Modify 1,000-line files without re-writing them. Uses `<patch>` with search/replace blocks.
*   **Project Persistence**: A long-term "Knowledge Base" (`.omega/knowledge.md`) that remembers your UI preferences and system constraints.
*   **Context & RAG**: Automatically indexes your local codebase into a semantic map (`rag.json`).
*   **Web Search**: Integrated DuckDuckGo search for fetching real-time API docs and technical facts.

---

## 🚀 Industrial Model Hub (`/models`)

OmegaAi features a provider-aware model ecosystem:
*   **Tiered Optimization**: Instantly switch between `paid`, `fullfree`, or the verified `extrafree` tier.
*   **Survivor Models**: `/auto-models` uses verified audit data to pick the fastest and most reliable models for each provider (e.g., Gemini 2.5 Flash for Google, Qwen3 Coder for OpenRouter).
*   **Performance Auditing**: Built-in `test_google_models.py` utility for real-time latency and connectivity checks.

---

## 🛡️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/houssemdub/OmegaAiAgent.git
    cd OmegaAiAgent
    ```

2.  **Install Requirements**:
    The engine will **automatically** install missing libs on startup, but you can pre-install:
    ```bash
    pip install rich httpx aiofiles prompt_toolkit python-dotenv duckduckgo_search google-genai
    ```

3.  **Configuration**:
    Create a `.env` file in the root directory:
    ```env
    OPENROUTER_API_KEY=sk-or-v1-your-key
    GOOGLE_API_KEY=your-google-api-key
    ```

---

## 🎮 Walkthrough: Your First Task

1.  **Launch**: `python OmegaAi.py`
2.  **Switch Provider**: Type `/provider google` to use Gemini models.
3.  **Optimize**: Run `/auto-models` to calibrate your neural links.
4.  **The Vibe**: Tell OmegaAi to build something:
    ```bash
    /vibe Build a React dashboard with glassmorphism and real-time weather stats.
    ```
5.  **Check Status**: Use `/menu` to see the dashboard and `/tree` for files.

---

## 🧭 System Commands

| Command | Description | Contextual Help |
| :--- | :--- | :--- |
| `/vibe <task>`| Launch the autonomous development loop. | `/help vibe` |
| `/provider` | Manage AI Gateway providers (Google/OR). | `/help provider` |
| `/models-tier`| Switch entire system tier (Paid/Free/Extra). | `/help models-tier` |
| `/auto-models`| Automatically assign the best models. | `/help auto-models` |
| `/tools` | Manage advanced tools (Search, RAG, Vision). | `/help tools` |
| `/menu` | Open the premium command dashboard. | `/help menu` |
| `/history` | Search and view command history. | `/help history` |
| `/undo` | Roll back to the previous iteration. | `/help undo` |

---

## 🛡️ Reliability & Safety
*   **Contextual Help System**: Type `/help <command>` for deep-dive documentation and usage examples.
*   **Nested Autocompletion**: Advanced TAB-completion for multi-level commands.
*   **Smart Circuit Breaker**: Detects repetitive writes and API errors (401/429/404) to halt before wasting resources.

---

**OmegaAi** — *Multi-Provider Autonomous Engineering.*