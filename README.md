# 🌌 OmegaAi: The Ultimate Autonomous Coding Assistant (V5.0)

**OmegaAi** is a high-fidelity, autonomous development engine engineered to transform high-level "vibes" into production-grade code. By combining a tiered multi-agent architecture with long-term project persistence and surgical code editing, OmegaAi operates as a senior engineer in your terminal.

---

## ✨ Key Features

### 🔌 Multi-Provider "Neural" Core
OmegaAi is a multi-gateway engine, allowing you to hot-swap between different AI backends seamlessly:
*   **OpenRouter Integration**: Access to 100+ models via a single API (Claude, GPT, Llama, DeepSeek).
*   **Native Google AI Studio (Gemini)**: Direct integration via the **`google-genai` SDK**, supporting the latest Gemini 2.0/3.0 Flash and "Thinking" models.
*   **Groq LPU (Ultra-Fast)**: Hyper-speed inference (250-500+ tokens/sec) using Groq's LPU architecture.
*   **Stateful Model Memory**: Each provider remembers its specific model assignments (e.g., your preferred Groq models are restored when you switch back from Google).
*   **Verified Groq Core**: Only neural core competency tested models are enabled for Groq provider.

### 🛰️ Neural Mission Profiles (V5.0)
*   **Instant Calibration**: Switch between **Ultra** (High Intelligence), **Core** (Balanced), and **Pulse** (Performance) presets optimized for your active provider.
*   **Global Catalogue**: View the entire mission profile library across Google, Groq, and OpenRouter with one command.

### 🧠 Triple-Phase Autonomous Pipeline
Every task follows a rigorous engineering lifecycle:
1.  **Thinking (Architect)**: Analyzes the workspace, builds a RAG index, and produces a technical blueprint.
2.  **Building (Developer)**: Implements the plan using a specialized Coder model, leveraging `<patch>` tools for surgical updates.
3.  **Optimizing (Debugger)**: Conducts a final audit, runs the code, and verifies the implementation.
4.  **Neural Identity**: The iteration header now displays the specific model ID used for each phase, ensuring full transparency.

### 📂 Proactive Project Architecting
*   **Smart Detection**: Automatically detects when you want to start a new project and creates the folder structure for you.
*   **Stateful Shell Navigation**: Integrated virtual `cd` support. The agent tracks its position across iterations, allowing complex multi-directory workflows.

### 🍱 Premium Dashboard UI (V5.0)
*   **High-Density Telemetry**: A 3-column real-time dashboard tracking Role Assignments, Infrastructure Meta, and Project State.
*   **Neural Healer Integration**: Real-time feedback when the self-healing system intercepts an error.
*   **Illuminated Toolbox**: Dedicated status indicators for hardware tools (Search, RAG, Persistence, Vision).
*   **Quick Command Dock**: Instant access to primary engine operations.
*   **Anti-Greedy Tool Parsing**: Robust regex engine ensures tool calls are parsed with high precision even when mentioned in conversation.

### 🧰 The Advanced Toolbox (`/tools`)
*   **Surgical Patching**: Modify large files without re-writing them using search/replace blocks.
*   **Project Persistence**: A long-term "Knowledge Base" (`.omega/knowledge.md`) for UI preferences and constraints.
*   **Context & RAG**: Automatically indexes your local codebase into a semantic map.
*   **Web Search**: Integrated `ddgs` (DuckDuckGo) search for fetching real-time facts and documentation.

---

## 🚀 Industrial Model Hub (`/models`)

OmegaAi features a provider-aware model ecosystem:
*   **Tiered Optimization**: Instantly switch between `paid`, `fullfree`, or the verified `extrafree` tier.
*   **Survivor Models**: `/auto-models` uses verified audit data to pick the fastest and most reliable models for each provider.
*   **Groq Velocity**: Experience the power of Llama 3.3 and Llama 4 at hardware speeds.

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
    pip install rich httpx aiofiles prompt_toolkit python-dotenv ddgs google-genai
    ```

3.  **Configuration**:
    Create a `.env` file in the root directory:
    ```env
    OPENROUTER_API_KEY=sk-or-v1-your-key
    GOOGLE_API_KEY=your-google-api-key
    GROQ_API_KEY=your-groq-api-key
    ```

---

## 🎮 Walkthrough: Your First Task

1.  **Launch**: `python OmegaAi.py`
2.  **Hot-Switching**: Use `/provider` to jump between backends without session loss.
*   **Neural Healer (Self-Healing)**: Automatically analyzes shell errors and suggests fixes.
*   **Native Vision (Multi-Modal)**: Use `<read_vision>` to analyze images or audit UI designs.
3.  **Optimize**: Run `/auto-models` to calibrate your neural links.
4.  **The Vibe**: Tell OmegaAi to build something:
    ```bash
    /vibe Create a real-time portfolio with glassmorphism using Next.js.
    ```

---

## 🧭 System Commands

| Command | Description | Contextual Help |
| :--- | :--- | :--- |
| `/vibe <task>`| Launch the autonomous development loop. | `/help vibe` |
| `/presets` | View global mission profiles catalogue. | `/help presets` |
| `/preset <tier>`| Apply Ultra, Core, or Pulse profiles. | `/help preset` |
| `/provider` | Manage AI Gateway providers (OR/Google/Groq). | `/help provider` |
| `/models` | Query available intelligence units for active provider. | `/help models` |
| `/models-tier`| Switch entire system tier (Paid/Free/Extra). | `/help models-tier` |
| `/auto-models`| Automatically assign the best models. | `/help auto-models` |
| `/tools` | Manage advanced tools (Search, RAG, Vision). | `/help tools` |
| `/menu` | Open the premium command dashboard. | `/help menu` |
| `/undo` | Roll back to the previous iteration. | `/help undo` |

---

## ⚡ Groq Integration (LPU Powered)
OmegaAi now harnesses the power of **Groq's LPU™ Inference Engine**.
*   **Speed**: ~250-500+ tokens/second.
*   **Efficiency**: Instant code generation and debugging.
*   **Intelligence**: Optimized for Llama 3.3 (70B) and Llama 4 Scout/Maverick.
*   **Verified**: Only models passing the core competency audit are enabled.

See [DOC.md](DOC.md) for full system documentation.

---

**OmegaAi** — *Multi-Provider Autonomous Engineering.*
