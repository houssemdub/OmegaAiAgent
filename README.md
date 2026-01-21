# 🌌 OmegaAi: The Ultimate Autonomous Coding Assistant

**OmegaAi** is a high-fidelity, autonomous development engine engineered to transform high-level "vibes" into production-grade code. By combining a tiered multi-agent architecture with long-term project persistence and surgical code editing, OmegaAi operates as a senior engineer in your terminal.

---

## ✨ Key Features

### 🧠 Triple-Phase Autonomous Pipeline
OmegaAi doesn't just "chat"—it builds. Every task follows a rigorous engineering lifecycle:
1.  **Thinking (Architect)**: Analyzes the workspace, builds a RAG index, and produces a technical blueprint with estimated iteration budgets.
2.  **Building (Developer)**: Implements the plan using a specialized Coder model, leveraging `<patch>` tools for surgical updates and `<search>` for real-world data.
3.  **Optimizing (Debugger)**: Conducts a final audit, runs the code, and verifies the implementation against the original intent.

### 🧰 The Advanced Toolbox (`/tools`)
*   **Surgical Patching**: Modify 1,000-line files without re-writing them. Uses `<patch>` with search/replace blocks to save tokens and prevent errors.
*   **Project Persistence**: A long-term "Knowledge Base" (`.omega/knowledge.md`) that remembers your UI preferences, system constraints, and past decisions.
*   **Context & RAG**: Automatically indexes your local codebase into a semantic map (`rag.json`) for instant context retrieval.
*   **Web Search**: Integrated DuckDuckGo search for fetching real-time API docs, weather, and technical facts.
*   **Visual Auditor**: Built-in visual engineering prompt that audits UI/UX harmony and aesthetic excellence.

### 🚀 Industrial Model Management (`/models`)
*   **Tiered Ecosystem**: Instantly switch between `paid`, `fullfree`, or `extrafree` tiers.
*   **Auto-Optimization**: `/auto-models` uses a built-in recommendation engine to assign the most powerful "Survivor" models (like MiMo-V2 and Qwen3 Coder) to specific roles.
*   **Role-Based Logic**: Assign different models to be your Architect vs. your Coder to maximize performance vs. cost.

### ⌨️ Professional Developer UX
*   **Intuition Engine**: Ghost-suggestions as you type (Right Arrow/End to complete).
*   **Lightning Completion**: Tab-autocompletion for all system commands.
*   **Unified History**: Navigable and searchable command history via `/history`.
*   **Wide-Format Dashboard**: A clean, left-aligned "Big Menu" aesthetic designed for readability in large terminals.

---

## 🛠️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/OmegaAiAgent.git
    cd OmegaAiAgent
    ```

2.  **Install Requirements**:
    ```bash
    pip install rich httpx aiofiles prompt_toolkit python-dotenv duckduckgo_search
    ```

3.  **Set Up API Key**:
    Create a `.env` file in the root directory:
    ```env
    OPENROUTER_API_KEY=your_key_here
    ```

---

## 🎮 Walkthrough: Your First Task

1.  **Initialize**: Launch the engine:
    ```bash
    python OmegaAi.py
    ```
2.  **Optimize Models**: Run `/auto-models` to ensure you are using the best possible free or paid tier.
3.  **The First Vibe**: Tell OmegaAi to build something:
    ```bash
    /vibe Search for the real weather in Algiers and save it to a gorgeous index.html
    ```
4.  **Watch the Magic**: 
    *   **Architect** will plan the search and HTML layout.
    *   **Developer** will fetch real data and write the code.
    *   **Debugger** will verify the 16-day forecast.
5.  **Audit**: Use `/tree` to see your new files and `/history` to review your session.

---

## 🧭 System Commands

| Command | Description |
| :--- | :--- |
| `/vibe <task>` | Launch the autonomous development loop. |
| `/tools` | Manage advanced tools (Search, RAG, Persistence). |
| `/models` | Open the Model Management Hub. |
| `/auto-models` | Automatically assign the best models to roles. |
| `/models-tier` | Switch entire system tier (Paid/Free/ExtraFree). |
| `/history` | Search and view command history. |
| `/tree` | Visualize the current workspace structure. |
| `/undo` | Roll back to the previous iteration snapshot. |
| `/exit` | Gracefully shut down the agent. |

---

## 🛡️ Reliability & Safety
*   **Windows Hardened**: Includes recursive permission handlers to fix the infamous "Access Denied" errors during file operations.
*   **Iteration Budgeting**: Prevents runaway loops by asking the Architect to estimate the "energy cost" of a task before starting.
*   **Smart Circuit Breaker**: Detects repetitive writes and API errors (401/429) to halt the process before wasting credits.

---

**OmegaAi** — *Code at the speed of thought.*