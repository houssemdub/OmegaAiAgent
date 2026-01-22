# 📖 OmegaAi Technical Documentation

Welcome to the full documentation for **OmegaAi**, the multi-provider autonomous coding engine.

---

## 🏗️ Architecture Overview

OmegaAi is built on a modular "Neural Gateway" architecture. Instead of being locked to a single AI model, it uses a provider-agnostic core that translates internal tool requests into provider-specific API calls.

### 👥 The Multi-Agent System
*   **The Architect (Thinking Phase)**:
    *   **Goal**: Create a technical specification and implementation plan.
    *   **Context**: Receives a full directory tree and any existing project knowledge.
    *   **Output**: Technical blueprint and estimated iteration budget.
*   **The Developer (Building Phase)**:
    *   **Goal**: Implement the Architect's plan.
    *   **Capability**: Can read, write, patch, run commands, and search the web.
    *   **Persistence**: Operates in a stateful loop until the task is marked complete.
*   **The Debugger (Verification Phase)**:
    *   **Goal**: Audit the implementation.
    *   **Action**: Runs test scripts or verifies the visual look of the code.

---

## 🛠️ Advanced Tool System

### 🧬 Surgical Patching (`<patch>`)
Unlike standard LLM agents that rewrite entire files, OmegaAi uses a precise patching system.
```xml
<patch path="app/api/route.ts">
  <search>old_code_block</search>
  <replace>new_optimized_logic</replace>
</patch>
```
*   **Benefit**: Saves context tokens and prevents accidental deletion of unrelated code in large files.

### 📂 Stateful Shell Navigation
OmegaAi supports persistent `cd` commands. If the agent runs `<run>mkdir src && cd src</run>`, all subsequent tools (read/write/run) will be executed relative to the `src` folder. This is managed via an internal state tracker that resolves paths against the dynamic project root.

### 🧠 Project Knowledge (`.omega/knowledge.md`)
The engine maintains a long-term memory of project-specific details:
*   Standard ports used.
*   Preferred styling libraries (e.g., Tailwind).
*   API keys or endpoint structures.
*   "Insights" captured during previous autonomous runs.

---

## 🚀 Provider & Model Management

### 🔌 Supported Gateways
1.  **OpenRouter**: The primary gateway for a wide variety of models.
2.  **Google AI Studio**: Native integration for Gemini models, offering the largest context windows (up to 2M tokens).
3.  **Groq**: Dedicated LPU inference for sub-second responses.

### 💾 Stateful Model Memory
The engine remembers model assignments per provider. Switching from `OpenRouter` to `Groq` will restore your last used Groq models for each agent role.

### 🏥 Self-Healing Dependencies
OmegaAi detects missing Python libraries on launch and offers to install them automatically using `pip`. It maintains a `REQUIRED_LIBS` manifest to ensure the environment is always ready for autonomous work.

### 🛰️ Neural Presets (High-Level Tuning)
The engine provides pre-calibrated mission profiles that match the best available models to their ideal engineering roles:
*   **Ultra**: Elite models for maximum logic (e.g., Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro).
*   **Core**: Balanced models for efficiency and capability (e.g., Llama 3.3 70B, DeepSeek Coder).
*   **Pulse**: High-speed models for rapid iteration and testing (e.g., Llama 3.1 8B, Gemini Flash).
Profiles are provider-aware and optimized based on real-world competency audits.

---

## ⌨️ Command Reference

| Command | Usage | Detailed Effect |
| :--- | :--- | :--- |
| `/vibe` | `/vibe build a news app` | Triggers the Architect -> Developer -> Debugger loop. |
| `/presets` | `/presets` | Displays the global catalogue of neural mission profiles. |
| `/preset` | `/preset Ultra` | Applies a specific mission profile tier to the active provider. |
| `/auto-models`| `/auto-models` | Scans the active provider and assigns the most capable models to roles. |
| `/undo` | `/undo` | Reverts the last file changes using the `.omega/backups` system. |
| `/tree` | `/tree 2` | Visualizes the project structure up to a specific depth. |
| `/provider` | `/provider groq` | Switches the active API gateway. |
| `/models-tier`| `/models-tier paid` | Filters the catalogue to specific pricing tiers. |

---

## ⚡ Performance Optimization

To get the best results from OmegaAi:
1.  **Use Groq for iteration**: The speed of Groq makes the feedback loop almost instantaneous.
2.  **Use Gemini for large contexts**: If you're refactoring a massive codebase, switch to `google` provider to leverage Gemini's massive input capacity.
3.  **Run `/auto-models` after switching providers**: Ensures the best "brain" is assigned to each stage of the pipeline.

---

**OmegaAi** — *The next evolution of autonomous engineering.*
