# TinyCoder VS Code Extension – Development Plan

## Overview
TinyCoder is a **local‑inference AI coding assistant** delivered as a VS Code extension. It loads **GGUF** models directly from the filesystem, runs inference on‑device, and provides a convenient UI for developers:
- **Sidebar panel** for chat interaction
- **Command‑palette commands** for explain, complete, and generate code
- **Status‑bar item** showing model status and quick actions
- **Terminal command** that forwards selected text to the model and streams the response back to the integrated terminal

All common inference parameters are configurable: **temperature, max‑tokens, top‑p, repeat‑penalty**.

---

## Feature Set
| Feature | Description |
|---|---|
| **Model loading** | Load a single *.gguf* model via a file‑picker. Only one model is active at a time. |
| **Sidebar chat** | Webview panel (see [`src/ts/panel.ts`](src/ts/panel.ts:1)) for interactive messaging. |
| **Command‑palette actions** | `tinycoder.explainCode`, `tinycoder.completeCode`, `tinycoder.generateCode`, `tinycoder.loadModel`. |
| **Status‑bar item** | Shows *model loaded* / *inference ready* and allows quick reload. |
| **Inference settings UI** | Settings contribution (`package.json`) with inputs for temperature, max‑tokens, top‑p, repeat‑penalty. |
| **Terminal integration** | New terminal command `TinyCoder: Infer Selected Text` that reads the current selection, sends it to the native bridge, and streams output to the terminal. |
| **Native bridge** | C++/CUDA backend (see [`src/cpp/bridge/*.cpp`](src/cpp/bridge/Bridge.cpp:1)) exposing `loadModel`, `runInference`. |

---

## Architecture
```mermaid
flowchart LR
    subgraph VSCode
        UI[Sidebar + Cmd Palette + Status Bar]
        Terminal[Terminal Command]
    end
    subgraph Backend
        Bridge[Native Bridge (C++)]
        Model[GGUF Model
        (CPU / CUDA)]
    end
    UI -->|via nativeBridge.ts| Bridge
    Terminal -->|via nativeBridge.ts| Bridge
    Bridge -->|loads| Model
    Bridge -->|returns inference| UI & Terminal
```

- **UI layer** (`src/ts/*.ts`) communicates with the **native bridge** (`src/ts/nativeBridge.ts`).
- The bridge forwards calls to the C++ library (`src/cpp/bridge/Bridge.cpp`).
- Inference runs on the CPU or CUDA (if available) using the GGUF model.

---

## Implementation Steps (High‑Level)
1. **Project scaffolding**
   - Add `plans/` directory, create `tinycoder_plan.md` (this file).
   - Update `package.json` with new commands, contributions, and activation events.
2. **Native bridge extension**
   - Define TypeScript API in `src/ts/nativeBridge.ts` for `loadModel`, `runInference`, `getStatus`.
   - Implement C++ side (`Bridge.cpp`) to call existing model code (`Model.cpp`).
3. **UI components**
   - **Sidebar panel** – refine `src/ts/panel.ts` layout, add settings panel.
   - **Status‑bar item** – register in `extension.ts`, bind to model status.
   - **Command‑palette commands** – ensure they invoke the panel with appropriate prompts.
4. **Settings UI**
   - Add configuration schema in `package.json` for `tinycoder.temperature`, `tinycoder.maxTokens`, `tinycoder.topP`, `tinycoder.repeatPenalty`.
   - Expose a quick‑settings webview inside the sidebar.
5. **Terminal command**
   - Register `tinycoder.inferFromTerminal` command.
   - When executed, read the active terminal selection (via VS Code API), send to native bridge, and stream output back using `Terminal.write`.
6. **Model loading flow**
   - Restrict file picker to `*.gguf` only.
   - Store chosen path in extension state, validate file existence.
7. **Error handling & UX polish**
   - Show notifications for load failures, inference errors.
   - Add spinner/loading indicator in the sidebar.
8. **Testing**
   - Unit tests for TypeScript bridge (`src/ts/*.test.ts`).
   - Integration tests launching the extension in a headless VS Code instance.
9. **Documentation**
   - Update `README.md` with usage guide, screenshots, and supported model format.
   - Add developer guide for building native binaries.
10. **Packaging & Release**
    - Adjust `vsce` packaging scripts.
    - Publish to the VS Code Marketplace.

---

## Milestones
| Milestone | Deliverable |
|---|---|
| **M1 – Core scaffolding** | Commands, config, basic UI skeleton |
| **M2 – Native bridge** | Load *.gguf* model, run a dummy inference |
| **M3 – Full UI + Settings** | Sidebar, status bar, settings UI |
| **M4 – Terminal integration** | `TinyCoder: Infer Selected Text` works |
| **M5 – Testing & Docs** | Unit tests, README updates |
| **M6 – Release** | Packaged extension ready for marketplace |

---

*This plan will be tracked with a detailed todo list that can be executed by the Code mode.*
