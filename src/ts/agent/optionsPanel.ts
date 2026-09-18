/*
⚡ TinyCoder AI - Agent Harness Options Panel

Copyright (c) 2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

import * as vscode from 'vscode';
import { AgentConfig } from './types';
import { loadAgentConfig, saveAgentConfig } from './config';

/**
 * Options/settings panel for the agent harness.
 *
 * Edits the same VS Code settings the config loader reads, so changes
 * persist across sessions. Reads the current config on open and posts it
 * to the webview; accepts `save` messages back.
 */
export class AgentOptionsPanel {
    private readonly panel: vscode.WebviewPanel;
    private readonly disposables: vscode.Disposable[] = [];

    constructor(private readonly context: vscode.ExtensionContext) {
        this.panel = vscode.window.createWebviewPanel(
            'tinycoder.agentOptions',
            'TinyCoder Agent Options',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );
        this.panel.webview.html = this.renderHtml();
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        this.panel.webview.onDidReceiveMessage(
            (msg) => this.handleMessage(msg),
            null,
            this.disposables
        );
    }

    private dispose(): void {
        for (const d of this.disposables) {
            d.dispose();
        }
        this.disposables.length = 0;
    }

    private async handleMessage(msg: any): Promise<void> {
        switch (msg?.type) {
            case 'ready':
                this.postConfig();
                break;
            case 'save':
                await this.handleSave(msg.config);
                break;
            case 'browseModel':
                await this.handleBrowseModel();
                break;
            case 'browseWorkspace':
                await this.handleBrowseWorkspace();
                break;
        }
    }

    private postConfig(): void {
        const config = loadAgentConfig();
        this.panel.webview.postMessage({ type: 'config', config });
    }

    private async handleSave(cfg: any): Promise<void> {
        try {
            const current = loadAgentConfig();
            const next: AgentConfig = {
                ...current,
                modelPath: typeof cfg?.modelPath === 'string' ? cfg.modelPath : current.modelPath,
                workspaceRoot: typeof cfg?.workspaceRoot === 'string' ? cfg.workspaceRoot : current.workspaceRoot,
                terminalName: typeof cfg?.terminalName === 'string' ? cfg.terminalName : current.terminalName,
                terminalMode: cfg?.terminalMode === 'vscode' || cfg?.terminalMode === 'child' || cfg?.terminalMode === 'auto'
                    ? cfg.terminalMode
                    : current.terminalMode,
                maxIterations: typeof cfg?.maxIterations === 'number' ? cfg.maxIterations : current.maxIterations,
                compactionThreshold: typeof cfg?.compactionThreshold === 'number' ? cfg.compactionThreshold : current.compactionThreshold,
                compactionPreserveRecent: typeof cfg?.compactionPreserveRecent === 'number' ? cfg.compactionPreserveRecent : current.compactionPreserveRecent,
                toolToggles: {
                    readFile: typeof cfg?.toolToggles?.readFile === 'boolean' ? cfg.toolToggles.readFile : current.toolToggles.readFile,
                    writeFile: typeof cfg?.toolToggles?.writeFile === 'boolean' ? cfg.toolToggles.writeFile : current.toolToggles.writeFile,
                    patchFile: typeof cfg?.toolToggles?.patchFile === 'boolean' ? cfg.toolToggles.patchFile : current.toolToggles.patchFile,
                    terminal: typeof cfg?.toolToggles?.terminal === 'boolean' ? cfg.toolToggles.terminal : current.toolToggles.terminal,
                    search: typeof cfg?.toolToggles?.search === 'boolean' ? cfg.toolToggles.search : current.toolToggles.search
                },
                inference: {
                    ...current.inference,
                    maxTokens: typeof cfg?.inference?.maxTokens === 'number' ? cfg.inference.maxTokens : current.inference.maxTokens,
                    temperature: typeof cfg?.inference?.temperature === 'number' ? cfg.inference.temperature : current.inference.temperature,
                    topP: typeof cfg?.inference?.topP === 'number' ? cfg.inference.topP : current.inference.topP,
                    topK: typeof cfg?.inference?.topK === 'number' ? cfg.inference.topK : current.inference.topK,
                    repeatPenalty: typeof cfg?.inference?.repeatPenalty === 'number' ? cfg.inference.repeatPenalty : current.inference.repeatPenalty,
                    repeatLastN: typeof cfg?.inference?.repeatLastN === 'number' ? cfg.inference.repeatLastN : current.inference.repeatLastN,
                    maxSeqLen: typeof cfg?.inference?.maxSeqLen === 'number' ? cfg.inference.maxSeqLen : current.inference.maxSeqLen,
                    nThreads: typeof cfg?.inference?.nThreads === 'number' ? cfg.inference.nThreads : current.inference.nThreads
                }
            };
            await saveAgentConfig(next);
            vscode.window.showInformationMessage('TinyCoder agent options saved.');
        } catch (err: any) {
            vscode.window.showErrorMessage(`Failed to save options: ${err?.message || err}`);
        }
    }

    private async handleBrowseModel(): Promise<void> {
        const picked = await vscode.window.showOpenDialog({
            canSelectMany: false,
            openLabel: 'Select GGUF Model',
            filters: { 'GGUF Models': ['gguf'], 'All Files': ['*'] }
        });
        if (picked && picked[0]) {
            this.panel.webview.postMessage({ type: 'modelPicked', path: picked[0].fsPath });
        }
    }

    private async handleBrowseWorkspace(): Promise<void> {
        const picked = await vscode.window.showOpenDialog({
            canSelectMany: false,
            canSelectFolders: true,
            openLabel: 'Select Workspace Root'
        });
        if (picked && picked[0]) {
            this.panel.webview.postMessage({ type: 'workspacePicked', path: picked[0].fsPath });
        }
    }

    private renderHtml(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    :root {
        --bg: #1e1e2e;
        --panel: #181825;
        --border: #45475a;
        --text: #cdd6f4;
        --muted: #a6adc8;
        --accent: #89b4fa;
        --accent2: #74c7ec;
        --input-bg: #313244;
        --error: #f38ba8;
        --ok: #a6e3a1;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--bg);
        color: var(--text);
        padding: 16px;
        font-size: 13px;
    }
    h2 { font-size: 15px; margin-bottom: 4px; color: var(--accent); }
    .section {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 16px;
    }
    .section-title { font-weight: 600; margin-bottom: 8px; color: var(--accent2); }
    .row { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; flex-wrap: wrap; }
    label { min-width: 150px; color: var(--muted); }
    input[type="text"], input[type="number"], select {
        flex: 1;
        min-width: 160px;
        max-width: 420px;
        background: var(--input-bg);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 6px 8px;
        font-size: 13px;
        font-family: inherit;
    }
    input[type="checkbox"] { transform: scale(1.2); accent-color: var(--accent); }
    .checkbox-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
    .checkbox-row label { min-width: auto; margin-right: 8px; }
    .hint { font-size: 11px; color: var(--muted); margin-top: -4px; margin-bottom: 8px; }
    button {
        background: var(--accent);
        color: #11111b;
        border: none;
        border-radius: 6px;
        padding: 8px 14px;
        font-weight: 600;
        cursor: pointer;
        font-size: 13px;
    }
    button.secondary { background: var(--input-bg); color: var(--text); border: 1px solid var(--border); }
    .actions { display: flex; gap: 8px; }
    .status { margin-top: 10px; font-size: 12px; min-height: 18px; }
    .status.ok { color: var(--ok); }
    .status.err { color: var(--error); }
    .inline-btn { padding: 4px 8px; font-size: 11px; border-radius: 4px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); cursor: pointer; }
</style>
</head>
<body>
    <h2>TinyCoder Agent Options</h2>
    <p class="hint">Settings are persisted to VS Code configuration (tinycoder.*).</p>

    <div class="section">
        <div class="section-title">Model</div>
        <div class="row">
            <label for="modelPath">Model path (.gguf)</label>
            <input id="modelPath" type="text" placeholder="/data/models/qwen/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf">
            <button class="inline-btn" id="browseModel">Browse…</button>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Inference Config</div>
        <div class="row">
            <label for="maxTokens">Max tokens</label>
            <input id="maxTokens" type="number" min="64" max="32768" value="2048">
        </div>
        <div class="row">
            <label for="temperature">Temperature</label>
            <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.7">
        </div>
        <div class="row">
            <label for="topP">Top-P</label>
            <input id="topP" type="number" min="0" max="1" step="0.05" value="0.9">
        </div>
        <div class="row">
            <label for="topK">Top-K</label>
            <input id="topK" type="number" min="0" max="200" value="40">
        </div>
        <div class="row">
            <label for="repeatPenalty">Repeat penalty</label>
            <input id="repeatPenalty" type="number" min="1" max="2" step="0.05" value="1.1">
        </div>
        <div class="row">
            <label for="maxSeqLen">Max sequence length</label>
            <input id="maxSeqLen" type="number" min="512" max="32768" value="8192">
        </div>
        <div class="row">
            <label for="nThreads">CPU threads (0=auto)</label>
            <input id="nThreads" type="number" min="0" max="64" value="0">
        </div>
    </div>

    <div class="section">
        <div class="section-title">Agent Config</div>
        <div class="row">
            <label for="workspaceRoot">Workspace root</label>
            <input id="workspaceRoot" type="text" placeholder="(default: first workspace folder)">
            <button class="inline-btn" id="browseWorkspace">Browse…</button>
        </div>
        <div class="row">
            <label for="terminalName">Terminal name</label>
            <input id="terminalName" type="text" value="TinyCoder Agent">
        </div>
        <div class="row">
            <label for="terminalMode">Terminal mode</label>
            <select id="terminalMode">
                <option value="auto">auto (prefer child, fall back to VS Code)</option>
                <option value="vscode">vscode (createTerminal)</option>
                <option value="child">child (subprocess)</option>
            </select>
        </div>
        <div class="row">
            <label for="maxIterations">Max loop iterations</label>
            <input id="maxIterations" type="number" min="1" max="100" value="8">
        </div>
        <div class="row">
            <label for="compactionThreshold">Compaction threshold (msgs)</label>
            <input id="compactionThreshold" type="number" min="4" max="1000" value="24">
        </div>
        <div class="row">
            <label for="compactionPreserveRecent">Preserve recent (msgs)</label>
            <input id="compactionPreserveRecent" type="number" min="0" max="200" value="8">
        </div>
        <div class="section-title" style="margin-top: 8px;">Tools</div>
        <div class="checkbox-row">
            <input id="enReadFile" type="checkbox"><label for="enReadFile">readFile</label>
            <input id="enWriteFile" type="checkbox"><label for="enWriteFile">writeFile</label>
            <input id="enPatchFile" type="checkbox"><label for="enPatchFile">patchFile</label>
        </div>
        <div class="checkbox-row">
            <input id="enTerminal" type="checkbox"><label for="enTerminal">executeTerminalCommand</label>
            <input id="enSearch" type="checkbox"><label for="enSearch">searchCodebase</label>
        </div>
        <div class="row" style="margin-top: 8px;">
            <label for="systemPrompt">System prompt override</label>
            <input id="systemPrompt" type="text" placeholder="(optional) base system prompt">
        </div>
    </div>

    <div class="actions">
        <button id="save">Save</button>
    </div>
    <div class="status" id="status"></div>

<script>
    const vscode = acquireVsCodeApi();
    const $ = (id) => document.getElementById(id);

    function readForm() {
        return {
            modelPath: $('modelPath').value,
            workspaceRoot: $('workspaceRoot').value,
            terminalName: $('terminalName').value,
            terminalMode: $('terminalMode').value,
            maxIterations: parseInt($('maxIterations').value, 10) || 8,
            compactionThreshold: parseInt($('compactionThreshold').value, 10) || 24,
            compactionPreserveRecent: parseInt($('compactionPreserveRecent').value, 10) || 8,
            toolToggles: {
                readFile: $('enReadFile').checked,
                writeFile: $('enWriteFile').checked,
                patchFile: $('enPatchFile').checked,
                terminal: $('enTerminal').checked,
                search: $('enSearch').checked
            },
            inference: {
                maxTokens: parseInt($('maxTokens').value, 10) || 2048,
                temperature: parseFloat($('temperature').value) || 0.7,
                topP: parseFloat($('topP').value) || 0.9,
                topK: parseInt($('topK').value, 10) || 40,
                repeatPenalty: parseFloat($('repeatPenalty').value) || 1.1,
                maxSeqLen: parseInt($('maxSeqLen').value, 10) || 8192,
                nThreads: parseInt($('nThreads').value, 10) || 0
            }
        };
    }

    window.addEventListener('message', (event) => {
        const msg = event.data;
        const status = $('status');
        if (msg.type === 'config') {
            const c = msg.config;
            $('modelPath').value = c.modelPath || '';
            $('workspaceRoot').value = c.workspaceRoot || '';
            $('terminalName').value = c.terminalName || 'TinyCoder Agent';
            $('terminalMode').value = c.terminalMode || 'auto';
            $('maxIterations').value = c.maxIterations ?? 8;
            $('compactionThreshold').value = c.compactionThreshold ?? 24;
            $('compactionPreserveRecent').value = c.compactionPreserveRecent ?? 8;
            $('enReadFile').checked = c.toolToggles?.readFile ?? true;
            $('enWriteFile').checked = c.toolToggles?.writeFile ?? true;
            $('enPatchFile').checked = c.toolToggles?.patchFile ?? true;
            $('enTerminal').checked = c.toolToggles?.terminal ?? true;
            $('enSearch').checked = c.toolToggles?.search ?? true;
            $('systemPrompt').value = c.systemPromptOverride || '';
            $('maxTokens').value = c.inference?.maxTokens ?? 2048;
            $('temperature').value = c.inference?.temperature ?? 0.7;
            $('topP').value = c.inference?.topP ?? 0.9;
            $('topK').value = c.inference?.topK ?? 40;
            $('repeatPenalty').value = c.inference?.repeatPenalty ?? 1.1;
            $('maxSeqLen').value = c.inference?.maxSeqLen ?? 8192;
            $('nThreads').value = c.inference?.nThreads ?? 0;
            status.textContent = '';
        } else if (msg.type === 'modelPicked') {
            $('modelPath').value = msg.path;
        } else if (msg.type === 'workspacePicked') {
            $('workspaceRoot').value = msg.path;
        }
    });

    $('save').addEventListener('click', () => {
        vscode.postMessage({ type: 'save', config: readForm() });
    });
    $('browseModel').addEventListener('click', () => vscode.postMessage({ type: 'browseModel' }));
    $('browseWorkspace').addEventListener('click', () => vscode.postMessage({ type: 'browseWorkspace' }));

    vscode.postMessage({ type: 'ready' });
</script>
</body>
</html>`;
    }
}
