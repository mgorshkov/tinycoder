/*
⚡ TinyCoder AI

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
import * as path from 'path';
import * as fs from 'fs';
import {
    loadModel,
    unloadModel,
    generate,
    stopGeneration,
    getStatus,
    isModelLoaded,
    clearKVCache,
    getHardwareInfo,
    InferenceParams,
    GenerateResult,
    getNativeAddonStatus,
    HardwareInfo
} from './nativeBridge';

interface PanelMessage {
    type: string;
    [key: string]: any;
}

/**
 * TinyCoder Chat Panel
 *
 * WebView-based chat interface for interacting with the AI model.
 * Supports:
 * - Chat with context (code selection, file contents)
 * - Code completion and generation
 * - Model loading/unloading with progress bar
 * - Streaming text generation
 * - Hardware info display (CPU/GPU)
 */
export class TinyCoderPanel {
    private panel: vscode.WebviewPanel | undefined;
    private sidebarView: vscode.WebviewView | undefined;
    private context: vscode.ExtensionContext;
    private disposables: vscode.Disposable[] = [];
    private isGenerating = false;
    private lastModelFolder: string | undefined;

    // Storage keys for persistent state
    private static readonly LAST_FOLDER_KEY = 'tinycoder.lastModelFolder';

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
        // Restore last selected folder
        this.lastModelFolder = context.workspaceState.get<string>(TinyCoderPanel.LAST_FOLDER_KEY);
    }

    /**
     * Set the sidebar webview view (used when the panel is shown in the sidebar).
     * Messages will be sent to both the sidebar view and the standalone panel.
     */
    setSidebarView(view: vscode.WebviewView): void {
        this.sidebarView = view;
    }

    /**
     * Show the panel (create if not exists).
     */
    show(): void {
        if (this.panel) {
            this.panel.reveal(vscode.ViewColumn.Beside);
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'tinycoder.chat',
            'TinyCoder AI',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.file(path.join(this.context.extensionPath, 'media'))
                ]
            }
        );

        this.panel.webview.html = this.getHtmlContent();
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        this.panel.webview.onDidReceiveMessage(
            (message: PanelMessage) => this.handleMessage(message),
            null,
            this.disposables
        );
    }

    /**
     * Send a message to the webview (both sidebar and standalone panel).
     */
    sendMessage(message: PanelMessage): void {
        // Send to standalone panel if open
        if (this.panel) {
            try {
                this.panel.webview.postMessage(message);
            } catch (err) {
                console.error('[TinyCoder] Failed to send message to panel:', err);
            }
        }
        // Also send to sidebar view if available
        if (this.sidebarView) {
            try {
                this.sidebarView.webview.postMessage(message);
            } catch (err) {
                console.error('[TinyCoder] Failed to send message to sidebar:', err);
            }
        }
    }

    /**
     * Dispose the panel.
     */
    dispose(): void {
        if (this.isGenerating) {
            stopGeneration();
        }

        this.panel = undefined;

        for (const disposable of this.disposables) {
            disposable.dispose();
        }
        this.disposables = [];
    }

    /**
     * Handle messages from the webview.
     */
    public async handleMessage(message: PanelMessage): Promise<void> {
        switch (message.type) {
            case 'ready':
                this.sendStatus();
                this.sendHardwareInfo();
                break;

            case 'openModelDialog':
                await this.handleOpenModelDialog();
                break;

            case 'loadModel':
                await this.handleLoadModel(message.path);
                break;

            case 'unloadModel':
                unloadModel();
                this.sendStatus();
                break;

            case 'chat':
                await this.handleChat(message.content, message.params);
                break;

            case 'stopGeneration':
                stopGeneration();
                this.isGenerating = false;
                break;

            case 'clearCache':
                clearKVCache();
                break;

            case 'getStatus':
                this.sendStatus();
                break;

            case 'getHardwareInfo':
                this.sendHardwareInfo();
                break;

            case 'insertCode':
                this.insertCodeToEditor(message.code);
                break;
        }
    }

    /**
     * Send hardware info to the webview.
     */
    private sendHardwareInfo(): void {
        try {
            const hwInfo = getHardwareInfo();
            this.sendMessage({
                type: 'hardwareInfo',
                info: hwInfo
            });
        } catch (err: any) {
            console.error('[TinyCoder] Failed to get hardware info:', err);
        }
    }

    /**
     * Handle the "Open Model" dialog request from the webview.
     * Opens VS Code's native file picker filtered for .gguf files,
     * defaulting to the last selected folder.
     */
    private async handleOpenModelDialog(): Promise<void> {
        console.log('[TinyCoder] handleOpenModelDialog called');
        const options: vscode.OpenDialogOptions = {
            canSelectMany: false,
            openLabel: 'Load Model',
            defaultUri: this.lastModelFolder ? vscode.Uri.file(this.lastModelFolder) : undefined,
            filters: {
                'GGUF Models': ['gguf'],
                'All Files': ['*']
            }
        };

        const fileUri = await vscode.window.showOpenDialog(options);
        console.log('[TinyCoder] showOpenDialog returned:', fileUri ? fileUri[0].fsPath : 'undefined (cancelled)');
        if (fileUri && fileUri[0]) {
            const modelPath = fileUri[0].fsPath;
            // Save the folder for next time
            this.lastModelFolder = path.dirname(modelPath);
            this.context.workspaceState.update(TinyCoderPanel.LAST_FOLDER_KEY, this.lastModelFolder);
            // Load the model directly
            await this.handleLoadModel(modelPath);
        }
    }

    /**
     * Handle model loading with progress reporting.
     */
    private async handleLoadModel(modelPath: string): Promise<void> {
        // Validate path
        if (!modelPath) {
            const msg = 'No model path specified. Please enter a path to a .gguf model file.';
            this.sendMessage({ type: 'error', message: msg });
            vscode.window.showErrorMessage(`TinyCoder: ${msg}`);
            return;
        }

        // Check file existence
        if (!fs.existsSync(modelPath)) {
            const msg = `Model file not found: ${modelPath}`;
            this.sendMessage({ type: 'error', message: msg });
            vscode.window.showErrorMessage(`TinyCoder: ${msg}`);
            return;
        }

        // Check file extension
        const ext = path.extname(modelPath).toLowerCase();
        if (ext !== '.gguf') {
            const msg = `Unsupported file format "${ext}". TinyCoder requires .gguf model files.`;
            this.sendMessage({ type: 'error', message: msg });
            vscode.window.showWarningMessage(`TinyCoder: ${msg}`);
        }

        // Check native addon status
        const addonStatus = getNativeAddonStatus();
        if (!addonStatus.available) {
            const msg = `Native addon not loaded: ${addonStatus.error || 'unknown error'}. Try rebuilding the extension.`;
            this.sendMessage({ type: 'error', message: msg });
            vscode.window.showErrorMessage(`TinyCoder: ${msg}`);
            return;
        }

        // Save the folder for next time
        this.lastModelFolder = path.dirname(modelPath);
        this.context.workspaceState.update(TinyCoderPanel.LAST_FOLDER_KEY, this.lastModelFolder);

        // Show progress bar
        this.sendMessage({
            type: 'loadProgress',
            progress: 0,
            stage: 'Starting...'
        });

        // nThreads 0 = auto: the bridge maps it to ThreadPool::recommendedThreadCount()
        // (logical CPU count, measured optimum). A physical-core default regresses
        // memory-bound generation (see plans/generation_optimizations.md §6.10).
        const result = await loadModel(modelPath, {
            nThreads: vscode.workspace.getConfiguration('tinycoder').get('nThreads', 0),
            maxSeqLen: vscode.workspace.getConfiguration('tinycoder').get('maxSeqLen', 2048)
        }, (progress: number, stage: string) => {
            this.sendMessage({
                type: 'loadProgress',
                progress,
                stage
            });
        });

        if (result.success) {
            this.sendMessage({
                type: 'modelLoaded',
                info: result.modelInfo
            });
            vscode.window.showInformationMessage(
                `TinyCoder: Model loaded (${result.modelInfo?.numLayers} layers, ` +
                `${result.modelInfo?.hiddenSize} hidden, ${result.modelInfo?.numAttentionHeads} heads)`
            );
        } else {
            const errMsg = `Failed to load model: ${result.error}`;
            this.sendMessage({ type: 'error', message: errMsg });
            vscode.window.showErrorMessage(`TinyCoder: ${errMsg}`);
        }
    }

    /**
     * Handle a chat message.
     */
    private async handleChat(content: string, params?: InferenceParams): Promise<void> {
        if (!isModelLoaded()) {
            this.sendMessage({
                type: 'error',
                message: 'No model loaded. Please load a model first.'
            });
            return;
        }

        if (this.isGenerating) {
            return;
        }

        this.isGenerating = true;

        // Build prompt with chat template
        const prompt = this.buildChatPrompt(content);

        // Get generation parameters from settings
        const genParams: InferenceParams = {
            maxTokens: params?.maxTokens || vscode.workspace.getConfiguration('tinycoder').get('maxTokens', 2048),
            temperature: params?.temperature || vscode.workspace.getConfiguration('tinycoder').get('temperature', 0.7),
            topP: params?.topP || vscode.workspace.getConfiguration('tinycoder').get('topP', 0.9),
            topK: params?.topK || vscode.workspace.getConfiguration('tinycoder').get('topK', 40),
            repeatPenalty: params?.repeatPenalty || vscode.workspace.getConfiguration('tinycoder').get('repeatPenalty', 1.1),
            repeatLastN: params?.repeatLastN || vscode.workspace.getConfiguration('tinycoder').get('repeatLastN', 64),
            seed: params?.seed || 0
        };

        let fullText = '';

        try {
            const result = await generate(prompt, genParams, (token, text) => {
                fullText += text;
                this.sendMessage({
                    type: 'token',
                    token,
                    text,
                    fullText
                });
                return true; // Continue generation
            });

            this.sendMessage({
                type: 'generationComplete',
                tokens: result.tokens,
                text: result.text,
                tokenCount: result.tokenCount
            });
        } catch (err: any) {
            this.sendMessage({
                type: 'error',
                message: `Generation error: ${err.message}`
            });
        } finally {
            this.isGenerating = false;
        }
    }

    /**
     * Build a chat prompt using the Qwen2.5-Coder chat template.
     */
    private buildChatPrompt(userMessage: string): string {
        return `<|im_start|>system
You are TinyCoder, an AI coding assistant powered by Qwen2.5-Coder. You help users write, debug, and understand code. Be concise and provide working code examples.<|im_end|>
<|im_start|>user
${userMessage}<|im_end|>
<|im_start|>assistant
`;
    }

    /**
     * Insert code into the active editor.
     */
    private insertCodeToEditor(code: string): void {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        editor.edit(editBuilder => {
            const selection = editor.selection;
            if (selection.isEmpty) {
                editBuilder.insert(selection.active, code);
            } else {
                editBuilder.replace(selection, code);
            }
        });
    }

    /**
     * Send current status to webview.
     */
    private sendStatus(): void {
        const status = getStatus();
        this.sendMessage({
            type: 'statusUpdate',
            status
        });
    }

    /**
     * Get the HTML content for the webview.
     */
    public getHtmlContent(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg-primary: #1e1e2e;
            --bg-secondary: #181825;
            --bg-tertiary: #313244;
            --text-primary: #cdd6f4;
            --text-secondary: #a6adc8;
            --text-muted: #6c7086;
            --accent: #89b4fa;
            --accent-hover: #74c7ec;
            --success: #a6e3a1;
            --warning: #f9e2af;
            --error: #f38ba8;
            --border: #45475a;
            --code-bg: #11111b;
            --progress-bg: #313244;
            --progress-fill: #89b4fa;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Header */
        .header {
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
        }

        .header-title {
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .header-title .icon {
            font-size: 18px;
        }

        .header-actions {
            display: flex;
            gap: 8px;
        }

        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        .status-indicator.loaded {
            background: var(--success);
        }

        .status-indicator.not-loaded {
            background: var(--warning);
        }

        .status-indicator.generating {
            background: var(--accent);
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Progress Bar */
        .progress-container {
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: none;
            flex-shrink: 0;
        }

        .progress-container.visible {
            display: block;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--progress-bg);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--accent), var(--success));
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-top: 6px;
            font-size: 11px;
            color: var(--text-muted);
        }

        .progress-stage {
            flex: 1;
        }

        .progress-percent {
            font-weight: 600;
        }

        /* Hardware Info */
        .hw-info {
            padding: 8px 16px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: none;
            flex-shrink: 0;
        }

        .hw-info.visible {
            display: block;
        }

        .hw-section {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: var(--text-muted);
            padding: 2px 0;
        }

        .hw-section .label {
            font-weight: 600;
            min-width: 32px;
        }

        .hw-section .value {
            color: var(--text-secondary);
        }

        .hw-section .badge {
            display: inline-block;
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 600;
        }

        .badge.cpu {
            background: #313244;
            color: #89b4fa;
        }

        .badge.gpu {
            background: #313244;
            color: #a6e3a1;
        }

        .badge.gpu.na {
            background: #313244;
            color: #6c7086;
        }

        /* Chat Messages */
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .message {
            max-width: 85%;
            padding: 10px 14px;
            border-radius: 12px;
            font-size: 13px;
            line-height: 1.5;
            animation: fadeIn 0.2s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            background: var(--accent);
            color: var(--bg-primary);
        }

        .message.assistant {
            align-self: flex-start;
            background: var(--bg-tertiary);
        }

        .message.system {
            align-self: center;
            background: var(--bg-secondary);
            color: var(--text-muted);
            font-size: 12px;
            max-width: 100%;
            text-align: center;
        }

        .message.error {
            align-self: center;
            background: var(--error);
            color: var(--bg-primary);
            max-width: 100%;
            text-align: center;
        }

        .message pre {
            background: var(--code-bg);
            padding: 10px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 8px 0;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 12px;
        }

        .message code {
            background: var(--code-bg);
            padding: 1px 4px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 12px;
        }

        .message .timestamp {
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* Input Area */
        .input-area {
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            flex-shrink: 0;
        }

        .input-container {
            display: flex;
            gap: 8px;
            align-items: flex-end;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        textarea {
            width: 100%;
            padding: 10px 14px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-primary);
            font-size: 13px;
            font-family: inherit;
            resize: none;
            outline: none;
            min-height: 40px;
            max-height: 120px;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: var(--accent);
        }

        textarea::placeholder {
            color: var(--text-muted);
        }

        .send-btn {
            padding: 8px 16px;
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: background 0.2s;
            white-space: nowrap;
        }

        .send-btn:hover {
            background: var(--accent-hover);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .stop-btn {
            padding: 8px 16px;
            background: var(--error);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <span class="icon">⚡</span>
            TinyCoder AI
            <span class="status-indicator not-loaded" id="statusIndicator"></span>
        </div>
        <div class="header-actions">
            <button onclick="loadModelDialog()" style="padding:4px 10px;background:var(--bg-tertiary);border:1px solid var(--border);border-radius:6px;color:var(--text-primary);cursor:pointer;font-size:12px;">
                📂 Load Model
            </button>
            <button onclick="unloadModel()" style="padding:4px 10px;background:var(--bg-tertiary);border:1px solid var(--border);border-radius:6px;color:var(--text-primary);cursor:pointer;font-size:12px;">
                ⏻ Unload
            </button>
            <button onclick="clearChat()" style="padding:4px 10px;background:var(--bg-tertiary);border:1px solid var(--border);border-radius:6px;color:var(--text-primary);cursor:pointer;font-size:12px;">
                🗑️ Clear
            </button>
        </div>
    </div>

    <!-- Progress Bar -->
    <div class="progress-container" id="progressContainer">
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill"></div>
        </div>
        <div class="progress-label">
            <span class="progress-stage" id="progressStage">Starting...</span>
            <span class="progress-percent" id="progressPercent">0%</span>
        </div>
    </div>

    <!-- Hardware Info -->
    <div class="hw-info" id="hwInfo">
        <div class="hw-section">
            <span class="badge cpu">CPU</span>
            <span class="value" id="cpuInfo">Detecting...</span>
        </div>
        <div class="hw-section">
            <span class="badge gpu" id="gpuBadge">GPU</span>
            <span class="value" id="gpuInfo">Detecting...</span>
        </div>
    </div>

    <div class="messages" id="messages">
        <div class="message system" id="welcomeMessage">
            ⚡ TinyCoder AI — Click "Load Model" to get started
        </div>
    </div>

    <div class="input-area">
        <div class="input-container">
            <div class="input-wrapper">
                <textarea id="input" placeholder="Ask TinyCoder to write, explain, or debug code..."
                          rows="1" oninput="autoResize(this)"
                          onkeydown="handleKeyDown(event)"></textarea>
            </div>
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let isGenerating = false;
        let currentAssistantMsg = null;

        // Notify extension that panel is ready
        vscode.postMessage({ type: 'ready' });

        // Listen for messages from extension
        window.addEventListener('message', event => {
            try {
                const msg = event.data;

                switch (msg.type) {
                case 'token':
                    handleToken(msg);
                    break;

                case 'generationComplete':
                    isGenerating = false;
                    updateSendButton();
                    break;

                case 'loadProgress':
                    updateProgress(msg.progress, msg.stage);
                    break;

                case 'modelLoaded':
                    updateStatus('loaded');
                    hideProgress();
                    addSystemMessage('✅ Model loaded: ' + msg.info.numLayers + ' layers, ' +
                        msg.info.hiddenSize + ' hidden, ' + msg.info.numAttentionHeads + ' heads');
                    break;

                case 'hardwareInfo':
                    updateHardwareInfo(msg.info);
                    break;

                case 'error':
                    isGenerating = false;
                    updateSendButton();
                    hideProgress();
                    addErrorMessage(msg.message);
                    break;

                case 'status':
                    addSystemMessage(msg.message);
                    break;

                case 'statusUpdate':
                    if (msg.status.loaded) {
                        updateStatus(msg.status.generating ? 'generating' : 'loaded');
                    } else {
                        updateStatus('not-loaded');
                    }
                    break;

                case 'setModelPath':
                    const modelPathEl = document.getElementById('modelPath');
                    if (modelPathEl) modelPathEl.value = msg.path;
                    const modelLoaderEl = document.getElementById('modelLoader');
                    if (modelLoaderEl) modelLoaderEl.classList.add('visible');
                    break;
            }
            } catch (err) {
                console.error('[TinyCoder] Webview message handler error:', err);
            }
        });

        function handleToken(msg) {
            if (!currentAssistantMsg) {
                currentAssistantMsg = addAssistantMessage('');
            }
            if (currentAssistantMsg) {
                currentAssistantMsg.innerHTML += escapeHtml(msg.text);
                scrollToBottom();
            }
        }

        function updateProgress(progress, stage) {
            const container = document.getElementById('progressContainer');
            const fill = document.getElementById('progressFill');
            const stageEl = document.getElementById('progressStage');
            const percentEl = document.getElementById('progressPercent');

            if (!container || !fill || !stageEl || !percentEl) return;

            // Remove welcome message once model starts loading
            const welcomeMsg = document.getElementById('welcomeMessage');
            if (welcomeMsg) {
                welcomeMsg.remove();
            }

            container.classList.add('visible');
            const pct = Math.round(progress * 100);
            fill.style.width = pct + '%';
            stageEl.textContent = stage;
            percentEl.textContent = pct + '%';
        }

        function hideProgress() {
            const container = document.getElementById('progressContainer');
            if (container) {
                container.classList.remove('visible');
            }
        }

        function updateHardwareInfo(info) {
            const hwInfo = document.getElementById('hwInfo');
            if (!hwInfo) return;
            hwInfo.classList.add('visible');

            // CPU
            const cpuInfo = document.getElementById('cpuInfo');
            if (cpuInfo) {
                cpuInfo.textContent =
                    info.cpu.model + ' (' + info.cpu.cores + ' cores, ' + info.cpu.ompThreads + ' OMP threads)';
            }

            // GPU
            const gpuBadge = document.getElementById('gpuBadge');
            const gpuInfo = document.getElementById('gpuInfo');
            if (gpuBadge && gpuInfo) {
                if (info.gpu.available) {
                    gpuBadge.className = 'badge gpu';
                    gpuInfo.textContent = info.gpu.name + ' (' + info.gpu.cores + ' cores' +
                        (info.gpu.memoryMB ? ', ' + info.gpu.memoryMB + ' MB' : '') + ')';
                } else {
                    gpuBadge.className = 'badge gpu na';
                    gpuInfo.textContent = info.gpu.name;
                }
            }
        }

        function sendMessage() {
            const input = document.getElementById('input');
            if (!input) return;
            const content = input.value.trim();

            if (!content || isGenerating) return;

            addUserMessage(content);
            input.value = '';
            input.style.height = 'auto';

            currentAssistantMsg = null;
            isGenerating = true;
            updateSendButton();

            vscode.postMessage({
                type: 'chat',
                content: content
            });
        }

        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function autoResize(textarea) {
            if (!textarea) return;
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
        }

        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
        }

        function loadModelDialog() {
            try {
                vscode.postMessage({ type: 'openModelDialog' });
            } catch (err) {
                console.error('[TinyCoder] Failed to send openModelDialog:', err);
            }
        }

        function unloadModel() {
            try {
                vscode.postMessage({ type: 'unloadModel' });
            } catch (err) {
                console.error('[TinyCoder] Failed to send unloadModel:', err);
            }
        }

        function clearChat() {
            const messagesEl = document.getElementById('messages');
            if (messagesEl) {
                messagesEl.innerHTML =
                    '<div class="message system">⚡ TinyCoder AI — Chat cleared</div>';
            }
            currentAssistantMsg = null;
        }

        function addUserMessage(text) {
            const messages = document.getElementById('messages');
            if (!messages) return;
            const div = document.createElement('div');
            div.className = 'message user';
            div.textContent = text;
            messages.appendChild(div);
            scrollToBottom();
        }

        function addAssistantMessage(text) {
            const messages = document.getElementById('messages');
            if (!messages) return null;
            const div = document.createElement('div');
            div.className = 'message assistant';
            div.innerHTML = text;
            messages.appendChild(div);
            scrollToBottom();
            return div;
        }

        function addSystemMessage(text) {
            const messages = document.getElementById('messages');
            if (!messages) return;
            const div = document.createElement('div');
            div.className = 'message system';
            div.textContent = text;
            messages.appendChild(div);
            scrollToBottom();
        }

        function addErrorMessage(text) {
            const messages = document.getElementById('messages');
            if (!messages) return;
            const div = document.createElement('div');
            div.className = 'message error';
            div.textContent = text;
            messages.appendChild(div);
            scrollToBottom();
        }

        function updateStatus(status) {
            const indicator = document.getElementById('statusIndicator');
            if (indicator) {
                indicator.className = 'status-indicator ' + status;
            }
        }

        function updateSendButton() {
            const btn = document.getElementById('sendBtn');
            if (!btn) return;
            if (isGenerating) {
                btn.textContent = 'Stop';
                btn.className = 'stop-btn';
                btn.onclick = () => {
                    vscode.postMessage({ type: 'stopGeneration' });
                    isGenerating = false;
                    updateSendButton();
                };
            } else {
                btn.textContent = 'Send';
                btn.className = 'send-btn';
                btn.onclick = sendMessage;
            }
        }

        function scrollToBottom() {
            const messages = document.getElementById('messages');
            if (messages) {
                messages.scrollTop = messages.scrollHeight;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>`;
    }
}
