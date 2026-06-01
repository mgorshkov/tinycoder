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
import { TinyCoderPanel } from './panel';
import { getStatus, isModelLoaded } from './nativeBridge';

let panel: TinyCoderPanel | undefined;

/**
 * Provider for the TinyCoder sidebar webview view.
 */
class TinyCoderViewProvider implements vscode.WebviewViewProvider {
    constructor(private readonly context: vscode.ExtensionContext) {}

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ): void {
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.file(path.join(this.context.extensionPath, 'media'))
            ]
        };

        // Create a panel instance to reuse its HTML and message handling
        if (!panel) {
            panel = new TinyCoderPanel(this.context);
        }
        // Register the sidebar view so messages can be sent back to it
        panel.setSidebarView(webviewView);
        webviewView.webview.html = panel.getHtmlContent();

        webviewView.webview.onDidReceiveMessage(
            async (message: any) => {
                console.log('[TinyCoder] Webview message:', JSON.stringify(message));
                if (panel) {
                    try {
                        await panel.handleMessage(message);
                    } catch (err: any) {
                        console.error('[TinyCoder] Error handling message:', err);
                        vscode.window.showErrorMessage(`TinyCoder internal error: ${err.message}`);
                    }
                } else {
                    console.warn('[TinyCoder] No panel instance to handle message');
                }
            }
        );
    }
}

/**
 * Activate the TinyCoder extension.
 */
export function activate(context: vscode.ExtensionContext) {
    console.log('[TinyCoder] ===== Activating TinyCoder extension =====');
    console.log('[TinyCoder] Extension path:', context.extensionPath);

    // Register the sidebar webview provider
    const provider = new TinyCoderViewProvider(context);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('tinycoder.chat', provider)
    );

    // Register the main chat panel command
    const openPanelCommand = vscode.commands.registerCommand('tinycoder.openPanel', () => {
        if (!panel) {
            panel = new TinyCoderPanel(context);
        }
        panel.show();
    });

    // Register command to explain selected code
    const explainCommand = vscode.commands.registerCommand('tinycoder.explainCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showInformationMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);

        if (!selectedText) {
            vscode.window.showInformationMessage('No code selected');
            return;
        }

        if (!panel) {
            panel = new TinyCoderPanel(context);
        }
        panel.show();
        panel.sendMessage({
            type: 'chat',
            content: `Explain this code:\n\`\`\`\n${selectedText}\n\`\`\``
        });
    });

    // Register command to complete code
    const completeCommand = vscode.commands.registerCommand('tinycoder.completeCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const document = editor.document;
        const position = editor.selection.active;
        const line = document.lineAt(position.line);

        // Get context before cursor
        const contextRange = new vscode.Range(
            Math.max(0, position.line - 20), 0,
            position.line, line.text.length
        );
        const codeContext = document.getText(contextRange);

        if (!panel) {
            panel = new TinyCoderPanel(context);
        }
        panel.show();
        panel.sendMessage({
            type: 'chat',
            content: `Complete the following code:\n\`\`\`${document.languageId}\n${codeContext}\n\`\`\``
        });
    });

    // Register command to generate code from comment
    const generateCommand = vscode.commands.registerCommand('tinycoder.generateCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);

        if (!panel) {
            panel = new TinyCoderPanel(context);
        }
        panel.show();
        panel.sendMessage({
            type: 'chat',
            content: selectedText
                ? `Generate code for: ${selectedText}`
                : 'What would you like me to generate?'
        });
    });

    // Register command to load a model
    const loadModelCommand = vscode.commands.registerCommand('tinycoder.loadModel', async () => {
        const options: vscode.OpenDialogOptions = {
            canSelectMany: false,
            openLabel: 'Load Model',
            filters: {
                'GGUF Models': ['gguf'],
                'All Files': ['*']
            }
        };

        const fileUri = await vscode.window.showOpenDialog(options);
        if (fileUri && fileUri[0]) {
            const modelPath = fileUri[0].fsPath;
            if (panel) {
                panel.sendMessage({
                    type: 'loadModel',
                    path: modelPath
                });
            } else {
                panel = new TinyCoderPanel(context);
                panel.show();
                // Wait for panel to be ready, then send
                setTimeout(() => {
                    panel?.sendMessage({
                        type: 'loadModel',
                        path: modelPath
                    });
                }, 1000);
            }
        }
    });

    // Register command to show model status
    const statusCommand = vscode.commands.registerCommand('tinycoder.showStatus', () => {
        const status = getStatus();
        if (status.loaded) {
            vscode.window.showInformationMessage(
                `TinyCoder: Model loaded | KV Cache: ${status.kvCacheSize} tokens | ` +
                `Threads: ${status.config?.nThreads}`
            );
        } else {
            vscode.window.showWarningMessage('TinyCoder: No model loaded. Use "Load Model" to get started.');
        }
    });

    context.subscriptions.push(
        openPanelCommand,
        explainCommand,
        completeCommand,
        generateCommand,
        loadModelCommand,
        statusCommand
    );

    // Add status bar item
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = '$(chip) TinyCoder';
    statusBarItem.command = 'tinycoder.showStatus';
    statusBarItem.tooltip = 'Click to check TinyCoder status';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Update status bar periodically
    setInterval(() => {
        const status = getStatus();
        if (status.loaded) {
            statusBarItem.text = status.generating
                ? '$(sync~spin) TinyCoder'
                : '$(chip) TinyCoder';
            statusBarItem.backgroundColor = undefined;
        } else {
            statusBarItem.text = '$(chip) TinyCoder (no model)';
            statusBarItem.backgroundColor = new vscode.ThemeColor(
                'statusBarItem.warningBackground'
            );
        }
    }, 2000);

    console.log('[TinyCoder] Extension activated');
}

/**
 * Deactivate the TinyCoder extension.
 */
export function deactivate() {
    if (panel) {
        panel.dispose();
        panel = undefined;
    }
}
