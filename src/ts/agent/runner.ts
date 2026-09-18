/*
⚡ TinyCoder AI - Agent Harness Runner

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
import { isModelLoaded, loadModel } from '../nativeBridge';
import { AgentConfig, AgentEvent } from './types';
import { loadAgentConfig } from './config';
import { runAgentLoop } from './agentLoop';

/**
 * Ensure the configured GGUF model is loaded in the native engine,
 * loading it from `tinycoder.modelPath` if necessary.
 */
export async function ensureModelLoaded(): Promise<boolean> {
    if (isModelLoaded()) {
        return true;
    }
    const config = loadAgentConfig();
    if (!config.modelPath) {
        vscode.window.showErrorMessage(
            'TinyCoder: no model path configured. Set "tinycoder.modelPath" or use Agent Options.'
        );
        return false;
    }
    const result = await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'TinyCoder: loading model…',
            cancellable: true
        },
        (progress) =>
            loadModel(config.modelPath, {
                nThreads: config.inference.nThreads,
                maxSeqLen: config.inference.maxSeqLen
            }, (p, stage) => {
                progress.report({ increment: p * 100, message: stage });
            })
    );
    if (result.success) {
        vscode.window.showInformationMessage(
            `TinyCoder: model loaded (${result.modelInfo?.numLayers} layers)`
        );
        return true;
    }
    vscode.window.showErrorMessage(`TinyCoder: failed to load model: ${result.error}`);
    return false;
}

/**
 * Run the ReAct agent for a user prompt.
 *
 * Loads config (workspace root required), ensures the model is loaded, then
 * invokes the loop, forwarding events to `onEvent`. Returns the final answer.
 */
export async function runAgentTask(
    prompt: string,
    onEvent?: (event: AgentEvent) => void,
    configOverride?: Partial<AgentConfig>
): Promise<string> {
    const config: AgentConfig = { ...loadAgentConfig(), ...configOverride };
    if (!config.workspaceRoot) {
        vscode.window.showErrorMessage(
            'TinyCoder agent: no workspace root. Open a folder or set tinycoder.agent.workspaceRoot.'
        );
        return '';
    }
    const loaded = await ensureModelLoaded();
    if (!loaded) {
        return '';
    }
    return runAgentLoop(prompt, config, onEvent);
}
