/*
⚡ TinyCoder AI - Agent Harness Config

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
import { AgentConfig } from './types';

/**
 * Load and normalize the agent configuration from VS Code settings.
 * Falls back to the first workspace folder for the workspace root when the
 * setting is empty.
 */
export function loadAgentConfig(): AgentConfig {
    const cfg = vscode.workspace.getConfiguration('tinycoder');

    // Model path
    const modelPath = cfg.get<string>('modelPath', '');
    const workspaceFolders = vscode.workspace.workspaceFolders;
    const defaultRoot = workspaceFolders && workspaceFolders.length > 0
        ? workspaceFolders[0].uri.fsPath
        : '';
    const workspaceRoot = cfg.get<string>('agent.workspaceRoot', '') || defaultRoot;

    // Tool toggles
    const toolToggles = {
        readFile: cfg.get<boolean>('agent.enableReadFile', true),
        writeFile: cfg.get<boolean>('agent.enableWriteFile', true),
        patchFile: cfg.get<boolean>('agent.enablePatchFile', true),
        terminal: cfg.get<boolean>('agent.enableTerminal', true),
        search: cfg.get<boolean>('agent.enableSearch', true)
    };

    const terminalMode = cfg.get<'vscode' | 'child' | 'auto'>('agent.terminalMode', 'auto');
    const inference = {
        maxTokens: cfg.get<number>('maxTokens', 2048),
        temperature: cfg.get<number>('temperature', 0.7),
        topP: cfg.get<number>('topP', 0.9),
        topK: cfg.get<number>('topK', 40),
        repeatPenalty: cfg.get<number>('repeatPenalty', 1.1),
        repeatLastN: cfg.get<number>('repeatLastN', 64),
        maxSeqLen: cfg.get<number>('maxSeqLen', 8192),
        nThreads: cfg.get<number>('nThreads', 0)
    };

    return {
        modelPath,
        workspaceRoot: path.normalize(workspaceRoot || ''),
        terminalName: cfg.get<string>('agent.terminalName', 'TinyCoder Agent'),
        terminalMode,
        maxIterations: cfg.get<number>('agent.maxIterations', 8),
        compactionThreshold: cfg.get<number>('agent.compactionThreshold', 24),
        compactionPreserveRecent: cfg.get<number>('agent.compactionPreserveRecent', 8),
        toolToggles,
        systemPromptOverride: cfg.get<string>('agent.systemPrompt', ''),
        inference
    };
}

/** Persist a partial config back to VS Code settings. */
export async function saveAgentConfig(config: AgentConfig): Promise<void> {
    const cfg = vscode.workspace.getConfiguration('tinycoder');
    const updates: Array<[string, unknown]> = [
        ['modelPath', config.modelPath],
        ['agent.workspaceRoot', config.workspaceRoot],
        ['agent.terminalName', config.terminalName],
        ['agent.terminalMode', config.terminalMode],
        ['agent.maxIterations', config.maxIterations],
        ['agent.compactionThreshold', config.compactionThreshold],
        ['agent.compactionPreserveRecent', config.compactionPreserveRecent],
        ['agent.systemPrompt', config.systemPromptOverride || ''],
        ['agent.enableReadFile', config.toolToggles.readFile],
        ['agent.enableWriteFile', config.toolToggles.writeFile],
        ['agent.enablePatchFile', config.toolToggles.patchFile],
        ['agent.enableTerminal', config.toolToggles.terminal],
        ['agent.enableSearch', config.toolToggles.search],
        ['maxTokens', config.inference.maxTokens],
        ['temperature', config.inference.temperature],
        ['topP', config.inference.topP],
        ['topK', config.inference.topK],
        ['repeatPenalty', config.inference.repeatPenalty],
        ['repeatLastN', config.inference.repeatLastN],
        ['maxSeqLen', config.inference.maxSeqLen],
        ['nThreads', config.inference.nThreads]
    ];
    for (const [key, value] of updates) {
        await cfg.update(key, value, vscode.ConfigurationTarget.Global);
    }
}
