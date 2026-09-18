/*
⚡ TinyCoder AI - Agent Harness Types

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

/**
 * Core shared types for the TinyCoder agent harness.
 *
 * The harness follows the ReAct (Reasoning + Acting) pattern: the model
 * receives a system prompt plus conversation history, returns either a final
 * answer or a structured tool-call JSON block, and the harness executes the
 * call and appends the raw result back to the history. Everything typed here
 * is deliberately decoupled from the inference library so that a custom AP
 * inference implementation can be plugged in via {@link registerAPInference}.
 */

/** Roles understood by the harness and rendered into the model's chat template. */
export type AgentRole = 'system' | 'user' | 'assistant' | 'tool';

/** The set of tools the harness can dispatch. */
export type ToolName =
    | 'readFile'
    | 'writeFile'
    | 'patchFile'
    | 'executeTerminalCommand'
    | 'searchCodebase';

/** JSON-Schema style parameter description (subset sufficient for small local models). */
export interface ToolParameterSchema {
    type: 'object';
    properties: Record<
        string,
        {
            type: string;
            description?: string;
            default?: unknown;
            enum?: string[];
        }
    >;
    required: string[];
}

/** A single tool the agent may call. */
export interface ToolDefinition {
    name: ToolName;
    description: string;
    parameters: ToolParameterSchema;
}

/** Structured request emitted by the model (parsed from JSON). */
export interface ToolCall {
    id: string;
    name: ToolName;
    arguments: Record<string, unknown>;
    /** Raw JSON text the model produced (kept for diagnostics). */
    raw?: string;
}

/** Result of executing one tool call. */
export interface ToolCallResult {
    toolCallId: string;
    name: ToolName;
    ok: boolean;
    output: string;
    durationMs: number;
    error?: string;
}

/**
 * A conversation message maintained by the loop engine.
 *
 * `role: 'assistant'` messages may carry `toolCalls`; the corresponding
 * tool outputs are appended as `role: 'tool'` messages keyed by
 * `toolCallId`/`toolName`.
 */
export interface Message {
    role: AgentRole;
    content: string;
    toolCallId?: string;
    toolName?: ToolName;
    toolCalls?: ToolCall[];
    toolResult?: ToolCallResult;
    /** Set when compaction truncated the raw tool output. */
    truncated?: boolean;
}

/** Namespace of tools the agent is allowed to execute (tool toggles). */
export interface ToolToggles {
    readFile: boolean;
    writeFile: boolean;
    patchFile: boolean;
    terminal: boolean;
    search: boolean;
}

/** Sampling / inference parameters forwarded to the tinycoder-inference bridge. */
export interface InferenceConfig {
    maxTokens: number;
    temperature: number;
    topP: number;
    topK: number;
    repeatPenalty: number;
    repeatLastN: number;
    maxSeqLen: number;
    nThreads: number;
    seed?: number;
}

/** Everything the loop needs to run. */
export interface AgentConfig {
    modelPath: string;
    /** Root the agent may read/write/search; relative paths resolve against it. */
    workspaceRoot: string;
    terminalName: string;
    terminalMode: 'vscode' | 'child' | 'auto';
    maxIterations: number;
    compactionThreshold: number;
    compactionPreserveRecent: number;
    toolToggles: ToolToggles;
    systemPromptOverride?: string;
    inference: InferenceConfig;
}

/** Result of a single AP inference call. */
export interface InferenceResult {
    text: string;
    tokenCount: number;
}

/** Events emitted by the loop so UI layers can render progress. */
export type AgentEvent =
    | { type: 'iteration'; iteration: number; maxIterations: number }
    | { type: 'assistantText'; text: string }
    | { type: 'toolCall'; call: ToolCall }
    | { type: 'toolResult'; result: ToolCallResult }
    | { type: 'compaction'; removed: number; remaining: number }
    | { type: 'final'; text: string }
    | { type: 'error'; message: string };
