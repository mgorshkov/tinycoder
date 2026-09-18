/*
⚡ TinyCoder AI - Agent Harness ReAct Loop

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

import { AgentConfig, AgentEvent, Message, ToolCall, ToolCallResult, ToolName } from './types';
import { buildSystemPrompt } from './prompt';
import { renderPrompt } from './chatTemplate';
import { callAPInference } from './inference';
import { executeTool, makeToolCallId } from './tools';
import { compactConversation, clampToolResultSize } from './compaction';

/** Cancel token respected by the loop between iterations. */
export interface LoopAbortSignal {
    aborted: boolean;
}

const TOOL_CALL_RE = /\{\s*"tool_call"\s*:\s*(\{[\s\S]*?\})\s*\}/;

// Triple backtick fence (built without literal backticks to stay template-safe).
/** Strip markdown fences, thinking/reasoning blocks and stray numbering. */
function normalizeRawOutput(input: string): string {
    const BT = String.fromCharCode(96);
    const fence = BT + BT + BT;
    return input
        .replace(/<thinking>[\s\S]*?<\/thinking>/gi, '')
        .replace(/<reasoning>[\s\S]*?<\/reasoning>/gi, '')
        .replace(new RegExp(fence + '\\s*json', 'gi'), '')
        .split(fence).join('')
        .replace(/^\s*\d+\.\s*/gm, '');
}

/**
 * Extract a tool-call from the model's raw output.
 *
 * The model is instructed to emit exactly one JSON object:
 *   {"tool_call":{"name":"readFile","arguments":{...}}}
 * We search for that shape; if absent we return null (final answer).
 */
export function parseToolCall(rawInput: string): ToolCall | null {
    const raw = normalizeRawOutput(rawInput);
    // Try the explicit wrapper first.
    const wrapper = raw.match(TOOL_CALL_RE);
    if (wrapper && wrapper[1]) {
        try {
            const inner = JSON.parse(wrapper[1]);
            if (inner && typeof inner.name === 'string' && inner.arguments && typeof inner.arguments === 'object') {
                return {
                    id: makeToolCallId(),
                    name: inner.name as ToolName,
                    arguments: inner.arguments as Record<string, unknown>,
                    raw: wrapper[0]
                };
            }
        } catch {
            // fall through
        }
    }

    // Fallback: bare JSON object with name+arguments anywhere in the output.
    const bare = raw.match(/\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}/);
    if (bare) {
        try {
            const args = JSON.parse(bare[2]);
            return {
                id: makeToolCallId(),
                name: bare[1] as ToolName,
                arguments: args as Record<string, unknown>,
                raw: bare[0]
            };
        } catch {
            return null;
        }
    }
    return null;
}

/**
 * Run the ReAct loop for a single user prompt.
 *
 * @param userPrompt  The user's request.
 * @param config      Fully-resolved agent configuration.
 * @param onEvent     Optional observer for UI progress (iteration, text, etc.).
 * @param signal      Optional abort/cancel token.
 * @returns The final assistant answer text (already delivered via events too).
 */
export async function runAgentLoop(
    userPrompt: string,
    config: AgentConfig,
    onEvent?: (event: AgentEvent) => void,
    signal?: LoopAbortSignal
): Promise<string> {
    const emit = (e: AgentEvent) => {
        try {
            onEvent?.(e);
        } catch {
            // observers must not break the loop
        }
    };

    const systemPrompt = buildSystemPrompt(config);
    let history: Message[] = [
        { role: 'user', content: userPrompt }
    ];

    let finalAnswer = '';

    for (let iteration = 0; iteration < config.maxIterations; iteration++) {
        if (signal?.aborted) {
            emit({ type: 'error', message: 'Loop aborted by user' });
            return 'Loop aborted.';
        }

        emit({ type: 'iteration', iteration: iteration + 1, maxIterations: config.maxIterations });

        // 1. Send the payload to inference.
        const prompt = renderPrompt(systemPrompt, history);
        let output: string;
        try {
            const result = await callAPInference(prompt, config);
            output = result.text;
        } catch (err: any) {
            const msg = `Inference error: ${err?.message || err}`;
            emit({ type: 'error', message: msg });
            return `Error: ${msg}`;
        }

        // 2. Append the assistant message (with the raw output).
        const assistantMsg: Message = { role: 'assistant', content: output };
        history.push(assistantMsg);

        // 3. Parse a tool call.
        const call = parseToolCall(output);
        if (!call) {
            // Final answer reached.
            finalAnswer = output.trim();
            assistantMsg.content = finalAnswer;
            emit({ type: 'assistantText', text: finalAnswer });
            emit({ type: 'final', text: finalAnswer });
            return finalAnswer;
        }

        emit({ type: 'toolCall', call });
        assistantMsg.toolCalls = [call];

        // 4. Execute the tool.
        const result: ToolCallResult = await executeTool(config, call);
        emit({ type: 'toolResult', result });

        // 5. Append the raw result back to history.
        const toolMsg: Message = {
            role: 'tool',
            content: result.ok ? result.output : `ERROR\n${result.output}`,
            toolCallId: call.id,
            toolName: call.name,
            toolResult: result
        };
        history.push(toolMsg);

        // 6. Context management: clamp oversized outputs, compact when deep.
        history = clampToolResultSize(history);
        if (history.length > config.compactionThreshold) {
            const { messages, removed } = compactConversation(history, config);
            if (removed > 0) {
                emit({ type: 'compaction', removed, remaining: messages.length });
                history = messages;
            }
        }
    }

    // The model never produced a final answer within budget.
    finalAnswer =
        'Reached maximum loop iterations without a final answer. ' +
        'The last tool results are above; please refine the request or raise maxIterations.';
    emit({ type: 'final', text: finalAnswer });
    return finalAnswer;
}
