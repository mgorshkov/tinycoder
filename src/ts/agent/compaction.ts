/*
⚡ TinyCoder AI - Agent Harness Compaction

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

import { AgentConfig, Message } from './types';

/**
 * Context management for the ReAct loop.
 *
 * Long agent runs explode the token count (each tool result is raw terminal
 * or file output). Rather than truncating blindly, we keep the system prompt
 * and the latest N messages verbatim, drop the *raw* tool outputs of older
 * steps, and collapse old assistant tool-call turns into one-line markers so
 * the original file-edit intent stays in the conversation.
 */

const MAX_SINGLE_TOOL_OUTPUT_CHARS = 4_000;

/** Cap the size of a single stored tool result (defense against big dumps). */
export function clampToolResultSize(messages: Message[]): Message[] {
    return messages.map((m) => {
        if (m.role !== 'tool' || !m.toolResult) {
            return m;
        }
        const out = m.toolResult.output;
        if (out.length <= MAX_SINGLE_TOOL_OUTPUT_CHARS) {
            return m;
        }
        return {
            ...m,
            content: `${out.slice(0, MAX_SINGLE_TOOL_OUTPUT_CHARS)}\n... [truncated ${out.length - MAX_SINGLE_TOOL_OUTPUT_CHARS} chars]`,
            truncated: true,
            toolResult: {
                ...m.toolResult,
                output: `${out.slice(0, MAX_SINGLE_TOOL_OUTPUT_CHARS)}... [truncated]`
            }
        };
    });
}

/**
 * Compact the conversation once it grows past `config.compactionThreshold`
 * messages. Returns the trimmed array plus the number of messages removed.
 *
 * The keep-alive window is `config.compactionPreserveRecent`; everything older
 * has its raw tool outputs dropped and assistant tool calls collapsed into a
 * one-line marker, so the array shrinks and the token count drops while the
 * original edits' intent stays in the conversation.
 */
export function compactConversation(
    messages: Message[],
    config: AgentConfig
): { messages: Message[]; removed: number } {
    const threshold = Math.max(4, config.compactionThreshold);
    const preserve = Math.max(2, config.compactionPreserveRecent);

    if (messages.length <= threshold) {
        return { messages, removed: 0 };
    }

    // Walk the keep-window start backward so a `tool` result message is never
    // separated from the `assistant` tool-call message that requested it.
    let keepStart = messages.length - preserve;
    while (keepStart > 0 && messages[keepStart].role === 'tool') {
        keepStart--;
    }

    const older = messages.slice(0, keepStart);
    const recent = messages.slice(keepStart);

    // Drop old tool output messages entirely; collapse old assistant tool-call
    // turns into a single marker; keep user messages verbatim.
    let removed = 0;
    const collapsedOld: Message[] = [];
    for (const m of older) {
        if (m.role === 'tool') {
            removed++;
            continue;
        }
        if (m.role === 'assistant' && m.toolCalls && m.toolCalls.length > 0) {
            collapsedOld.push({
                ...m,
                toolCalls: undefined,
                content: `[planned tool calls: ${m.toolCalls.map((c) => c.name).join(', ')}]`
            });
            continue;
        }
        collapsedOld.push(m);
    }

    return {
        messages: [...collapsedOld, ...recent],
        removed
    };
}
