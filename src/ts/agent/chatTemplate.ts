/*
⚡ TinyCoder AI - Agent Harness Chat Template

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
 * Prompt assembly for the agent harness.
 *
 * The tinycoder-inference library accepts a raw prompt string and renders the
 * chat template internally (Jinja from GGUF metadata) — it is *not*
 * message-list aware. The harness therefore flattens the conversation into a
 * JSON message array embedded in the prompt, and instructs the model to reply
 * with either a final answer or a JSON tool-call block that the loop parses
 * from the raw output.
 */

import { Message } from './types';

/**
 * Serialize a Message[] into the JSON `[{"role":...,"content":...}, ...]`
 * payload that the model sees inside the conversation section.
 */
export function serializeMessages(messages: Message[]): string {
    return JSON.stringify(
        messages.map((m) => {
            const obj: Record<string, unknown> = { role: m.role, content: m.content };
            if (m.toolCallId) obj.toolCallId = m.toolCallId;
            if (m.toolName) obj.toolName = m.toolName;
            if (m.toolCalls) obj.toolCalls = m.toolCalls;
            if (m.truncated) obj.truncated = true;
            return obj;
        })
    );
}

/**
 * Join the base system prompt with the conversation history to build the
 * single raw prompt string that is sent to the inference library.
 */
export function renderPrompt(systemPrompt: string, messages: Message[]): string {
    if (messages.length === 0) {
        return systemPrompt;
    }
    return `${systemPrompt}\n\n=== CONVERSATION (JSON messages array) ===\n${serializeMessages(messages)}`;
}
