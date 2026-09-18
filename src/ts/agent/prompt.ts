/*
⚡ TinyCoder AI - Agent Harness System Prompt

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
import { buildToolDefinitions } from './tools';

/**
 * Render the tool protocol section of the system prompt as JSON schemas.
 * The model receives the schemas inline so it can produce well-formed
 * tool-call JSON without a separate tool-call API.
 */
export function renderToolProtocol(tools: ReturnType<typeof buildToolDefinitions>): string {
    const body = tools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: t.parameters
    }));
    return JSON.stringify(body, null, 2);
}

/**
 * Build the base system prompt for the ReAct loop.
 * Keeps the tool call contract explicit:
 *  - think inside <reasoning> tags
 *  - either answer directly, or emit a single JSON tool call:
 *    {"tool_call":{"name":..., "arguments":{...}}}
 */
export function buildSystemPrompt(config: AgentConfig): string {
    const tools = buildToolDefinitions(config.toolToggles);
    const toolNames = tools.map((t) => t.name).join(', ');

    const header =
        config.systemPromptOverride && config.systemPromptOverride.trim().length > 0
            ? config.systemPromptOverride.trim()
            : [
                  'You are TinyCoder, an autonomous coding agent running inside the VS Code extension.',
                  'You reason about the user\'s request and take actions using the provided tools.',
                  'You operate inside a workspace. All file paths are relative to the workspace root.',
                  'You may only use the tools listed below. Do not invent tools.',
                  'When you are done, provide a final answer to the user.'
              ].join('\n');

    return [
        header,
        '',
        '=== TOOLS ===',
        'You may call any of the following tools. ' +
            `Available: ${toolNames}.`,
        '',
        'Tool call JSON format (exactly one object, no extra text around it):',
        '{"tool_call":{"name":"<tool name>","arguments":{...}}}',
        '',
        'Tool schemas:',
        renderToolProtocol(tools),
        '',
        '=== PROTOCOL ===',
        [
            '1. Always start your turn with a short <reasoning>...</reasoning> block explaining your plan.',
            '2. Then EITHER provide the final answer (plain text, no JSON),',
            '   OR emit exactly one tool call as the JSON object shown above.',
            '3. After a tool call, there will be a "tool" message with the result.',
            '   Continue reasoning and either call another tool or give the final answer.',
            '4. Never repeat the same exact tool call twice in a row without changing arguments.',
            '5. Keep final answers concise and reference the files you changed.'
        ].join('\n'),
        '',
        '=== CONVERSATION ==='
    ].join('\n');
}

/**
 * A message role wrapper for compaction output (paraphrase of history).
 */
export function summarizeHistory(messages: Message[]): string {
    const brief = messages.map((m) => {
        const head = m.content.split('\n')[0]?.slice(0, 120) ?? '';
        return `${m.role}: ${head}`;
    });
    return `[compacted history]\n${brief.join('\n')}`;
}
