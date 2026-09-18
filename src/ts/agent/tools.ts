/*
⚡ TinyCoder AI - Agent Harness Tools

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
import { exec } from 'child_process';
import { promisify } from 'util';
import {
    AgentConfig,
    ToolCall,
    ToolCallResult,
    ToolDefinition,
    ToolName,
    ToolToggles,
} from './types';

const execAsync = promisify(exec);

// ---------------------------------------------------------------------------
// Tool JSON Schemas
//
// The model sees these as plain JSON and must produce a `tool_calls` array.
// Schemas are intentionally small and self-describing for small quantized
// models (Qwen3.6-35B-A3B) that have 8-9 tok/s on CPU: keep descriptions
// terse and required fields explicit.
// ---------------------------------------------------------------------------

/** JSON Schema for `readFile` — read a text file (optionally a line range). */
const readFileSchema: ToolDefinition = {
    name: 'readFile',
    description:
        'Read the contents of a text file from the workspace. ' +
        'Use a relative path from the workspace root. ' +
        'Optionally pass "offset" (1-based) and "limit" (line count) to read a slice.',
    parameters: {
        type: 'object',
        properties: {
            path: {
                type: 'string',
                description: 'Path to the file, relative to the workspace root'
            },
            offset: {
                type: 'integer',
                description: 'Optional 1-based line number to start reading from'
            },
            limit: {
                type: 'integer',
                description: 'Optional number of lines to read (whole file when omitted)'
            }
        },
        required: ['path']
    }
};

/** JSON Schema for `writeFile` — create or fully overwrite a file. */
const writeFileSchema: ToolDefinition = {
    name: 'writeFile',
    description:
        'Create a new file or overwrite an existing file with the given content. ' +
        'The file is created including missing parent directories. ' +
        'Use patchFile instead when only specific lines must change.',
    parameters: {
        type: 'object',
        properties: {
            path: {
                type: 'string',
                description: 'Path to the file, relative to the workspace root'
            },
            content: {
                type: 'string',
                description: 'Full new file content'
            }
        },
        required: ['path', 'content']
    }
};

/** JSON Schema for `patchFile` — apply a unified diff (line-based replace). */
const patchFileSchema: ToolDefinition = {
    name: 'patchFile',
    description:
        'Apply a targeted, line-based edit to a file using a single SEARCH/REPLACE pair. ' +
        '"oldText" and "newText" must be exact literal slices of the file (whitespace matters). ' +
        'The first exact occurrence of oldText is replaced with newText. ' +
        'Preferred over writeFile for small, surgical changes.',
    parameters: {
        type: 'object',
        properties: {
            path: {
                type: 'string',
                description: 'Path to the file, relative to the workspace root'
            },
            oldText: {
                type: 'string',
                description: 'Exact existing text (including indentation) to be replaced'
            },
            newText: {
                type: 'string',
                description: 'Replacement text'
            }
        },
        required: ['path', 'oldText', 'newText']
    }
};

/** JSON Schema for `executeTerminalCommand` — run a shell command and capture output. */
const executeTerminalCommandSchema: ToolDefinition = {
    name: 'executeTerminalCommand',
    description:
        'Run a shell command in the project directory and return its stdout/stderr. ' +
        'Working directory is the configured workspace root. ' +
        'Long commands are truncated in the result. Use for builds, tests, git, etc.',
    parameters: {
        type: 'object',
        properties: {
            command: {
                type: 'string',
                description: 'The shell command to execute'
            },
            timeoutMs: {
                type: 'integer',
                description:
                    'Optional timeout in milliseconds (default 30000, max 120000). ' +
                    'Long-running commands (dev servers) are not supported — use 0 to skip waiting.'
            }
        },
        required: ['command']
    }
};

/** JSON Schema for `searchCodebase` — ripgrep-based pattern search. */
const searchCodebaseSchema: ToolDefinition = {
    name: 'searchCodebase',
    description:
        'Search the workspace for lines matching a regular expression (uses ripgrep). ' +
        'Returns matching lines with file and line numbers, up to a limit.',
    parameters: {
        type: 'object',
        properties: {
            pattern: {
                type: 'string',
                description: 'Regular expression to search for'
            },
            glob: {
                type: 'string',
                description: 'Optional file glob filter, e.g. "*.ts" or "!node_modules/**"'
            },
            maxResults: {
                type: 'integer',
                description: 'Maximum number of matches to return (default 50)'
            }
        },
        required: ['pattern']
    }
};

/** Builds the full tool list honoring per-tool toggles. */
export function buildToolDefinitions(toggles: ToolToggles): ToolDefinition[] {
    const defs: ToolDefinition[] = [];
    if (toggles.readFile) defs.push(readFileSchema);
    if (toggles.writeFile) defs.push(writeFileSchema);
    if (toggles.patchFile) defs.push(patchFileSchema);
    if (toggles.terminal) defs.push(executeTerminalCommandSchema);
    if (toggles.search) defs.push(searchCodebaseSchema);
    return defs;
}

// ---------------------------------------------------------------------------
// Path safety
// ---------------------------------------------------------------------------

/**
 * Resolve a workspace-relative path safely. Rejects absolute escapes outside
 * the workspace root so the model cannot read/write arbitrary host files.
 */
function resolveWithinWorkspace(root: string, relativePath: string): string {
    if (!root) {
        throw new Error('No workspace root is configured. Open a folder or set tinycoder.agent.workspaceRoot.');
    }
    const absolute = path.resolve(root, relativePath);
    const rel = path.relative(root, absolute);
    if (rel.startsWith('..') || path.isAbsolute(rel)) {
        throw new Error(`Path escapes workspace root: ${relativePath}`);
    }
    return absolute;
}

// ---------------------------------------------------------------------------
// Tool execution helpers (VS Code side)
// ---------------------------------------------------------------------------

const MAX_TEXT_LENGTH = 60_000;

function truncateText(text: string, maxLength: number = MAX_TEXT_LENGTH): string {
    if (text.length <= maxLength) {
        return text;
    }
    return `${text.slice(0, maxLength)}\n\n... [truncated ${text.length - maxLength} more chars]`;
}

function jsonStr(arg: unknown, fallback: string): string {
    if (typeof arg === 'string') {
        return arg;
    }
    if (arg === undefined || arg === null) {
        return fallback;
    }
    return String(arg);
}

/** Read a file (with optional line range) via vscode.workspace.fs. */
async function readFileTool(
    config: AgentConfig,
    args: Record<string, unknown>
): Promise<string> {
    const abs = resolveWithinWorkspace(config.workspaceRoot, jsonStr(args.path, ''));
    const offset = typeof args.offset === 'number' ? args.offset : undefined;
    const limit = typeof args.limit === 'number' ? args.limit : undefined;

    const data = await vscode.workspace.fs.readFile(vscode.Uri.file(abs));
    const text = Buffer.from(data).toString('utf8');
    if (offset === undefined && limit === undefined) {
        return truncateText(text);
    }

    const lines = text.split(/\r?\n/);
    const start = Math.max(1, Math.floor(offset || 1));
    const count = limit !== undefined ? Math.max(1, Math.floor(limit)) : lines.length;
    const slice = lines.slice(start - 1, start - 1 + count);
    const shown = slice.length;
    const total = lines.length;
    return `# ${path.basename(abs)}:${start}-${start + shown - 1} of ${total} lines\n${slice.join('\n')}`;
}

/** Write a file (create or overwrite) via vscode.workspace.fs. */
async function writeFileTool(
    config: AgentConfig,
    args: Record<string, unknown>
): Promise<string> {
    const abs = resolveWithinWorkspace(config.workspaceRoot, jsonStr(args.path, ''));
    const content = jsonStr(args.content, '');

    await vscode.workspace.fs.createDirectory(vscode.Uri.file(path.dirname(abs)));
    await vscode.workspace.fs.writeFile(vscode.Uri.file(abs), Buffer.from(content, 'utf8'));

    // If the file is open in an editor, refresh it so the user sees the change.
    const openDoc = vscode.workspace.textDocuments.find(
        (doc) => doc.uri.fsPath === abs && !doc.isDirty
    );
    if (openDoc) {
        await vscode.languages.setTextDocumentLanguage(openDoc, path.extname(abs).slice(1) || 'plaintext');
    }

    const bytes = Buffer.byteLength(content, 'utf8');
    return `OK: wrote ${bytes} bytes (${content.split(/\r?\n/).length} lines) to ${abs}`;
}

/** Apply a surgical oldText→newText replace to one file. */
async function patchFileTool(
    config: AgentConfig,
    args: Record<string, unknown>
): Promise<string> {
    const abs = resolveWithinWorkspace(config.workspaceRoot, jsonStr(args.path, ''));
    const oldText = jsonStr(args.oldText, '');
    const newText = jsonStr(args.newText, '');

    if (!oldText) {
        return 'ERROR: oldText must not be empty';
    }

    const data = await vscode.workspace.fs.readFile(vscode.Uri.file(abs));
    const text = Buffer.from(data).toString('utf8');

    const idx = text.indexOf(oldText);
    if (idx === -1) {
        return 'ERROR: oldText not found in file. The patch failed — the file may have changed. Re-read the file and retry with an exact match.';
    }

    const patched = text.slice(0, idx) + newText + text.slice(idx + oldText.length);
    await vscode.workspace.fs.writeFile(vscode.Uri.file(abs), Buffer.from(patched, 'utf8'));

    const removed = oldText.split(/\r?\n/).length;
    const added = newText.split(/\r?\n/).length;
    return `OK: replaced ${removed} lines with ${added} lines at character ${idx} in ${abs}`;
}

/**
 * Execute a command in the VS Code integrated terminal.
 * Returns a promise that resolves once the shell finishes (or times out).
 *
 * NOTE: `onDidEndTerminalShellExecution` requires VS Code 1.93+. On older
 * hosts the event never fires and we fall back to a timeout message. The
 * `child`/`auto` modes are the recommended deterministic path.
 */
async function runTerminalCommandVscode(
    config: AgentConfig,
    command: string,
    timeoutMs: number
): Promise<string> {
    const terminalName = config.terminalName || 'TinyCoder Agent';
    let terminal: vscode.Terminal | undefined = vscode.window.terminals.find(
        (t) => t.name === terminalName
    );
    if (!terminal) {
        terminal = vscode.window.createTerminal({
            name: terminalName,
            cwd: config.workspaceRoot || undefined
        });
    }
    terminal.show(true);

    let finished = false;
    const collected: string[] = [];
    const listener = vscode.window.onDidEndTerminalShellExecution(async (e) => {
        try {
            const text = e.execution.read();
            for await (const chunk of text) {
                collected.push(chunk);
            }
        } catch {
            // reading shell output can fail if the terminal is closed
        }
        finished = true;
        listener.dispose();
    });

    terminal.sendText(command, true);

    // Resolve on shell end, or after the timeout whichever comes first.
    await new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
            listener.dispose();
            resolve();
        }, timeoutMs);
        const flush = setInterval(() => {
            if (finished) {
                clearInterval(flush);
                clearTimeout(timer);
                listener.dispose();
                resolve();
            }
        }, 150);
    });

    const output = collected.join('');
    if (finished) {
        return truncateText(output || '(command completed; no output captured)');
    }
    return `(command may still be running in terminal "${terminalName}"; no completion event after ${timeoutMs}ms)\ncommand: ${command}\n${output ? `\n[captured so far]\n${truncateText(output)}` : ''}`.trim();
}

/** Execute a command in a child process for deterministic stdout/stderr capture. */
async function runTerminalCommandChild(
    config: AgentConfig,
    command: string,
    timeoutMs: number
): Promise<string> {
    const cwd = config.workspaceRoot || process.cwd();
    const timeout = timeoutMs > 0 ? Math.min(timeoutMs, 120_000) : 30_000;
    try {
        const { stdout, stderr } = await execAsync(command, { cwd, timeout, maxBuffer: 10 * 1024 * 1024 });
        const out = stdout || '';
        const err = stderr || '';
        const body = [
            out ? `[stdout]\n${out}` : '',
            err ? `[stderr]\n${err}` : ''
        ].filter(Boolean).join('\n');
        return truncateText(body || '(no output)');
    } catch (err: any) {
        const detail = err && err.stdout ? `\n[stdout]\n${String(err.stdout)}` : '';
        const detailErr = err && err.stderr ? `\n[stderr]\n${String(err.stderr)}` : '';
        return `ERROR: ${err?.message || err}\n${detail}${detailErr}`;
    }
}

/** Execute a terminal command honoring the configured capture mode. */
async function executeTerminalCommandTool(
    config: AgentConfig,
    args: Record<string, unknown>
): Promise<string> {
    const command = jsonStr(args.command, '');
    if (!command) {
        return 'ERROR: command must not be empty';
    }
    const rawTimeout = typeof args.timeoutMs === 'number' && args.timeoutMs > 0
        ? Math.floor(args.timeoutMs)
        : 30_000;
    const timeoutMs = Math.min(Math.max(rawTimeout, 1_000), 120_000);

    const mode = config.terminalMode;
    if (mode === 'child') {
        return runTerminalCommandChild(config, command, timeoutMs);
    }
    if (mode === 'vscode') {
        return runTerminalCommandVscode(config, command, timeoutMs);
    }
    // auto: prefer child for deterministic capture, fall back to terminal.
    try {
        return await runTerminalCommandChild(config, command, timeoutMs);
    } catch {
        return runTerminalCommandVscode(config, command, timeoutMs);
    }
}

/** Minimal glob→RegExp for the search tool (supports *, **, ? and [] classes). */
function globToRegExp(glob: string): RegExp {
    let re = '';
    for (let i = 0; i < glob.length; i++) {
        const ch = glob[i];
        if (ch === '*') {
            if (glob[i + 1] === '*') {
                re += '.*';
                i++;
            } else {
                re += '[^/]*';
            }
        } else if (ch === '?') {
            re += '[^/]';
        } else if (ch === '[') {
            const end = glob.indexOf(']', i + 1);
            if (end !== -1) {
                re += glob.slice(i, end + 1);
                i = end;
            } else {
                re += '\\[';
            }
        } else {
            re += ch.replace(/[.+^${}()|\\]/g, '\\$&');
        }
    }
    return new RegExp(`^${re}$`);
}

/** Ripgrep-backed codebase search with glob filtering. */
async function searchCodebaseTool(
    config: AgentConfig,
    args: Record<string, unknown>
): Promise<string> {
    const pattern = jsonStr(args.pattern, '');
    if (!pattern) {
        return 'ERROR: pattern must not be empty';
    }
    const glob = typeof args.glob === 'string' ? args.glob : undefined;
    const max = typeof args.maxResults === 'number'
        ? Math.min(Math.max(1, Math.floor(args.maxResults)), 200)
        : 50;

    let re: RegExp;
    try {
        re = new RegExp(pattern, 'i');
    } catch (err: any) {
        return `ERROR: invalid regex: ${err?.message || err}`;
    }

    const includeGlobs: string[] = [];
    const excludeGlobs: string[] = [];
    if (glob) {
        for (const part of glob.split(',')) {
            const g = part.trim().replace(/^\.\//, '');
            if (!g) continue;
            if (g.startsWith('!')) {
                excludeGlobs.push(g.slice(1));
            } else {
                includeGlobs.push(g);
            }
        }
    }
    const excludeRes = excludeGlobs.map(globToRegExp);

    const root = vscode.Uri.file(config.workspaceRoot);
    const patterns = includeGlobs.length > 0 ? includeGlobs : ['**/*'];
    const files = new Set<string>();

    for (const pat of patterns) {
        const uris = await vscode.workspace.findFiles(
            new vscode.RelativePattern(root, pat),
            '**/node_modules/**',
            10_000
        );
        for (const u of uris) {
            files.add(u.fsPath);
        }
    }

    const results: string[] = [];
    let matched = 0;

    for (const fsPath of files) {
        if (matched >= max) break;
        const rel = path.relative(config.workspaceRoot, fsPath).split(path.sep).join('/');
        if (excludeRes.some((ex) => ex.test(rel))) {
            continue;
        }
        try {
            const uri = vscode.Uri.file(fsPath);
            const data = await vscode.workspace.fs.readFile(uri);
            const text = Buffer.from(data).toString('utf8');
            const lines = text.split(/\r?\n/);
            for (let i = 0; i < lines.length; i++) {
                if (re.test(lines[i])) {
                    results.push(`${rel}:${i + 1}: ${lines[i].slice(0, 300)}`);
                    matched++;
                    if (matched >= max) break;
                }
            }
        } catch {
            // skip unreadable files
        }
    }

    if (results.length === 0) {
        return 'No matches found';
    }
    return results.join('\n');
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/**
 * Execute one parsed tool call.
 *
 * This is the only surface that touches the VS Code API; the AI-side
 * (`agentLoop`) stays free of any vscode import so the inference layer is
 * fully swappable.
 */
export async function executeTool(
    config: AgentConfig,
    call: ToolCall
): Promise<ToolCallResult> {
    const started = Date.now();
    const done = (output: string, ok = true, error?: string): ToolCallResult => ({
        toolCallId: call.id,
        name: call.name,
        ok,
        output,
        error,
        durationMs: Date.now() - started
    });

    try {
        switch (call.name) {
            case 'readFile':
                return done(await readFileTool(config, call.arguments));
            case 'writeFile':
                return done(await writeFileTool(config, call.arguments));
            case 'patchFile':
                return done(await patchFileTool(config, call.arguments));
            case 'executeTerminalCommand':
                return done(await executeTerminalCommandTool(config, call.arguments));
            case 'searchCodebase':
                return done(await searchCodebaseTool(config, call.arguments));
            default:
                return done(`ERROR: unknown tool "${call.name}"`, false, 'unknown tool');
        }
    } catch (err: any) {
        return done(`ERROR: ${err?.message || String(err)}`, false, err?.message || String(err));
    }
}

/** Generate a stable, ASCII tool call id. */
export function makeToolCallId(): string {
    return `call_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

/** List of tool names currently enabled. */
export function enabledToolNames(toggles: ToolToggles): ToolName[] {
    return buildToolDefinitions(toggles).map((d) => d.name);
}

/** Render a tool result as a compact string attached to the `tool` message. */
export function formatToolResult(result: ToolCallResult): string {
    return result.ok ? result.output : `ERROR\n${result.output}`;
}
