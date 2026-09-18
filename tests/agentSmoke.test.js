/*
⚡ TinyCoder AI - Agent Harness Smoke Test

Minimal Node assertion harness for the pure (VS Code-independent) parts of the
agent loop: tool-call JSON parsing and conversation compaction. Run with:

    node tests/agentSmoke.test.js

Requires the extension to be compiled first (`npm run compile`), which emits
out/agent/*.js.
*/

const assert = require('assert');
const Module = require('module');

// Minimal vscode module stub so the compiled agent modules load under plain
// Node (they only import vscode for the tool executors, which the smoke test
// does not invoke).
const vscodeStub = {
    workspace: {
        fs: {},
        findFiles: async () => [],
        asRelativePath: (p) => p,
        getConfiguration: () => ({ get: () => undefined })
    },
    window: {
        createTerminal: () => ({ show: () => {}, sendText: () => {} }),
        createOutputChannel: () => ({ appendLine: () => {}, show: () => {}, clear: () => {} }),
        onDidEndTerminalShellExecution: () => ({ dispose: () => {} }),
        terminals: []
    },
    Uri: { file: (p) => ({ fsPath: p }) },
    languages: { setTextDocumentLanguage: async () => {} },
    ThemeColor: class {}
};
const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
    if (request === 'vscode') {
        return vscodeStub;
    }
    return originalLoad.apply(this, arguments);
};

const { parseToolCall } = require('../out/agent/agentLoop');
const { compactConversation, clampToolResultSize } = require('../out/agent/compaction');
const { buildToolDefinitions } = require('../out/agent/tools');

let passed = 0;
function test(name, fn) {
    try {
        fn();
        passed++;
        console.log(`  ✓ ${name}`);
    } catch (err) {
        console.error(`  ✗ ${name}`);
        console.error(err);
        process.exitCode = 1;
    }
}

test('parseToolCall: extracts wrapper JSON tool call', () => {
    const raw = `Some reasoning\n{"tool_call":{"name":"readFile","arguments":{"path":"src/ts/extension.ts"}}}`;
    const call = parseToolCall(raw);
    assert.ok(call, 'expected a tool call');
    assert.strictEqual(call.name, 'readFile');
    assert.deepStrictEqual(call.arguments, { path: 'src/ts/extension.ts' });
    assert.ok(call.id.startsWith('call_'));
});

test('parseToolCall: extracts bare JSON tool call', () => {
    const raw = `{"name":"writeFile","arguments":{"path":"x.ts","content":"hello"}}`;
    const call = parseToolCall(raw);
    assert.ok(call);
    assert.strictEqual(call.name, 'writeFile');
    assert.strictEqual(call.arguments.content, 'hello');
});

test('parseToolCall: returns null for final answer text', () => {
    assert.strictEqual(parseToolCall('I fixed the bug. Done.'), null);
    assert.strictEqual(parseToolCall(''), null);
});

test('parseToolCall: returns null for malformed JSON', () => {
    const raw = `{"tool_call":{"name":"readFile","arguments":{ broken }}}`;
    assert.strictEqual(parseToolCall(raw), null);
});

test('compactConversation: trims old tool outputs, keeps intent', () => {
    const config = {
        compactionThreshold: 6,
        compactionPreserveRecent: 3,
    };
    const history = [];
    // 10 messages: 2 (user/assistant) + interleaved tool pairs.
    history.push({ role: 'user', content: 'u0' });
    history.push({ role: 'assistant', content: 'a0', toolCalls: [{ id: 'c1', name: 'readFile', arguments: {} }] });
    history.push({ role: 'tool', content: 'LONG'.repeat(500), toolCallId: 'c1', toolName: 'readFile' });
    history.push({ role: 'assistant', content: 'a1', toolCalls: [{ id: 'c2', name: 'writeFile', arguments: {} }] });
    history.push({ role: 'tool', content: 'second result', toolCallId: 'c2', toolName: 'writeFile' });
    history.push({ role: 'assistant', content: 'a2', toolCalls: [{ id: 'c3', name: 'readFile', arguments: {} }] });
    history.push({ role: 'tool', content: 'third result', toolCallId: 'c3', toolName: 'readFile' });
    history.push({ role: 'assistant', content: 'a3', toolCalls: [{ id: 'c4', name: 'patchFile', arguments: {} }] });
    history.push({ role: 'tool', content: 'fourth result', toolCallId: 'c4', toolName: 'patchFile' });
    history.push({ role: 'assistant', content: 'a4' });

    const { messages, removed } = compactConversation(history, config);
    assert.ok(removed > 0, 'should have removed messages');
    assert.ok(messages.length < history.length, 'should be shorter');
    // No raw 2000-char blob left.
    const allText = JSON.stringify(messages);
    assert.ok(!allText.includes('LONG'.repeat(500)), 'raw tool output should be gone');
    // The tail (assistant pairing + final answer) must survive.
    assert.ok(messages.some((m) => m.content.includes('planned tool calls')), 'assistant tool-call markers present');
    assert.strictEqual(messages[messages.length - 1].content, 'a4');
});

test('clampToolResultSize: truncates oversized single tool result', () => {
    const big = 'x'.repeat(10_000);
    const messages = [
        { role: 'tool', content: big, toolResult: { output: big, ok: true } }
    ];
    const clamped = clampToolResultSize(messages);
    assert.ok(clamped[0].content.length < big.length);
    assert.strictEqual(clamped[0].truncated, true);
});

test('buildToolDefinitions: honors toggles', () => {
    const all = buildToolDefinitions({ readFile: true, writeFile: true, patchFile: true, terminal: true, search: true });
    assert.strictEqual(all.length, 5);
    const none = buildToolDefinitions({ readFile: false, writeFile: false, patchFile: false, terminal: false, search: false });
    assert.strictEqual(none.length, 0);
    const partial = buildToolDefinitions({ readFile: true, writeFile: false, patchFile: false, terminal: true, search: false });
    assert.deepStrictEqual(partial.map((t) => t.name), ['readFile', 'executeTerminalCommand']);
});

test('parseToolCall: handles thinking blocks + fenced JSON', () => {
    const fence = '```';
    const raw = `<thinking>I should read the file first</thinking>\n${fence}json\n{"tool_call":{"name":"readFile","arguments":{"path":"a.ts"}}}\n${fence}\n`;
    const call = parseToolCall(raw);
    assert.ok(call, 'expected a tool call despite thinking + fences');
    assert.strictEqual(call.name, 'readFile');
    assert.deepStrictEqual(call.arguments, { path: 'a.ts' });
});

test('parseToolCall: strips trailing stray tokens after JSON', () => {
    const raw = '{"tool_call":{"name":"writeFile","arguments":{"path":"x.txt","content":"hi"}}}`\nSome trailing text 1';
    const call = parseToolCall(raw);
    assert.ok(call);
    assert.strictEqual(call.name, 'writeFile');
});

console.log(`\n${passed} tests passed.`);
if (process.exitCode) {
    process.exit(process.exitCode);
}
