/*
⚡ TinyCoder AI - Agent Harness Loop Integration Test

Exercises `runAgentLoop` end-to-end with a stubbed `vscode` module and a mock
inference backend. The real file tools (readFile/writeFile/patchFile) run
against a temporary directory through the VS Code-stub adapter (which maps
`vscode.workspace.fs` to the real `fs`).

Run: node tests/agentLoop.test.js  (after `npm run compile`)
*/

const assert = require('assert');
const Module = require('module');
const fs = require('fs');
const os = require('os');
const path = require('path');

// ---- vscode stub backed by real fs ----
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'tinycoder-agent-'));
const realFs = {
    readFile: async (uri) => Buffer.from(fs.readFileSync(uri.fsPath, 'utf8')),
    writeFile: async (uri, data) => {
        fs.mkdirSync(path.dirname(uri.fsPath), { recursive: true });
        fs.writeFileSync(uri.fsPath, Buffer.from(data));
    },
    createDirectory: async (uri) => fs.mkdirSync(uri.fsPath, { recursive: true })
};

const vscodeStub = {
    workspace: {
        fs: realFs,
        findFiles: async () => [],
        asRelativePath: (p) => p,
        getConfiguration: () => ({ get: () => undefined }),
        workspaceFolders: [{ uri: { fsPath: tmpRoot } }],
        textDocuments: []
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
    if (request === 'vscode') return vscodeStub;
    return originalLoad.apply(this, arguments);
};

const { runAgentLoop } = require('../out/agent/agentLoop');
const { registerAPInference } = require('../out/agent/inference');

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

(async () => {
    // ---- Mock inference: scripted replies ----
    const tmpFile = path.join(tmpRoot, 'notes.txt');
    fs.writeFileSync(tmpFile, 'alpha\nbeta\ngamma\n');

    // Each "model turn" returns output based on the tool results it has seen.
    let turn = 0;
    registerAPInference(async (prompt) => {
        turn++;
        if (turn === 1) {
            // First: read the file.
            return { text: `{"tool_call":{"name":"readFile","arguments":{"path":"notes.txt"}}}`, tokenCount: 10 };
        }
        if (turn === 2) {
            // Second: saw the file; patch it.
            return { text: `{"tool_call":{"name":"patchFile","arguments":{"path":"notes.txt","oldText":"beta","newText":"BETA!"}}}`, tokenCount: 10 };
        }
        // Finally answer.
        return { text: 'Done, I patched notes.txt.', tokenCount: 5 };
    });

    const config = {
        modelPath: '/dev/null',
        workspaceRoot: tmpRoot,
        terminalName: 'test',
        terminalMode: 'child',
        maxIterations: 5,
        compactionThreshold: 20,
        compactionPreserveRecent: 5,
        toolToggles: { readFile: true, writeFile: true, patchFile: true, terminal: false, search: false },
        systemPromptOverride: '',
        inference: { maxTokens: 128, temperature: 0, topP: 1, topK: 1, repeatPenalty: 1, repeatLastN: 0, maxSeqLen: 1024, nThreads: 1 }
    };

    {
        const events = [];
        const final = await runAgentLoop('Change beta to BETA!', config, (e) => events.push(e));
        assert.strictEqual(final, 'Done, I patched notes.txt.');
        assert.strictEqual(fs.readFileSync(tmpFile, 'utf8'), 'alpha\nBETA!\ngamma\n');
        assert.ok(events.some((e) => e.type === 'iteration'));
        assert.ok(events.some((e) => e.type === 'toolResult' && e.result.ok));
        passed++;
        console.log('  ✓ runAgentLoop: reads, patches, then answers');
    }

    // Cleanup
    fs.rmSync(tmpRoot, { recursive: true, force: true });
    console.log(`\n${passed} tests passed.`);
    process.exit(process.exitCode || 0);
})();
