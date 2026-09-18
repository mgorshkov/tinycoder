/*
⚡ TinyCoder AI - Agent Harness Inference Adapter

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
 * Inference adapter.
 *
 * This is the only place that knows about the local C++ inference engine
 * (nativeBridge's `generate`, which wraps tinycoder-inference). The ReAct
 * loop calls {@link callAPInference} with a fully-rendered prompt string and
 * receives plain text + token count. To swap in a different backend (HTTP,
 * ollama, ...) implement the same signature and register it.
 */

import { runInference, InferenceParams } from '../nativeBridge';
import { AgentConfig, InferenceConfig, InferenceResult } from './types';

// Lazy registration slot so tests / alternate backends can replace the
// default native-bridge implementation.
let inferenceImpl: APInferenceFn | undefined;

/** Signature every inference backend must satisfy. */
export type APInferenceFn = (
    prompt: string,
    config: AgentConfig
) => Promise<InferenceResult>;

/** Register a custom inference implementation (e.g. HTTP endpoint). */
export function registerAPInference(impl: APInferenceFn): void {
    inferenceImpl = impl;
}

/** Convert the agent's inference settings into nativeBridge InferenceParams. */
export function toBridgeParams(cfg: InferenceConfig): InferenceParams {
    const params: InferenceParams = {
        maxTokens: cfg.maxTokens,
        temperature: cfg.temperature,
        topP: cfg.topP,
        topK: cfg.topK,
        repeatPenalty: cfg.repeatPenalty,
        repeatLastN: cfg.repeatLastN
    };
    if (cfg.seed !== undefined && cfg.seed > 0) {
        params.seed = cfg.seed;
    }
    return params;
}

/**
 * Default implementation that talks to the local tinycoder-inference engine
 * through nativeBridge.
 */
export async function callNativeInference(
    prompt: string,
    config: AgentConfig
): Promise<InferenceResult> {
    const result = await runInference(prompt, toBridgeParams(config.inference));
    return { text: result.text, tokenCount: result.tokenCount };
}

/**
 * The abstraction the ReAct loop uses. Falls back to the native bridge when
 * no custom backend is registered.
 */
export async function callAPInference(
    prompt: string,
    config: AgentConfig
): Promise<InferenceResult> {
    if (inferenceImpl) {
        return inferenceImpl(prompt, config);
    }
    return callNativeInference(prompt, config);
}
