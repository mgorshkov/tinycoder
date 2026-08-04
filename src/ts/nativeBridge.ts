/*
⚡ TinyCoder AI

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
 * TinyCoder Native Bridge
 *
 * TypeScript wrapper around the C++ native addon (N-API).
 * Provides a clean async interface for model loading, text generation,
 * and status management.
 */

export interface HardwareInfo {
    cpu: {
        model: string;
        cores: number;
        ompThreads: number;
    };
    gpu: {
        available: boolean;
        name: string;
        cores: number;
        count?: number;
        memoryMB?: number;
    };
}

// The native addon is built by CMake and placed in the build output
let nativeAddon: any = null;
let nativeAddonError: string | null = null;

export interface NativeAddonStatus {
    available: boolean;
    path?: string;
    error?: string;
}

/**
 * Get the status of the native addon without throwing.
 * Useful for diagnostics before attempting model operations.
 */
export function getNativeAddonStatus(): NativeAddonStatus {
    if (nativeAddon) {
        return { available: true };
    }
    if (nativeAddonError) {
        return { available: false, error: nativeAddonError };
    }
    // Attempt to load
    try {
        getNativeAddon();
        return { available: true };
    } catch (err: any) {
        return { available: false, error: err.message };
    }
}

function getNativeAddon(): any {
    if (!nativeAddon) {
        try {
            // Try loading from various locations
            const paths = [
                // Development: out/../build/Release/
                require('path').join(__dirname, '..', 'build', 'Release', 'tinycoder_native.node'),
                // Development: out/../../build/Release/
                require('path').join(__dirname, '..', '..', 'build', 'Release', 'tinycoder_native.node'),
                // Installed extension: native/
                require('path').join(__dirname, 'native', 'tinycoder_native.node'),
            ];

            for (const p of paths) {
                try {
                    nativeAddon = require(p);
                    console.log('[TinyCoder] Loaded native addon from: ' + p);
                    break;
                } catch (err: any) {
                    console.log('[TinyCoder] Failed to load native addon from: ' + p + ' - ' + err.message);
                    continue;
                }
            }

            if (!nativeAddon) {
                nativeAddonError = 'Native addon not found. Tried: ' + paths.join(', ');
                throw new Error(nativeAddonError);
            }
        } catch (err) {
            console.error('[TinyCoder] Failed to load native addon:', err);
            throw err;
        }
    }
    return nativeAddon;
}

export interface ModelInfo {
    numLayers: number;
    hiddenSize: number;
    numAttentionHeads: number;
    numKVHeads: number;
    vocabSize: number;
    maxSeqLen: number;
}

export interface LoadResult {
    success: boolean;
    error?: string;
    modelInfo?: ModelInfo;
}

export interface InferenceParams {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
    repeatPenalty?: number;
    repeatLastN?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
    seed?: number;
}

export interface GenerateResult {
    tokens: number[];
    text: string;
    tokenCount: number;
}

export interface ModelStatus {
    loaded: boolean;
    generating: boolean;
    kvCacheSize?: number;
    config?: {
        nThreads: number;
        maxSeqLen: number;
    };
}

export type TokenCallback = (token: number, text: string) => boolean;

export type ProgressCallback = (progress: number, stage: string) => void;

/**
 * Load a model from a GGUF file with progress reporting.
 * Returns a Promise that resolves when loading is complete.
 */
export async function loadModel(
    modelPath: string,
    config?: { nThreads?: number; maxSeqLen?: number },
    onProgress?: ProgressCallback
): Promise<LoadResult> {
    try {
        const addon = getNativeAddon();
        const result: LoadResult = await addon.loadModel(modelPath, config || {}, onProgress);
        return result;
    } catch (err: any) {
        return { success: false, error: err.message };
    }
}

/**
 * Unload the current model.
 */
export function unloadModel(): boolean {
    try {
        const addon = getNativeAddon();
        const result = addon.unloadModel();
        return result.success;
    } catch {
        return false;
    }
}

/**
 * Check if a model is loaded.
 */
export function isModelLoaded(): boolean {
    try {
        const addon = getNativeAddon();
        return addon.isModelLoaded();
    } catch {
        return false;
    }
}

/**
 * Generate text with streaming callback.
 *
 * @param prompt - Input text prompt
 * @param params - Generation parameters
 * @param onToken - Optional callback for each generated token
 * @returns Promise with generation result
 */
export async function generate(
    prompt: string,
    params: InferenceParams = {},
    onToken?: TokenCallback
): Promise<GenerateResult> {
    const addon = getNativeAddon();

    // If no callback provided, use a no-op
    const callback = onToken || ((_token: number, _text: string) => true);

    return addon.generate(prompt, params, callback);
}
/**
 * Run a single inference request.
 * Wrapper around {@link generate} without a token callback.
 */
export async function runInference(
    prompt: string,
    params: InferenceParams = {}
): Promise<GenerateResult> {
    // Reuse generate implementation; no streaming callback needed.
    return generate(prompt, params);
}

/**
 * Stop the current generation.
 */
export function stopGeneration(): void {
    try {
        const addon = getNativeAddon();
        addon.stopGeneration();
    } catch {
        // Ignore
    }
}

/**
 * Get model status.
 */
export function getStatus(): ModelStatus {
    try {
        const addon = getNativeAddon();
        return addon.getStatus();
    } catch {
        return { loaded: false, generating: false };
    }
}

/**
 * Clear the KV cache.
 */
export function clearKVCache(): void {
    try {
        const addon = getNativeAddon();
        addon.clearKVCache();
    } catch {
        // Ignore
    }
}

/**
 * Get hardware information (CPU model/cores, GPU info).
 */
export function getHardwareInfo(): HardwareInfo {
    try {
        const addon = getNativeAddon();
        return addon.getHardwareInfo();
    } catch {
        return {
            cpu: { model: 'Unknown', cores: 1, ompThreads: 1 },
            gpu: { available: false, name: 'N/A', cores: 0 }
        };
    }
}
