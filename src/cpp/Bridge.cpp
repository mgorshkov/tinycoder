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

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <napi.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Model.hpp"

/// @brief N-API native addon bridge for TinyCoder.
///
/// This bridge exposes the C++ inference engine to Node.js/VS Code via N-API.
/// It provides:
/// - Model loading/unloading
/// - Text generation (streaming via callbacks)
/// - Configuration management
/// - Status reporting
/// - Hardware info (CPU/GPU)

namespace {

    // Global model instance (singleton per extension session)
    // Use shared_ptr so it can be safely captured in ThreadSafeFunction callbacks
    // that require copy-constructible lambdas.
    static std::shared_ptr<tinycoder::Model> g_model;
    static std::atomic<bool> g_generating{false};
    static std::atomic<bool> g_stopRequested{false};

    /// @brief Load a model from a GGUF file with progress reporting.
    /// JS: loadModel(modelPath: string, config?: object, onProgress?: (progress: number, stage: string) => void)
    ///     => Promise<{ success: boolean, error?: string, modelInfo?: object }>
    Napi::Value LoadModel(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "String argument expected (modelPath)")
                    .ThrowAsJavaScriptException();
            return env.Null();
        }

        std::string modelPath = info[0].As<Napi::String>().Utf8Value();

        // Parse optional config
        tinycoder::ModelConfig config;
        if (info.Length() >= 2 && info[1].IsObject()) {
            Napi::Object jsConfig = info[1].As<Napi::Object>();
            if (jsConfig.Has("nThreads"))
                config.nThreads = jsConfig.Get("nThreads").As<Napi::Number>().Uint32Value();
            if (jsConfig.Has("maxSeqLen"))
                config.maxSeqLen = jsConfig.Get("maxSeqLen").As<Napi::Number>().Uint32Value();
        }

        // Always create a ThreadSafeFunction — it is required to safely call V8 APIs
        // from the background thread when the load completes.
        Napi::Function jsCallback = info[2].IsFunction()
                                            ? info[2].As<Napi::Function>()
                                            : Napi::Function::New(env, [env](const Napi::CallbackInfo &) -> Napi::Value {
                                                  return env.Undefined();
                                              });

        Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
                env,
                jsCallback,
                "loadModelProgress",
                0,// unlimited queue
                1 // only one thread
        );

        bool hasProgressCallback = info.Length() >= 3 && info[2].IsFunction();

        // Create a deferred promise for async loading
        auto deferred = std::make_shared<Napi::Promise::Deferred>(env);
        Napi::Promise promise = deferred->Promise();

        // Run loading in a separate thread
        std::thread loadThread([modelPath, config, hasProgressCallback, tsfn, deferred]() {
            auto model = std::make_shared<tinycoder::Model>();

            // Set config on model before loading
            auto &cfg = const_cast<tinycoder::ModelConfig &>(model->config());
            cfg.nThreads = config.nThreads;
            cfg.maxSeqLen = config.maxSeqLen;

            // Progress callback that calls the ThreadSafeFunction
            auto progressCb = [tsfn](float progress, const std::string &stage) {
                if (tsfn) {
                    auto status = tsfn.BlockingCall([progress, stage](Napi::Env env, Napi::Function jsCallback) {
                        jsCallback.Call({Napi::Number::New(env, progress),
                                         Napi::String::New(env, stage)});
                    });
                    (void) status;
                }
            };

            std::string loadError;
            bool success = model->load(modelPath, &loadError,
                                       hasProgressCallback ? tinycoder::Model::ProgressCallback(progressCb) : nullptr);

            // Extract model info on the worker thread (no V8 calls needed)
            tinycoder::ModelConfig modelCfg = model->config();

            // Finalize on the main JS thread via BlockingCall — this is the ONLY safe
            // way to call V8 APIs (HandleScope, Object::New, deferred->Resolve, etc.)
            // from a background thread.
            tsfn.BlockingCall([success, loadError, model, modelCfg, deferred](Napi::Env env, Napi::Function /*jsCallback*/) {
                Napi::HandleScope scope(env);

                Napi::Object result = Napi::Object::New(env);
                result.Set("success", Napi::Boolean::New(env, success));

                if (!success) {
                    std::string errorMsg = "Failed to load model";
                    if (!loadError.empty()) {
                        errorMsg += ": " + loadError;
                    }
                    result.Set("error", Napi::String::New(env, errorMsg));
                } else {
                    // Store the model globally
                    g_model = std::move(model);

                    // Return model info
                    Napi::Object modelInfo = Napi::Object::New(env);
                    modelInfo.Set("numLayers", Napi::Number::New(env, modelCfg.numLayers));
                    modelInfo.Set("hiddenSize", Napi::Number::New(env, modelCfg.hiddenSize));
                    modelInfo.Set("numAttentionHeads", Napi::Number::New(env, modelCfg.numAttentionHeads));
                    modelInfo.Set("numKVHeads", Napi::Number::New(env, modelCfg.numKVHeads));
                    modelInfo.Set("vocabSize", Napi::Number::New(env, modelCfg.vocabSize));
                    modelInfo.Set("maxSeqLen", Napi::Number::New(env, modelCfg.maxSeqLen));
                    result.Set("modelInfo", modelInfo);
                }

                deferred->Resolve(result);
            });

            // Release the TSFN (safe to call from any thread)
            tsfn.Release();
        });

        loadThread.detach();

        return promise;
    }

    /// @brief Unload the current model.
    /// JS: unloadModel() => { success: boolean }
    Napi::Value UnloadModel(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();
        g_model.reset();
        g_generating = false;
        g_stopRequested = false;

        Napi::Object result = Napi::Object::New(env);
        result.Set("success", Napi::Boolean::New(env, true));
        return result;
    }

    /// @brief Check if model is loaded.
    /// JS: isModelLoaded() => boolean
    Napi::Value IsModelLoaded(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();
        return Napi::Boolean::New(env, g_model && g_model->isLoaded());
    }

    /// @brief Generate text with streaming callback.
    /// JS: generate(prompt: string, params: object, onToken: (token: number, text: string) => boolean)
    ///     => Promise<{ tokens: number[], text: string, tokenCount: number }>
    Napi::Value Generate(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();

        if (!g_model || !g_model->isLoaded()) {
            Napi::Error::New(env, "Model not loaded").ThrowAsJavaScriptException();
            return env.Null();
        }

        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "String argument expected (prompt)")
                    .ThrowAsJavaScriptException();
            return env.Null();
        }

        std::string prompt = info[0].As<Napi::String>().Utf8Value();

        // Parse inference parameters
        tinycoder::InferenceParams params;
        if (info.Length() >= 2 && info[1].IsObject()) {
            Napi::Object jsParams = info[1].As<Napi::Object>();

            if (jsParams.Has("maxTokens"))
                params.maxTokens = jsParams.Get("maxTokens").As<Napi::Number>().Int32Value();
            if (jsParams.Has("temperature"))
                params.temperature = jsParams.Get("temperature").As<Napi::Number>().FloatValue();
            if (jsParams.Has("topP"))
                params.topP = jsParams.Get("topP").As<Napi::Number>().FloatValue();
            if (jsParams.Has("topK"))
                params.topK = jsParams.Get("topK").As<Napi::Number>().FloatValue();
            if (jsParams.Has("repeatPenalty"))
                params.repeatPenalty = jsParams.Get("repeatPenalty").As<Napi::Number>().FloatValue();
            if (jsParams.Has("repeatLastN"))
                params.repeatLastN = jsParams.Get("repeatLastN").As<Napi::Number>().Int32Value();
            if (jsParams.Has("seed"))
                params.seed = jsParams.Get("seed").As<Napi::Number>().Uint32Value();
        }

        // Always create a ThreadSafeFunction — it is required to safely call V8 APIs
        // from the background thread when generation completes.
        Napi::Function jsCallback = info[2].IsFunction()
                                            ? info[2].As<Napi::Function>()
                                            : Napi::Function::New(env, [env](const Napi::CallbackInfo &) -> Napi::Value {
                                                  return env.Undefined();
                                              });

        Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
                env,
                jsCallback,
                "generateToken",
                0,// unlimited queue
                1 // only one thread
        );

        bool hasCallback = info.Length() >= 3 && info[2].IsFunction();

        // Create a deferred promise for async generation
        auto deferred = std::make_shared<Napi::Promise::Deferred>(env);
        Napi::Promise promise = deferred->Promise();

        // Run generation in a separate thread
        g_generating = true;
        g_stopRequested = false;

        std::thread genThread([prompt, params, hasCallback, tsfn, deferred]() {
            // Generate text with ThreadSafeFunction callback
            auto generated = g_model->generate(
                    prompt, params,
                    [tsfn](int32_t token, const std::string &text) -> bool {
                        if (g_stopRequested)
                            return false;

                        if (tsfn) {
                            bool shouldContinue = true;
                            auto status = tsfn.BlockingCall([token, text, &shouldContinue](Napi::Env env, Napi::Function jsCallback) {
                                auto result = jsCallback.Call({Napi::Number::New(env, token),
                                                               Napi::String::New(env, text)});
                                if (result.IsBoolean()) {
                                    shouldContinue = result.As<Napi::Boolean>().Value();
                                }
                            });
                            (void) status;
                            return shouldContinue;
                        }
                        return true;
                    });

            g_generating = false;

            // Build the full text on the worker thread (no V8 calls needed)
            std::string fullText;
            for (auto t: generated) {
                fullText += g_model->tokenizer().decodeToken(t);
            }

            // Finalize on the main JS thread via BlockingCall — this is the ONLY safe
            // way to call V8 APIs (HandleScope, Object::New, deferred->Resolve, etc.)
            // from a background thread.
            tsfn.BlockingCall([generated, fullText, deferred](Napi::Env env, Napi::Function /*jsCallback*/) {
                Napi::HandleScope scope(env);

                Napi::Object result = Napi::Object::New(env);

                // Token array
                Napi::Array tokenArray = Napi::Array::New(env, generated.size());
                for (size_t i = 0; i < generated.size(); ++i) {
                    tokenArray.Set(i, Napi::Number::New(env, generated[i]));
                }
                result.Set("tokens", tokenArray);

                result.Set("text", Napi::String::New(env, fullText));
                result.Set("tokenCount", Napi::Number::New(env, generated.size()));

                deferred->Resolve(result);
            });

            // Release the TSFN (safe to call from any thread)
            tsfn.Release();
        });

        genThread.detach();

        return promise;
    }

    /// @brief Stop the current generation.
    /// JS: stopGeneration() => void
    Napi::Value StopGeneration(const Napi::CallbackInfo &info) {
        g_stopRequested = true;
        return info.Env().Undefined();
    }

    /// @brief Get model status information.
    /// JS: getStatus() => { loaded: boolean, generating: boolean, kvCacheSize: number, config?: { nThreads: number, maxSeqLen: number } }
    Napi::Value GetStatus(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();

        Napi::Object status = Napi::Object::New(env);
        status.Set("loaded", Napi::Boolean::New(env, g_model && g_model->isLoaded()));
        status.Set("generating", Napi::Boolean::New(env, g_generating.load()));

        if (g_model && g_model->isLoaded()) {
            status.Set("kvCacheSize", Napi::Number::New(env, g_model->kvCacheSize()));

            const auto &cfg = g_model->config();
            Napi::Object config = Napi::Object::New(env);
            config.Set("nThreads", Napi::Number::New(env, cfg.nThreads));
            config.Set("maxSeqLen", Napi::Number::New(env, cfg.maxSeqLen));
            status.Set("config", config);
        }

        return status;
    }

    /// @brief Clear the KV cache.
    /// JS: clearKVCache() => void
    Napi::Value ClearKVCache(const Napi::CallbackInfo &info) {
        if (g_model) {
            g_model->clearKVCache();
        }
        return info.Env().Undefined();
    }

    /// @brief Get hardware information (CPU cores, GPU info).
    /// JS: getHardwareInfo() => { cpu: { model: string, cores: number }, gpu: { available: boolean, name: string, cores: number } }
    Napi::Value GetHardwareInfo(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();

        Napi::Object hwInfo = Napi::Object::New(env);

        // CPU info
        Napi::Object cpuInfo = Napi::Object::New(env);

        // Get CPU model from /proc/cpuinfo
        std::string cpuModel = "Unknown";
        FILE *fp = fopen("/proc/cpuinfo", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                std::string s(line);
                if (s.find("model name") != std::string::npos) {
                    auto colon = s.find(':');
                    if (colon != std::string::npos) {
                        cpuModel = s.substr(colon + 2);
                        // Trim trailing newline
                        while (!cpuModel.empty() && (cpuModel.back() == '\n' || cpuModel.back() == '\r')) {
                            cpuModel.pop_back();
                        }
                    }
                    break;
                }
            }
            fclose(fp);
        }
        cpuInfo.Set("model", Napi::String::New(env, cpuModel));

        // Get CPU core count
        long nCPUs = sysconf(_SC_NPROCESSORS_ONLN);
        cpuInfo.Set("cores", Napi::Number::New(env, static_cast<int32_t>(nCPUs > 0 ? nCPUs : 1)));

        // OpenMP threads
#ifdef _OPENMP
        cpuInfo.Set("ompThreads", Napi::Number::New(env, omp_get_max_threads()));
#else
        cpuInfo.Set("ompThreads", Napi::Number::New(env, 1));
#endif

        hwInfo.Set("cpu", cpuInfo);

        // GPU info
        Napi::Object gpuInfo = Napi::Object::New(env);

#ifdef USE_CUDA
        // Try to detect CUDA GPUs
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err == cudaSuccess && deviceCount > 0) {
            gpuInfo.Set("available", Napi::Boolean::New(env, true));
            gpuInfo.Set("count", Napi::Number::New(env, deviceCount));

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            gpuInfo.Set("name", Napi::String::New(env, prop.name));
            gpuInfo.Set("cores", Napi::Number::New(env, prop.multiProcessorCount * 128));// approximate CUDA cores
            gpuInfo.Set("memoryMB", Napi::Number::New(env, static_cast<int32_t>(prop.totalGlobalMem / (1024 * 1024))));
        } else {
            gpuInfo.Set("available", Napi::Boolean::New(env, false));
            gpuInfo.Set("name", Napi::String::New(env, "N/A"));
            gpuInfo.Set("cores", Napi::Number::New(env, 0));
        }
#else
        gpuInfo.Set("available", Napi::Boolean::New(env, false));
        gpuInfo.Set("name", Napi::String::New(env, "Not available (no CUDA)"));
        gpuInfo.Set("cores", Napi::Number::New(env, 0));
#endif

        hwInfo.Set("gpu", gpuInfo);

        return hwInfo;
    }

}// anonymous namespace

/// @brief Initialize the native addon module.
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Auto-detect CPU count and set OpenMP threads to match.
    // Uses sysconf (_SC_NPROCESSORS_ONLN) on Linux to get the number of
    // online (available) processors. This ensures all parallel loops use
    // the full CPU count without requiring an external env variable.
#ifdef _OPENMP
    long nCPUs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nCPUs > 0) {
        omp_set_num_threads(static_cast<int>(nCPUs));
    }
#endif

    exports.Set("loadModel", Napi::Function::New(env, LoadModel));
    exports.Set("unloadModel", Napi::Function::New(env, UnloadModel));
    exports.Set("isModelLoaded", Napi::Function::New(env, IsModelLoaded));
    exports.Set("generate", Napi::Function::New(env, Generate));
    exports.Set("stopGeneration", Napi::Function::New(env, StopGeneration));
    exports.Set("getStatus", Napi::Function::New(env, GetStatus));
    exports.Set("clearKVCache", Napi::Function::New(env, ClearKVCache));
    exports.Set("getHardwareInfo", Napi::Function::New(env, GetHardwareInfo));
    return exports;
}

NODE_API_MODULE(tinycoder_native, Init)
