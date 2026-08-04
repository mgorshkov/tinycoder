// -----------------------------------------------------------------------------
// tinycoder_bench — llama-bench-parity generation benchmark.
//
// WHY THIS EXISTS (2026-08-29):
//   The gtest Q2 (SampleQuestionTest) measures a SINGLE COLD 64-token run with
//   SAMPLING (temp 0.7, topK 40, topP 0.9, repeatPenalty 1.1). llama-bench's
//   `tg64` is the mean of WARM REPEAT runs with GREEDY decoding (temp 0) and a
//   FIXED token stream. The two were previously quoted side-by-side in the plan
//   (plans/generation_optimizations.md §6.6/§6.7), which is a measurement
//   confound: the cold first run pays page-cache + thread-pool startup and the
//   sampled token stream (and therefore the KV-cache position pattern) differs
//   run-to-run.
//
//   This harness reproduces llama-bench's tg protocol exactly so TinyCoder
//   numbers are comparable to `llama-bench -p 64 -n 64`:
//     - fixed 64-token prompt (llama-bench's pp64/tg64 uses a 64-token prompt)
//     - GREEDY decode: plain argmax over logits, repeatPenalty = 1.0 (llama-bench
//       defaults: repeat_penalty 1.0, no topK/topP, temp 0)
//     - timing EXCLUDES prefill: only the decode loop (per-token forward)
//       is timed, exactly like llama-bench's tgen
//     - warmup run first (llama-bench runs a warmup batch), then N reps,
//       reporting mean ± stdev over reps (llama-bench reports mean/stddev of
//       the reps).
//
// Usage:
//   tinycoder_bench [--model <path>] [--n-prompts 64] [--n-gen 64]
//                   [--reps 5] [--threads N]
//   TINYCODER_THREADS=N env var is honored by ThreadPool::recommendedThreadCount.
// -----------------------------------------------------------------------------

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Model.hpp"
#include "ModelConfig.hpp"
#include "ThreadPool.hpp"

namespace {

    struct BenchOpts {
        std::string modelPath;
        int32_t nPrompts = 64;  // prompt token count (llama-bench pp64)
        int32_t nGen = 64;      // generated token count (llama-bench tg64)
        int32_t reps = 5;       // warm repeats (llama-bench default 5)
        int32_t threads = 0;    // 0 = recommendedThreadCount()
        bool gpu = false;       // --gpu: use the CUDA offload engine
    };

    double nowMs() {
        return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
    }

    std::vector<int32_t> buildPrompt(int32_t nTokens) {
        // llama-bench uses a fixed pool of random latin tokens to build the
        // pp64 prompt. Any deterministic 64-token stream is equivalent for
        // bandwidth measurement purposes; use a fixed seed so reps are
        // bit-identical AND re-runs are reproducible.
        std::vector<int32_t> t;
        t.reserve(nTokens);
        uint64_t s = 0x5EED123456789ULL;
        for (int32_t i = 0; i < nTokens; ++i) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            // BPE ids in the low range (safe general-token range, avoids special
            // tokens that may be filtered).
            t.push_back(static_cast<int32_t>(s % 3000) + 10);
        }
        return t;
    }

    int32_t argmaxTop(const float *logits, int32_t vocabSize) {
        int32_t best = 0;
        float bestVal = logits[0];
        for (int32_t i = 1; i < vocabSize; ++i) {
            if (logits[i] > bestVal) {
                bestVal = logits[i];
                best = i;
            }
        }
        return best;
    }

    struct RepResult {
        double ms = 0.0;    // decode loop wall time (excludes prefill)
        int32_t tokens = 0; // generated tokens this rep
    };

    // Run one llama-bench-style benchmark rep: prefill (untimed), then decode
    // nGen tokens greedily, timing ONLY the decode loop.
    RepResult runOneRep(tinycoder::Model &model, const std::vector<int32_t> &prompt,
                        int32_t nGen, double &prefillMsOut,
                        const int32_t vocabSize) {
        // Prefill on a cleared cache, timed separately as a diagnostic
        // (EXCLUDED from the decode rate, exactly like llama-bench's tgen).
        model.clearKVCache();
        auto t0 = nowMs();
        auto logits = model.forward(prompt, false);
        auto t1 = nowMs();
        prefillMsOut = t1 - t0;

        // Decode loop — THE measured region (llama-bench tgen).
        double tDec0 = nowMs();
        int32_t tok = argmaxTop(logits.data(), vocabSize);
        int32_t gen = 0;
        for (int32_t i = 0; i < nGen; ++i) {
            logits = model.forward({tok}, false);
            tok = argmaxTop(logits.data(), vocabSize);
            ++gen;
        }
        double tDec1 = nowMs();

        RepResult r;
        r.ms = tDec1 - tDec0;
        r.tokens = gen;
        return r;
    }

    void printUsage(const char *argv0) {
        std::fprintf(stderr,
                     "Usage: %s [options]\n"
                     "  --model <path>   GGUF model (default: $TINYCODER_MODEL_PATH)\n"
                     "  --n-prompts N    prompt token count (default 64)\n"
                     "  --n-gen N        generated token count (default 64)\n"
                     "  --reps N         warm repeat count (default 5)\n"
                     "  --threads N      thread pool size (default: logical CPUs)\n"
#ifdef USE_CUDA
                     "  --gpu            use the CUDA offload engine (default when built with CUDA)\n"
#endif
                     ,
                     argv0);
    }

} // namespace

int main(int argc, char **argv) {
    BenchOpts opt;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", argv[i]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--model") == 0) {
            opt.modelPath = next();
        } else if (std::strcmp(argv[i], "--n-prompts") == 0) {
            opt.nPrompts = std::atoi(next().c_str());
        } else if (std::strcmp(argv[i], "--n-gen") == 0) {
            opt.nGen = std::atoi(next().c_str());
        } else if (std::strcmp(argv[i], "--reps") == 0) {
            opt.reps = std::atoi(next().c_str());
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            opt.threads = std::atoi(next().c_str());
#ifdef USE_CUDA
        } else if (std::strcmp(argv[i], "--gpu") == 0) {
            opt.gpu = true;
#endif
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 2;
        }
    }

    if (opt.modelPath.empty()) {
        const char *env = std::getenv("TINYCODER_MODEL_PATH");
        if (env != nullptr && env[0] != '\0') {
            opt.modelPath = env;
        } else {
            opt.modelPath = "/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf";
        }
    }
#ifdef USE_CUDA
    // --gpu: force the CUDA offload engine on (the default in CUDA builds;
    // this also overrides a TINYCODER_GPU=0 opt-out e.g. for a CPU baseline).
    // Must happen before Model::load() so weights are uploaded once at first
    // forward.
    if (opt.gpu) {
        std::fprintf(stderr, "[bench] CUDA offload engine enabled\n");
        setenv("TINYCODER_GPU", "1", 1);
    }
#endif
    if (opt.reps < 1) opt.reps = 1;

    size_t threads = opt.threads > 0
                             ? static_cast<size_t>(opt.threads)
                             : tinycoder::ThreadPool::recommendedThreadCount();
    tinycoder::ThreadPool::instance().initialize(threads);

    std::fprintf(stderr, "Loading model: %s\n", opt.modelPath.c_str());
    tinycoder::Model model;
    std::string err;
    if (!model.load(opt.modelPath, &err)) {
        std::fprintf(stderr, "FATAL: model load failed: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "  %s, %u layers, hidden %u, intermediate %u, vocab %u\n",
                 model.config().modelName.c_str(), model.config().numLayers,
                 model.config().hiddenSize, model.config().intermediateSize,
                 model.config().vocabSize);

    const auto prompt = buildPrompt(opt.nPrompts);
    const int32_t vocab = static_cast<int32_t>(model.config().vocabSize);

    std::fprintf(stderr,
                 "Bench: pp=%d tg=%d reps=%d threads=%zu  (greedy, decode-only "
                 "timing, warmup+reps)\n\n",
                 opt.nPrompts, opt.nGen, opt.reps, threads);

    // ---- Warmup rep (llama-bench runs a warmup batch first) ----
    {
        double prefillMs = 0.0;
        auto r = runOneRep(model, prompt, opt.nGen, prefillMs, vocab);
        std::fprintf(stderr,
                     "  warmup: %d tg tok in %.1f ms (%.1f tg tok/s); prefill %d tok "
                     "in %.1f ms (%.1f pp tok/s)\n",
                     r.tokens, r.ms, r.tokens / (r.ms / 1000.0), opt.nPrompts,
                     prefillMs, opt.nPrompts / (prefillMs / 1000.0));
    }

    // ---- Measured reps ----
    std::vector<double> tokPerSec;
    std::vector<double> decodeMs;
    std::vector<double> prefillTokPerSec;
    std::vector<double> prefillMsVec;
    for (int r = 0; r < opt.reps; ++r) {
        double prefillMs = 0.0;
        auto res = runOneRep(model, prompt, opt.nGen, prefillMs, vocab);
        prefillMsVec.push_back(prefillMs);
        double rate = res.tokens / (res.ms / 1000.0);
        double ppRate = opt.nPrompts / (prefillMs / 1000.0);
        tokPerSec.push_back(rate);
        decodeMs.push_back(res.ms);
        prefillTokPerSec.push_back(ppRate);
        std::fprintf(stderr,
                     "  rep %d: %d tok in %.1f ms (%.1f tg tok/s); prefill %d tok "
                     "in %.1f ms (%.1f pp tok/s)\n",
                     r + 1, res.tokens, res.ms, rate, opt.nPrompts, prefillMs,
                     ppRate);
    }

    // ---- Aggregate (mean ± stdev over reps, like llama-bench) ----
    double mean = 0.0;
    for (double v: tokPerSec) mean += v;
    mean /= static_cast<double>(opt.reps);
    double var = 0.0;
    for (double v: tokPerSec) var += (v - mean) * (v - mean);
    var /= static_cast<double>(std::max(1, opt.reps - 1));
    double stdev = std::sqrt(var);

    double meanMs = 0.0;
    for (double v: decodeMs) meanMs += v;
    meanMs /= static_cast<double>(opt.reps);

    double meanPrefillMs = 0.0;
    for (double v: prefillMsVec) meanPrefillMs += v;
    meanPrefillMs /= static_cast<double>(opt.reps);

    double meanPp = 0.0;
    for (double v: prefillTokPerSec) meanPp += v;
    meanPp /= static_cast<double>(opt.reps);

    std::printf("tg%d | %d threads | %.2f +/- %.2f tok/s | %.1f ms/decode (mean) || "
                "pp%d | %.1f +/- %.1f tok/s | %.1f ms (mean)\n",
                opt.nGen, static_cast<int>(threads), mean, stdev, meanMs,
                opt.nPrompts, meanPp, 0.0, meanPrefillMs);

    return 0;
}