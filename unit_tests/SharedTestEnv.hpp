#pragma once

#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "Model.hpp"
#include "ModelConfig.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// SharedTestEnv: Google Test global environment that loads the model once
// for ALL test suites. Registered in main() via ::testing::AddGlobalTestEnvironment().
// ---------------------------------------------------------------------------

class SharedTestEnv : public ::testing::Environment {
public:
    static std::string modelPath;
    static tinycoder::Model *model;
    static bool modelLoaded;
    static tinycoder::ModelConfig config;

    void SetUp() override {
        ASSERT_FALSE(modelPath.empty())
                << "Model path must be provided via TINYCODER_MODEL_PATH env var or "
                   "--model-path argument";
        ASSERT_TRUE(fs::exists(modelPath)) << "Model file not found: " << modelPath;

        model = new tinycoder::Model();
        std::string loadError;
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = model->load(modelPath, &loadError);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto loadMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        if (!ok) {
            std::cerr << "Model load failed: " << loadError << std::endl;
            delete model;
            model = nullptr;
            modelLoaded = false;
            FAIL() << "Model load failed: " << loadError;
        } else {
            modelLoaded = true;
            config = model->config();
            std::cout << "Model loaded in " << loadMs << " ms" << std::endl;
            std::cout << "  Model: \"" << config.modelName << "\" ("
                      << config.architecture << ")" << std::endl;
            std::cout << "  Config: " << config.numLayers << " layers, "
                      << config.hiddenSize << " hidden, "
                      << config.numAttentionHeads << " heads, "
                      << config.numKVHeads << " KV heads, "
                      << config.vocabSize << " vocab, "
                      << config.intermediateSize << " intermediate, "
                      << config.headDim << " headDim, "
                      << "ropeTheta=" << config.ropeTheta << std::endl;
            std::cout << "  Tokenizer: " << model->tokenizer().vocabSize()
                      << " tokens" << std::endl;
        }
    }

    void TearDown() override {
        delete model;
        model = nullptr;
        modelLoaded = false;
    }
};