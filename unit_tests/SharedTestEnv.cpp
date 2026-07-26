#include "SharedTestEnv.hpp"
#include "Model.hpp"
#include "ModelConfig.hpp"
#include <string>

// Definition of static members
std::string SharedTestEnv::modelPath;
tinycoder::Model *SharedTestEnv::model = nullptr;
bool SharedTestEnv::modelLoaded = false;
tinycoder::ModelConfig SharedTestEnv::config;
