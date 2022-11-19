#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "util/tokenization.h"
#include "util/common.h"

const static std::vector<std::string> key = {                                                                                                                                                                                                                     
    "finance",
    "realty",
    "stocks",
    "education",
    "science",
    "society",
    "politics",
    "sports",
    "game",
    "entertainment"
};


class Model {
public:
    Model(const std::string& model_path, const std::string& vocab_path);

    std::string predict(const std::string& text);
    int infer(const std::string& text);

protected:
    std::vector<std::vector<int64_t>> build_input(const std::string& text);
        
private:
    FullTokenizer* tokenizer_ = nullptr;
    Ort::Session* ses_ = nullptr;
    Ort::Env env_;
};
