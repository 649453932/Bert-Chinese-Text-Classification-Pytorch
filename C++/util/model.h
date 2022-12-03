#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "util/tokenization.h"
#include "util/common.h"

const static std::vector<std::string> kNames = {            
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
    ~Model() {delete tokenizer_; delete ses_;}

    // 执行文本预测，返回预测的分类ID
    int infer(const std::string& text, float* score=nullptr);

    // 执行文本预测，返回预测的分类名称
    std::string predict(const std::string& text, float* score=nullptr);

protected:
    // 将文本向量化，返回ids和mask两个向量
    std::vector<std::vector<int64_t>> build_input(const std::string& text);
        
private:
    FullTokenizer* tokenizer_ = nullptr;
    Ort::Session* ses_ = nullptr;
    Ort::Env env_;
};
