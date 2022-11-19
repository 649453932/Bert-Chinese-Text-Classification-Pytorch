#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "util/tokenization.h"
#include "util/common.h"
#include "model.h"


Model::Model(const std::string& model_path, const std::string& vocab_path):env_(ORT_LOGGING_LEVEL_WARNING, "test") {
    tokenizer_ = new FullTokenizer(vocab_path);

    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options; //= {
//          0,
//          //OrtCudnnConvAlgoSearch::EXHAUSTIVE,
//          OrtCudnnConvAlgoSearchExhaustive,
//          std::numeric_limits<size_t>::max(),
//          0,
//          true
//      };

    session_options.AppendExecutionProvider_CUDA(cuda_options);
    ses_ = new Ort::Session(env_, model_path.c_str(), session_options);
}

std::vector<std::vector<int64_t>> Model::build_input(const std::string& text) {
    auto tokens = tokenizer_->tokenize(text);
    auto token_ids = tokenizer_->convertTokensToIds(tokens);

    std::vector<std::vector<int64_t>> res;

    std::vector<int64_t> input(32);
    std::vector<int64_t> mask(32);
    input[0] = 101;
    mask[0] = 1;
    for (int i = 0; i < token_ids.size() && i < 31; ++i) {
        input[i+1] = token_ids[i];
        mask[i+1] = token_ids[i] > 0;
    }
    res.push_back(std::move(input));
    res.push_back(std::move(mask));
    return res;
}

std::string Model::predict(const std::string& text) {
    int idx = infer(text);
    return key[idx];
}

int Model::infer(const std::string& text) {
    auto& session = *ses_;

    auto res = build_input(text);
    std::vector<int64_t> shape = {1, 32};

    auto& input_tensor_values = res[0];
    auto& mask_tensor_values = res[1];

    //std::cout<<input_tensor_values;
    //std::cout<<mask_tensor_values;
    //size_t input_tensor_size = 32;
    //for (auto i : input_tensor_values) {
    //    std::cout << i << "\t" ;
    //}
    //std::cout<<std::endl;
    //for (auto i : mask_tensor_values) {
    //    std::cout << i << "\t" ;
    //}
    //std::cout<<std::endl;
        
    // create input tensor object from data values ！！！！！！！！！！
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_tensor_values.data(),
                                                            input_tensor_values.size(), shape.data(), 2);

    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, mask_tensor_values.data(),
                                                            mask_tensor_values.size(), shape.data(), 2);

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(mask_tensor));

    const static std::vector<const char*> input_node_names = {"ids", "mask"};
    const static std::vector<const char*> output_node_names = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                    ort_inputs.size(), output_node_names.data(), 1);

    if (output_tensors.size() != output_node_names.size()) {
        return -1;
    }
    std::cout<<output_tensors.size()<<std::endl;
    //float* output = output_tensors[0].GetTensorMutableData<float>();
    const float* output = output_tensors[0].GetTensorData<float>();

    //for (int i = 0; i < 10; i++) {
    //    std::cout<<floatarr[i]<<std::endl;
    //}
    return argmax(output, output+10);
}

