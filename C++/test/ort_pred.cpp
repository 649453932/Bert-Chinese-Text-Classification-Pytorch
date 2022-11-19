#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

using namespace std;

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

template <typename T>
int argmax(const std::vector<T>& v) {
    if (v.empty()) {
        return -1;
    }
    return std::max_element(v.begin(), v.end()) - v.begin();
}
template <typename T>
int argmax(T a, T b) {
    return std::max_element(a, b) - a;
}


int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
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
    const char* model_path = "/home/guodong/github/Bert-Chinese-Text-Classification-Pytorch/model.onnx";

    std::vector<int> token_ids = {101, 3330, 4940, 5878,  131, 6814, 1343,  123, 2399, 2834, 4554,  711,
                                  3680,  782, 1872, 2195, 8108, 1921,    0,    0,    0,    0,    0,    0,
                                  0,    0,    0,    0,    0,    0,    0,    0};

    Ort::Session session(env, model_path, session_options);
    // print model input layer (node names, types, shape etc.)
    //Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::cout<< num_input_nodes <<std::endl;
    std::cout<< session.GetOutputCount() <<std::endl;

    std::vector<int64_t> input_node_dims = {1, 32};

    size_t input_tensor_size = 32;
    std::vector<int64_t> input_tensor_values(input_tensor_size);
    std::vector<int64_t> mask_tensor_values(input_tensor_size);
    for (unsigned int i = 0; i < input_tensor_size; i++) {
        input_tensor_values[i] = int64_t(token_ids[i]);
        mask_tensor_values[i] = int64_t(token_ids[i]) != 0;
    }
        
    // create input tensor object from data values ！！！！！！！！！！
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_tensor_values.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);

    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, mask_tensor_values.data(),
                                                            input_tensor_size, input_node_dims.data(), 2);

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    ort_inputs.push_back(std::move(mask_tensor));

    std::vector<const char*> input_node_names = {"ids", "mask"};
    std::vector<const char*> output_node_names = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                    ort_inputs.size(), output_node_names.data(), 1);

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();

    for (int i=0; i<10; i++)
    {
        std::cout<<floatarr[i]<<std::endl;
    }
    std::cout<< key[argmax(floatarr, floatarr+10)] << std::endl;

    return 0;
}
