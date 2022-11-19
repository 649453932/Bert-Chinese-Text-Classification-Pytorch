#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include "util/model.h"

using namespace std;

int main() {
    const char* vocab_path = "/home/guodong/bert_pretrain/vocab.txt";
    const char* model_path = "/home/guodong/github/Bert-Chinese-Text-Classification-Pytorch/model.onnx";

    Model model(model_path, vocab_path);

    //const char* text = "李稻葵:过去2年抗疫为每人增寿10天";
    //int idx = model.predict(text);
    
    std::string line;
    while (std::getline(std::cin, line)) {
        auto a = gettimeofday_us();
        std::string r = model.predict(line);
        auto b = gettimeofday_us();
        std::cout << line << " is " << r << " cost:" << (b-a) <<" us" <<std::endl;
    }

    return 0;
}
