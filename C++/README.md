# 依赖
[boost](https://www.boost.org/)，[utf8proc](https://github.com/guodongxiaren/utf8proc)
强烈不建议使用Github上面的boost项的Release（缺少submodule）

 g++ token.cpp -std=c++11 -I ~/local/ -I ~/local/include -L ~/local/lib/ -lutf8proc
 export LD_LIBRARY_PATH=~/local/lib:$LB_LIBRARY_PATH

 g++ ort_pred.cpp -I ~/local/onnxruntime/include --std=c++11 -L ~/local/onnxruntime/lib -lonnxruntime
