#!/usr/bin/env python
# coding=utf-8
import onnxruntime as ort
import numpy as np
from pred import build_predict_text, key
import time

def get_ort_session(model_path):
    providers = ort.get_available_providers()
    # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    return [ort.InferenceSession(model_path, providers=[provider]) for provider in providers]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict(sess, text):
    data = build_predict_text(t)
    '''
    print(len(sess.get_inputs()))
    for i in sess.get_inputs():
        print(i.name)
    print(len(data))
    '''
    input = {
        sess.get_inputs()[0].name: to_numpy(data[0]),
        sess.get_inputs()[1].name: to_numpy(data[2]),
        #sess.get_inputs()[2].name: data[2],
            }
    outs = sess.run(None, input)
    num = np.argmax(outs)
    return key[num]

if __name__ == '__main__':
    ts = [
        "李稻葵:过去2年抗疫为每人增寿10天",
        "4个小学生离家出走30公里想去广州塔",
        "朱一龙戏路打通电影电视剧",
        "天问一号着陆火星一周年",
    ]
    sesses = get_ort_session("./model.onnx")
    for sess in sesses: 
        print("\n")
        a = time.time()
        for t in ts:
            res = predict(sess, t)
            print("%s is %s" % (t, res))
        b = time.time()
        provider = sess._providers[0]
        print("%s cost: %.4f" % (provider, (b - a)))

