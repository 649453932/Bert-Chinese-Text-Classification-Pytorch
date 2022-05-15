#!/usr/bin/env python
# coding=utf-8
import onnxruntime
import torch
import numpy as np
from pred import build_predict_text, key
import time

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#sess = onnxruntime.InferenceSession("./model.onnx", providers=["CUDAExecutionProvider"])    # use gpu
sess = onnxruntime.InferenceSession("./model.onnx")    # use cpu

def predict(text):
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
    t = "李稻葵:过去2年抗疫为每人增寿10天"
    t = "4个小学生离家出走30公里想去广州塔"
    t = "朱一龙戏路打通电影电视剧"
    t = "天问一号着陆火星一周年"

    a = time.time()
    print(predict(t))
    b = time.time()
    print(b - a)
