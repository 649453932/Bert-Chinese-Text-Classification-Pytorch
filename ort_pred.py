#!/usr/bin/env python
# coding=utf-8
import onnxruntime as ort
import numpy as np
from pred import build_predict_text, key
from pred import build_predict_text1
import time

def get_ort_session(model_path, providers = None):
    # providers : ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    if providers is None:
        providers = ort.get_available_providers()
    return [ort.InferenceSession(model_path, providers=[provider]) for provider in providers]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict(sess, text):
    ids, seq_len, mask = build_predict_text(t)
    print(type(ids))

    input = {
        sess.get_inputs()[0].name: to_numpy(ids),
        sess.get_inputs()[1].name: to_numpy(mask),
    }
    outs = sess.run(None, input)
    num = np.argmax(outs)
    return key[num]

def predict1(sess, text):
    ids, seq_len, mask = build_predict_text1(t)

    input = {
        sess.get_inputs()[0].name: np.array(ids),
        sess.get_inputs()[1].name: np.array(mask),
    }
    outs = sess.run(None, input)
    num = np.argmax(outs)
    return key[num]

def predict2(sess, text):
    ids, seq_len, mask = build_predict_text1(t)

    input = {
        'ids': np.array(ids),
        'mask': np.array(mask),
    }
    outs = sess.run(None, input)
    num = np.argmax(outs)
    return key[num]

if __name__ == '__main__':
    #sesses = get_ort_session("./model.onnx", ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    sesses = get_ort_session("./model.onnx", ['CUDAExecutionProvider'])

    '''
    sess = sesses[0]
    print(len(sess.get_inputs()))
    for i in sess.get_inputs():
        print(i.name)
    exit("")
    '''
    ts = [
        "李稻葵:过去2年抗疫为每人增寿10天",
        "4个小学生离家出走30公里想去广州塔",
        "朱一龙戏路打通电影电视剧",
        "天问一号着陆火星一周年",
    ]
    for sess in sesses: 
        print("\n")
        a = time.time()
        for t in ts:
            res = predict(sess, t)
            print("%s is %s" % (t, res))
        b = time.time()
        provider = sess._providers[0]
        print("%s cost: %.4f" % (provider, (b - a)))

    for sess in sesses: 
        print("\n")
        a = time.time()
        for t in ts:
            res = predict1(sess, t)
            print("%s is %s" % (t, res))
        b = time.time()
        provider = sess._providers[0]
        print("%s cost: %.4f" % (provider, (b - a)))

    for sess in sesses: 
        print("\n")
        a = time.time()
        for t in ts:
            res = predict2(sess, t)
            print("%s is %s" % (t, res))
        b = time.time()
        provider = sess._providers[0]
        print("%s cost: %.4f" % (provider, (b - a)))
