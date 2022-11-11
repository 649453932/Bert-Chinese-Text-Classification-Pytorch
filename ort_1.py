#!/usr/bin/env python
# coding=utf-8
import onnxruntime as ort
import numpy as np
from pred import build_predict_text, key

def get_ort_session(model_path, providers = None):
    if providers is None:
        providers = ort.get_available_providers()
    return [ort.InferenceSession(model_path, providers=[provider]) for provider in providers]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict(sess, text):
    ids, seq_len, mask = build_predict_text(t)

    input = {
        sess.get_inputs()[0].name: to_numpy(ids),
        sess.get_inputs()[1].name: to_numpy(mask),
    }
    outs = sess.run(None, input)
    num = np.argmax(outs)
    return key[num]

if __name__ == '__main__':
    sess =  ort.InferenceSession("./model.onnx", providers=['CUDAExecutionProvider'])

    ts = [
        "李稻葵:过去2年抗疫为每人增寿10天",
        "4个小学生离家出走30公里想去广州塔",
        "朱一龙戏路打通电影电视剧",
        "天问一号着陆火星一周年",
    ]
    for t in ts:
        res = predict(sess, t)
        print("%s is %s" % (t, res))

