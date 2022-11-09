#!/usr/bin/env python
# coding=utf-8
import numpy as np
import onnxruntime as ort
from importlib import import_module

key = {
    0: 'finance',
    1: 'realty',
    2: 'stocks',
    3: 'education',
    4: 'science',
    5: 'society',
    6: 'politics',
    7: 'sports',
    8: 'game',
    9: 'entertainment'
}

model_name = 'bert'
x = import_module('models.' + model_name)
config = x.Config('THUCNews')

def build_predict_text(text):
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = config.pad_size
    # 下面进行padding，用0补足位数
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
    return [token_ids], [mask]

def predict(sess, text):
    ids, mask = build_predict_text(t)

    input = {
        'ids': np.array(ids),
        'mask': np.array(mask),
    }
    outs = sess.run(None, input)
    num = np.argmax(outs)
    return key[num]

if __name__ == '__main__':
    sess =  ort.InferenceSession('./model.onnx', providers=['CUDAExecutionProvider'])

    ts = [
        '李稻葵:过去2年抗疫为每人增寿10天',
        '4个小学生离家出走30公里想去广州塔',
        '朱一龙戏路打通电影电视剧',
        '天问一号着陆火星一周年',
    ]
    for t in ts:
        res = predict(sess, t)
        print('%s is %s' % (t, res))

