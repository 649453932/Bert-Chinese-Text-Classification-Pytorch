#!/usr/bin/env python
# coding=utf-8
import onnxruntime as ort
import numpy as np
from pred import key, config
import pred
import time

def to_numpy(tensor):
    """
    Tensor转numpy数组
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_ort_session(model_path, provider_prefix):
    '''
    获取指定provider的Session
    provider_prefix : ['Tensorrt', 'CUDA', 'CPU']
    '''
    provider = provider_prefix + 'ExecutionProvider'
    return ort.InferenceSession(model_path, providers=[provider])

def get_all_ort_session(model_path):
    # providers : ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    providers = ort.get_available_providers()
    return [ort.InferenceSession(model_path, providers=[provider]) for provider in providers]

def onnx_predict0(sess, text):
    """
    输入采用torch.Tensor转numpy数组
    """
    ids, seq_len, mask = pred.build_predict_text_raw(text)

    input = {
        sess.get_inputs()[0].name: to_numpy([ids]),
        sess.get_inputs()[1].name: to_numpy([mask]),
    }
    outs = sess.run(None, input)
    num = np.argmax(outs)
    return key[num]

def onnx_predict(sess, text):
    ids, seq_len, mask = pred.build_predict_text_raw(text)

    input = {
        sess.get_inputs()[0].name: np.array([ids]),
        sess.get_inputs()[1].name: np.array([mask]),
    }
    """
    等价于：
    input = {
        'ids': np.array(ids),
        'mask': np.array(mask),
    }
    """
    outs = sess.run(None, input)
    num = np.argmax(outs)
    return key[num]

def load_title(fname):
    ts = []
    with open(fname)  as f:
        for line in f.readlines():
            ts.append(line.strip())
    return ts

def batch_predict(ts, predict_fun, name):
    print('')
    a = time.time()
    for t in ts:
        res = predict_fun(t)
        print('%s is %s' % (t, res))
    b = time.time()
    print('%s cost: %.4f' % (name, (b - a)))

if __name__ == '__main__':
    model_path = './model.onnx'
    #sesses = get_all_ort_session('./model.onnx')
    cuda_ses = get_ort_session(model_path, 'CUDA')

    ts = [
        '兰州野生动物园观光车侧翻事故新进展：2人经抢救无效死亡',
        '4个小学生离家出走30公里想去广州塔',
        '朱一龙戏路打通电影电视剧',
        '天问一号着陆火星一周年',

    ]
    ts = load_title('./news_title.txt')
    #batch_predict(ts, lambda t: onnx_predict0(cuda_ses, t), 'ONNX_CUDA_tensor_to_numpy')
    batch_predict(ts, lambda t: onnx_predict(cuda_ses, t), 'ONNX_CUDA')
    batch_predict(ts, lambda t: pred.predict(t), 'Pytorch_CUDA')

