#!/usr/bin/env python
# coding=utf-8
import numpy as np
import onnxruntime as ort
import pred

def predict(sess, text):
    ids, seq_len, mask = pred.build_predict_text_raw(text)

    input = {
        'ids': np.array([ids]),
        'mask': np.array([mask]),
    }
    outs = sess.run(None, input)
    num = np.argmax(outs)
    return pred.key[num]

if __name__ == '__main__':
    sess =  ort.InferenceSession('./model.onnx', providers=['CUDAExecutionProvider'])

    t = '天问一号着陆火星一周年'
    res = predict(sess, t)
    print('%s is %s' % (t, res))

