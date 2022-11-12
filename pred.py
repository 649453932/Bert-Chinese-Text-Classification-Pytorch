import torch
from importlib import import_module
import time

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
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))

def build_predict_text_raw(text):
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
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
            seq_len = pad_size
    return token_ids, seq_len, mask

def build_predict_text(text):
    token_ids, seq_len, mask = build_predict_text_raw(text)
    ids = torch.LongTensor([token_ids]).cuda()
    seq_len = torch.LongTensor([seq_len]).cuda()
    mask = torch.LongTensor([mask]).cuda()

    return ids, seq_len, mask

def predict(text):
    """
    单个文本预测
    :param text:
    :return:
    """
    data = build_predict_text(text)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
    return key[int(num)]


if __name__ == '__main__':
    t = "李稻葵:过去2年抗疫为每人增寿10天"
    t = "天问一号着陆火星一周年"
    a = time.time()
    print(predict(t))
    b = time.time()
    print(b-a)
