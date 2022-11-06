import torch
from importlib import import_module

model_name = 'bert'
x = import_module('models.' + model_name)
config = x.Config('THUCNews')
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu')) # CPU?

def build_predict_text(text):
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = config.pad_size
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    ids = torch.LongTensor([token_ids]).cuda()
    seq_len = torch.LongTensor([seq_len]).cuda()
    mask = torch.LongTensor([mask]).cuda()
    return ids, seq_len, mask # 返回的是tuple类型，return [ids, seq_len, mask] 返回的是list类型，这里如果返回list，下面就不用再转型了

def build_input_data0():
    t = "李稻葵:过去2年抗疫为每人增寿10天"
    a = build_predict_text(t)
    return list(a) # 转型成list，不能[a] ，这是list(tuple())类型！

def build_input_data1():
    pad_size = config.pad_size
    ids = torch.LongTensor([[0]*pad_size]).cuda()
    seq_len = torch.LongTensor([0]).cuda()
    mask = torch.LongTensor([[0]*pad_size]).cuda()
    return [ids, seq_len, mask]

def build_input_data2():
    pad_size = config.pad_size
    mask = torch.LongTensor([[0]*pad_size]).cuda()
    ids = torch.randint(1, 10, (1, pad_size)).cuda()
    seq_len = torch.randint(1, 10, (1,)).cuda() # , 不能少
    mask = torch.randint(1, 10, (1, pad_size)).cuda()
    return [ids, seq_len, mask]


if __name__ == '__main__':
    data = build_input_data0()

    input_names = ['ids','seq_len', 'mask']
    torch.onnx.export(model, 
                      data,
                      'model.onnx',
                      export_params = True,
                      opset_version=11,
                      input_names = input_names,   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'ids' : {0 : 'batch_size'},    # variable lenght axes
                                    'seq_len' : {0 : 'batch_size'},
                                    'mask' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
