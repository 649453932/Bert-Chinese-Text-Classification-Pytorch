import torch
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
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))


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
    data = build_predict_text(t)
    #print(predict(t))
    input_names = ['input']
    torch.onnx.export(model, 
                      data,
                      'model.onnx',
                      export_params = True,
                      opset_version=10,
                      input_names = input_names,   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})
    '''
    input_names = ['input_ids', 'input_masks', 'token_type_ids']
    torch.onnx.export(model, 
                      data,
                      'model.onnx',
                      export_params = True,
                      opset_version=10,
                      input_names = input_names,   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input_ids' : {0 : 'batch_size'},    # variable lenght axes
                                    'input_masks' : {0 : 'batch_size'},
                                    'token_type_ids' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
    '''
