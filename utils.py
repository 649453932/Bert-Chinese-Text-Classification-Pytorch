# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):  # 对可遍历对象用tqdm维护
                lin = line.strip()  # remove leading and tailing whitespace
                if not lin:
                    continue
                content, label = lin.split('\t')  #  运行到\t时，判断当前字符串长度，将当前字符串长度补到8的倍数
                token = config.tokenizer.tokenize(content)  
                #  来自BertTokenizer.from_pretrained(self.bert_path + '/bert-base-chinese-vocab.txt')
                #  tokenize的目标是把输入的文本流，切分成一个个子串，三种粒度：word/subword/char
                token = [CLS] + token  # sentence的开头是[CLS]
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)  # to ids

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]  # 截断
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, datas, batch_size, device):
        # datas和batch
        self.batch_size = batch_size
        self.batches = datas
        self.n_batches = len(datas) // batch_size
        self.residue = False
        if len(datas) % self.n_batches != 0:
            self.residue = True  # 分成batch后剩余不到一个batch的数据
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # 设置x,y 元素为长整型64-bit integer (python自带的类型是变长的，浪费内存)
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        # tensor格式, 
        return (x, seq_len, mask), y
    # 含有__next__()函数的对象都是一个迭代器
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))  # the difference between two datetime objects.
