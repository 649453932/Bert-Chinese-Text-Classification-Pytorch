# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
# args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = 'bert'  # bert
    x = import_module('models.' + model_name)  # 等价于import models.bert
    config = x.Config(dataset)
    # 固定随机数种子，方便复现结果
    np.random.seed(1)  
    torch.manual_seed(1)  # 
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 解决因cuda, gpu并行计算可能造成卷积操作的不确定性
    """For all purposes, training is stochastic. We use stochastic gradient descent, shuffle data, 
        use random initialization and dropout – and more importantly, 
        training data is itself but a random sample of data. From that standpoint, 
        the fact that computers can only generate pseudo-random numbers with a seed is an artifact. 
        When you train, your loss is a value that also comes with a confidence interval due to this stochastic nature. 
    """

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    # train_data每一行形式, [token_ids, label, seq_len, mask]
    # 实现访问和遍历数据的迭代器工具类
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
