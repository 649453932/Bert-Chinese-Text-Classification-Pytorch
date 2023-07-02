# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

# argparse是Python的一个标准库，用于处理命令行参数的解析和生成。它提供了一个易于使用的方式来定义命令行接口，并自动生成帮助信息。
parser = argparse.ArgumentParser(description='Chinese Text Classification') #定义parser对象，用于解析命令行参数
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE') #定义需要的命令行参数及其属性：在这段代码中，有一个参数'--model'，其类型为字符串('str')，并且是必须的(required=True)，同时还提供了帮助信息('help')
args = parser.parse_args()  #解析命令行参数并返回一个包含解析结果的命名空间namespace对象，其中参数'--model'的值可以通过'arg-model'访问。这样，可以在代码中使用'arg-model'来获取在命令行中指定的模型名称。
# 如，可在命令行输入：python run.py --model bert

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # 获取命令行中指定的模型名称（bert）
    # import_module()函数是动态导入模块的一种通用方式，它可以接受任何有效的模块名称作为参数，并不仅限于模块文件名
    x = import_module('models.' + model_name)   #根据构建的模块名称动态导入对应的模块，并将模块对象赋值给变量x
    config = x.Config(dataset)  #加载配置文件
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config) # 这里返回的数据是已经完成分词和token2id之后的数据，可以直接拿来导入预训练模型进行训练
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

