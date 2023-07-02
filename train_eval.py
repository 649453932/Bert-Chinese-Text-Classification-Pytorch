# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam


# 初始化神经网络模型的参数
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    # model：模型；method：初始化方法；exclude：需要排除的参数名称；seed：随机种子
    for name, w in model.named_parameters():    # name：参数名称；w：参数值
        if exclude not in name:
            if len(w.size()) < 2:   # 跳过维度小于2的参数
                continue
            if 'weight' in name:    # 按给定方法初始化权重
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:    # 偏置项初始化为0
                nn.init.constant_(w, 0)
            else:
                pass

"""
'Kaiming'（又称为He初始化）和'Xavier'是两种常用的权重初始化方法，它们在神经网络训练中起到重要的作用。

1. Xavier初始化：
   Xavier初始化是一种权重初始化方法，旨在保持前向和反向传播的激活值在不同层之间的方差相对一致。Xavier初始化方法假设激活函数是线性的，并且每一层的输入和输出神经元数量相等（或大致相等）。
   Xavier初始化根据每个参数的输入和输出神经元的数量自适应地调整初始化权重的范围。对于具有 n 个输入神经元和 n 个输出神经元的参数，Xavier初始化从均值为 0、标准差为 sqrt(1/n) 的分布中随机采样。

2. Kaiming初始化：
   Kaiming初始化是一种权重初始化方法，特别适用于使用ReLU（Rectified Linear Unit）激活函数的神经网络。ReLU激活函数在实践中非常常见，并且能够有效处理梯度消失的问题。Kaiming初始化根据激活函数的性质自适应地调整初始化权重的范围。
   对于具有 n 个输入神经元的参数，Kaiming初始化从均值为 0、标准差为 sqrt(2/n) 的分布中随机采样。这种初始化方法可以更好地传播梯度，并有助于减少梯度消失和梯度爆炸的问题。

区别与联系：
- 区别：'Xavier'初始化是为线性激活函数设计的，假设输入和输出神经元的数量相等，而'Kaiming'初始化则适用于使用ReLU激活函数的情况。'Kaiming'初始化在计算标准差时使用了更高的系数，以适应ReLU的非线性特性。
- 联系：'Xavier'和'Kaiming'初始化方法都是为了避免梯度消失和梯度爆炸问题。它们通过适当的权重初始化范围，使得在前向传播和反向传播过程中梯度的方差保持在可接受的范围内。这有助于更稳定地训练神经网络，并提高网络的性能和收敛速度。

选择何种初始化方法要视具体的神经网络结构、激活函数和任务类型而定。在实践中，根据经验和实验结果，选择合适的初始化方法可以提升模型的表现和训练效果。"""

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()   # 将模型设置为训练模式，启用训练截断特定的行为，例如启用Dropout或Batch Normalization
    param_optimizer = list(model.named_parameters())    # 获取模型的所有参数及其名称
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']   #定义不需要进行权重衰减的参数名称列表

    # 将模型的参数分组，并为每个参数组指定不同的权重衰减（weight decay）值
    optimizer_grouped_parameters = [
        # 这部分代码用于创建一个参数组，其中包含不需要进行权重衰减的模型参数。具体来说，它通过遍历 param_optimizer 中的参数和名称对，对于不包含在 no_decay 列表中的参数，将其添加到该参数组中。
        # 在这个参数组中，设置了 'weight_decay' 参数为 0.01，即对这些参数应用权重衰减。
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        # 这部分代码用于创建另一个参数组，其中包含需要进行权重衰减的模型参数。类似地，它通过遍历 param_optimizer 中的参数和名称对，对于包含在 no_decay 列表中的参数，将其添加到该参数组中。
        # 在这个参数组中，设置了 'weight_decay' 参数为 0.0，即对这些参数不应用权重衰减。
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    #      一般来说，对于偏置参数（bias）或层归一化参数（layer normalization parameters），通常不需要应用权重衰减，而对于权重参数（weight），可以根据需要进行权重衰减。
    #     权重衰减（Weight Decay）是一种用于正则化神经网络模型的技术。它通过在损失函数中添加一个惩罚项来限制权重的大小，以减少过拟合现象。

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)   # 优化器
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')    # 初始化验证集上的最佳损失值为正无穷大
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升的标志
    model.train()
    for epoch in range(config.num_epochs):  # 进行多轮次迭代
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):   # 遍历训练数据迭代器，获取输入数据的标签
            outputs = model(trains) # 前向传播，计算模型的输出
            model.zero_grad()   # 清除模型参数的梯度
            loss = F.cross_entropy(outputs, labels) # 计算loss（交叉熵损失）
            loss.backward() # 反向传播，计算梯度
            optimizer.step()    # 更新模型参数，执行优化器的一步更新
            if total_batch % 100 == 0:  # 每经过一定数量的batch后进行一次训练效果的输出和验证集上的评估
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()    # 将标签数据移到cpu上
                predic = torch.max(outputs.data, 1)[1].cpu()    # 计算预测结果，选择输出中概率最大的类别
                train_acc = metrics.accuracy_score(true, predic)    # 计算训练集的准确率
                dev_acc, dev_loss = evaluate(config, model, dev_iter)   # 在验证集上评估模型的准确率和损失
                if dev_loss < dev_best_loss:    # 如果当前验证集损失比之前记录的最佳损失更小，则更新最佳损失和保存模型的状态字典
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)    # 保存模型的字典状态
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'    # 定义输出信息的格式
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))    #输出训练信息
                model.train()
            total_batch += 1    # 增加batch计数
            if total_batch - last_improve > config.require_improvement: # 如果验证集的损失在一定batch数量内没有下降，则提前结束训练
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)  #在测试集上评估模型的性能


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

"""
`metrics.accuracy_score(true, predic)`是用于计算分类模型的准确率（accuracy）的函数。

在这行代码中，`true`是真实的标签值，`predic`是模型预测的标签值。`metrics.accuracy_score()`会将真实标签和预测标签进行比较，并返回分类模型的准确率。准确率是分类模型中常用的性能指标之一，表示模型预测正确的样本数占总样本数的比例。

Scikit-learn（sklearn）的`metrics`模块提供了许多用于评估分类、回归和聚类模型性能的指标和函数。除了准确率，还包括精确率、召回率、F1分数等常见的分类指标，以及均方误差、R2分数等回归指标。这些指标可以帮助我们评估和比较不同模型的性能，从而选择最佳的模型或进行模型调优。
"""