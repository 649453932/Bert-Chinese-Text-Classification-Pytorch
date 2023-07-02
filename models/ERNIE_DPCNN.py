# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        # self.bert_path = './bert_pretrain'
        self.bert_path = 'nghuyong/ernie-3.0-base-zh'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_filters = 250                                          # 卷积核数量(channels数)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out = self.bert(context, attention_mask=mask).last_hidden_state # [batch_size, sequence_length, hidden_size]
        x = encoder_out.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size]
        x = self.conv_region(x)  # [batch_size, 250(out_channels), seq_len-3+1, 1(hidden_size-hidden_size+1)]

        # 填充——激活——卷积，执行2轮
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)  #while循环的最终输出形状：[batch_size, 250, 1, 1]
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)  # 全连接层
        return x

    def _block(self, x):    #输入形状：[batch_size, 250, seq_len-3+1, 1]
        x = self.padding2(x)    # 底部填充，[batch_size, 250, seq_len-3+1+1, 1]
        px = self.max_pool(x)   # 最大池化，[batch_size, 250, (seq_len-3+1+1)//2, 1]
        x = self.padding1(px)   # 顶部和底部填充，[batch_size, 250, (seq_len-3+1+1)//2+2, 1]
        x = F.relu(x)           # ReLU激活函数，形状不变[batch_size, 250, (seq_len-3+1+1)//2+2, 1]
        x = self.conv(x)        # 卷积，[batch_size, 250, (seq_len-3+1+1)//2, 1]
        x = self.padding1(x)    # 顶部和底部填充，[batch_size, 250, (seq_len-3+1+1)//2+2, 1]
        x = F.relu(x)           # ReLU激活函数，形状不变[batch_size, 250, (seq_len-3+1+1)//2+2, 1]
        x = self.conv(x)        # 卷积，[batch_size, 250, (seq_len-3+1+1)//2, 1]
        x = x + px              # 残差连接，将上一步的输出张量与最大池化操作的结果相加，形状不变[batch_size, 250, (seq_len-3+1+1)//2, 1]
        return x    # 输出形状[batch_size, 250, (seq_len-3+1+1)//2, 1]