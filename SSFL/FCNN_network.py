import torch
import torch.nn as nn


# 创建神经网络模型
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        # 输入层有8个神经元
        self.input_layer = nn.Linear(8, 32)
        # 隐藏层有3层，神经元个数分别为32，32和16
        self.hidden_layer1 = nn.Linear(32, 32)
        #self.hidden_layer2 = nn.Linear(32, 32)
        self.hidden_layer3 = nn.Linear(32, 16)
        # 输出层有1个神经元
        self.output_layer = nn.Linear(16, 2)
        # 激活函数为Relu
        self.relu = nn.ReLU()
        # 损失函数为交叉熵损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # 前向传播
        x = self.input_layer(x)
        x = self.relu(x)
        
        x = self.hidden_layer1(x)
        x = self.relu(x)
        
        x = self.hidden_layer3(x)
        x = self.relu(x)
        
        x = self.output_layer(x)
        return x
