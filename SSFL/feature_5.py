import numpy as np
import torch
import pandas as pd

from divide_dataset import Divide
from one_hot_neighbor import Feature
from FCNN_network import FCNN
from Train_data import Train


rate = 0.9
path = "./datasets/Chebyshev1_261_1542.txt"

# 设置一些超参数，如训练轮数（epochs），批量大小（batch_size），打印间隔（print_interval）
epochs = 2
batch_size = 16
print_interval = 1
train_number = 30

#用于存放每一轮的分数(AUC)
Sub_graph_AUCscores = []

#用于存放每一轮的分数(PR)
Sub_graph_PRscores = []

for i in range(train_number):
    #构造训练集、验证集、测试集
    divide_model = Divide(rate, path)
    train_x, train_y, val_x, val_y, test_x, test_y, h = divide_model.divide_data()

    # 构造输入特征
    input_feature_model = Feature(h)
    train_data_vector = input_feature_model.construct_vector(train_x)  # 训练集的输入特征向量
    val_data_vector = input_feature_model.construct_vector(val_x)  # 验证集的输入特征向量
    test_data_vector = input_feature_model.construct_vector(test_x)  # 测试集的输入特征向量

    # 创建一个神经网络的实例
    model = FCNN()
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 开始训练模型
    data_train = Train(model, optimizer, epochs, batch_size, print_interval)
    data_train.train(torch.tensor(np.array(train_data_vector)).float(), train_y,
                     torch.tensor(np.array(val_data_vector)).float(), val_y)

    # 测试
    AUC_score, PR_score = data_train.test(torch.tensor(np.array(test_data_vector)).float(), test_y)
    Sub_graph_AUCscores.append(AUC_score)
    Sub_graph_PRscores.append(PR_score)



AUC_scores = []
average_Sub_AUCscore = [sum(Sub_graph_AUCscores) / train_number]
AUC_scores.append(average_Sub_AUCscore)

df_AUC = pd.DataFrame(AUC_scores)
df_AUC.to_excel('AUC_data.xlsx', index=False)


PR_scores = []
average_Sub_PRscore = [sum(Sub_graph_PRscores) / train_number]
PR_scores.append(average_Sub_PRscore)

df_PR = pd.DataFrame(PR_scores)
df_PR.to_excel('PR_data.xlsx', index=False)