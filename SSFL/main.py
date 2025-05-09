import numpy as np
import torch
import pandas as pd
#from sklearn import preprocessing

from divide_dataset import Divide
from one_hot_neighbor import Feature
from FCNN_network import FCNN
from Train_data import Train

from CN_method import CN
from AA_method import AA
from AA_MAX_method import AA_MAX
from AA_MUL_method import AA_MUL
from JS_method import JS
from JS_MAX_method import JS_MAX
from JS_MUL_method import JS_MUL
#from PA_method import PA
from RA_method import RA
#from TRPR_method import TRPR
#from TRPRW_method import TRPRW
from CNDP_method import CNDP
from node2vec_method import node2vec

#构造网络和训练比例
rate = 0.9
#path = "E:\\datasets\\all\\inf-openflights_2905_15645.txt"
path = "./datasets/delaunay_n11_2048_6127.txt"

# 设置一些超参数，如训练轮数（epochs），批量大小（batch_size），打印间隔（print_interval）
epochs = 2
batch_size = 16
print_interval = 1
train_number = 1

#用于存放每一轮的分数(AUC)
Sub_graph_AUCscores = []
CN_AUCscores = []
AA_AUCscores = []
AA_MAX_AUCscores = []
AA_MUL_AUCscores = []
JS_AUCscores = []
JS_MAX_AUCscores = []
JS_MUL_AUCscores = []
RA_AUCscores = []
#TRPR_AUCscores = []
#TRPRW_AUCscores = []
CNDP_AUCscores = []
node2vec_AUCscores = []

#用于存放每一轮的分数(PR)
Sub_graph_PRscores = []
CN_PRscores = []
AA_PRscores = []
AA_MAX_PRscores = []
AA_MUL_PRscores = []
JS_PRscores = []
JS_MAX_PRscores = []
JS_MUL_PRscores = []
RA_PRscores = []
#TRPR_PRscores = []
#TRPRW_PRscores = []
CNDP_PRscores = []
node2vec_PRscores = []

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

    # CN baseline
    CN_model = CN(test_x, test_y, h)
    CN_AUC_score, CN_PR_score = CN_model.train()
    CN_AUCscores.append(CN_AUC_score)
    CN_PRscores.append(CN_PR_score)

    # AA baseline
    AA_model = AA(test_x, test_y, h)
    AA_AUC_score, AA_PR_score = AA_model.train()
    AA_AUCscores.append(AA_AUC_score)
    AA_PRscores.append(AA_PR_score)

    # AA_MAX baseline
    AA_MAX_model = AA_MAX(test_x, test_y, h)
    AA_MAX_AUC_score, AA_MAX_PR_score = AA_MAX_model.train()
    AA_MAX_AUCscores.append(AA_MAX_AUC_score)
    AA_MAX_PRscores.append(AA_MAX_PR_score)

    # AA_MUL baseline
    AA_MUL_model = AA_MUL(test_x, test_y, h)
    AA_MUL_AUC_score, AA_MUL_PR_score = AA_MUL_model.train()
    AA_MUL_AUCscores.append(AA_MUL_AUC_score)
    AA_MUL_PRscores.append(AA_MUL_PR_score)

    # JS baseline
    JS_model = JS(test_x, test_y, h)
    JS_AUC_score, JS_PR_score = JS_model.train()
    JS_AUCscores.append(JS_AUC_score)
    JS_PRscores.append(JS_PR_score)

    # JS_MAX baseline
    JS_MAX_model = JS_MAX(test_x, test_y, h)
    JS_MAX_AUC_score, JS_MAX_PR_score = JS_MAX_model.train()
    JS_MAX_AUCscores.append(JS_MAX_AUC_score)
    JS_MAX_PRscores.append(JS_MAX_PR_score)

    # JS_MUL baseline
    JS_MUL_model = JS_MUL(test_x, test_y, h)
    JS_MUL_AUC_score, JS_MUL_PR_score = JS_MUL_model.train()
    JS_MUL_AUCscores.append(JS_MUL_AUC_score)
    JS_MUL_PRscores.append(JS_MUL_PR_score)

    # PA baseline
    #PA_model = PA(test_x, test_y, h)
    #PA_AUC_score, PA_PR_score = PA_model.train()
    #PA_AUCscores.append(PA_AUC_score)
    #PA_PRscores.append(PA_PR_score)

    # RA baseline
    RA_model = RA(test_x, test_y, h)
    RA_AUC_score, RA_PR_score = RA_model.train()
    RA_AUCscores.append(RA_AUC_score)
    RA_PRscores.append(RA_PR_score)

    # TRPR baseline
    #TRPR_model = TRPR(test_x, test_y, h)
    #TRPR_AUC_score, TRPR_PR_score = TRPR_model.train()
    #TRPR_AUCscores.append(TRPR_AUC_score)
    #TRPR_PRscores.append(TRPR_PR_score)

    # TRPRW baseline
    #TRPRW_model = TRPRW(test_x, test_y, h)
    #TRPRW_AUC_score, TRPRW_PR_score = TRPRW_model.train()
    #TRPRW_AUCscores.append(TRPRW_AUC_score)
    #TRPRW_PRscores.append(TRPRW_PR_score)

    # CNDP baseline
    CNDP_model = CNDP(test_x, test_y, h)
    CNDP_AUC_score, CNDP_PR_score = CNDP_model.train()
    CNDP_AUCscores.append(CNDP_AUC_score)
    CNDP_PRscores.append(CNDP_PR_score)

    # node2vec baseline
    node2vec_model = node2vec(test_x, test_y, h)
    node2vec_AUC_score, node2vec_PR_score = node2vec_model.train()
    node2vec_AUCscores.append(node2vec_AUC_score)
    node2vec_PRscores.append(node2vec_PR_score)

AUC_scores = []
average_Sub_AUCscore = [sum(Sub_graph_AUCscores) / train_number]
AUC_scores.append(average_Sub_AUCscore)

average_CN_AUCscore = [sum(CN_AUCscores) / train_number]
AUC_scores.append(average_CN_AUCscore)

average_AA_AUCscore = [sum(AA_AUCscores) / train_number]
AUC_scores.append(average_AA_AUCscore)

average_AA_MAX_AUCscore = [sum(AA_MAX_AUCscores) / train_number]
AUC_scores.append(average_AA_MAX_AUCscore)

average_AA_MUL_AUCscore = [sum(AA_MUL_AUCscores) / train_number]
AUC_scores.append(average_AA_MUL_AUCscore)

average_JS_AUCscore = [sum(JS_AUCscores) / train_number]
AUC_scores.append(average_JS_AUCscore)

average_JS_MAX_AUCscore = [sum(JS_MAX_AUCscores) / train_number]
AUC_scores.append(average_JS_MAX_AUCscore)

average_JS_MUL_AUCscore = [sum(JS_MUL_AUCscores) / train_number]
AUC_scores.append(average_JS_MUL_AUCscore)

average_RA_AUCscore = [sum(RA_AUCscores) / train_number]
AUC_scores.append(average_RA_AUCscore)

#average_TRPR_AUCscore = [sum(TRPR_AUCscores) / train_number]
#AUC_scores.append(average_TRPR_AUCscore)

#average_TRPRW_AUCscore = [sum(TRPRW_AUCscores) / train_number]
#AUC_scores.append(average_TRPRW_AUCscore)

average_CNDP_AUCscore = [sum(CNDP_AUCscores) / train_number]
AUC_scores.append(average_CNDP_AUCscore)

average_node2vec_AUCscore = [sum(node2vec_AUCscores) / train_number]
AUC_scores.append(average_node2vec_AUCscore)

df_AUC = pd.DataFrame(AUC_scores)
df_AUC.to_excel('AUC_data.xlsx', index=False)


PR_scores = []
average_Sub_PRscore = [sum(Sub_graph_PRscores) / train_number]
PR_scores.append(average_Sub_PRscore)

average_CN_PRscore = [sum(CN_PRscores) / train_number]
PR_scores.append(average_CN_PRscore)

average_AA_PRscore = [sum(AA_PRscores) / train_number]
PR_scores.append(average_AA_PRscore)

average_AA_MAX_PRscore = [sum(AA_MAX_PRscores) / train_number]
PR_scores.append(average_AA_MAX_PRscore)

average_AA_MUL_PRscore = [sum(AA_MUL_PRscores) / train_number]
PR_scores.append(average_AA_MUL_PRscore)

average_JS_PRscore = [sum(JS_PRscores) / train_number]
PR_scores.append(average_JS_PRscore)

average_JS_MAX_PRscore = [sum(JS_MAX_PRscores) / train_number]
PR_scores.append(average_JS_MAX_PRscore)

average_JS_MUL_PRscore = [sum(JS_MUL_PRscores) / train_number]
PR_scores.append(average_JS_MUL_PRscore)

average_RA_PRscore = [sum(RA_PRscores) / train_number]
PR_scores.append(average_RA_PRscore)

#average_TRPR_PRscore = [sum(TRPR_PRscores) / train_number]
#PR_scores.append(average_TRPR_PRscore)

#average_TRPRW_PRscore = [sum(TRPRW_PRscores) / train_number]
#PR_scores.append(average_TRPRW_PRscore)

average_CNDP_PRscore = [sum(CNDP_PRscores) / train_number]
PR_scores.append(average_CNDP_PRscore)

average_node2vec_PRscore = [sum(node2vec_PRscores) / train_number]
PR_scores.append(average_node2vec_PRscore)

df_PR = pd.DataFrame(PR_scores)
df_PR.to_excel('PR_data.xlsx', index=False)

