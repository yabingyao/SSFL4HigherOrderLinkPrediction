import shap
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from divide_dataset import Divide
from one_hot_neighbor import Feature
from keras.utils import to_categorical

model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

#构造网络和训练比例
rate = 0.9
path = "./datasets/delaunay_n11_2048_6127.txt"

#构造训练集、验证集、测试集
divide_model = Divide(rate, path)
train_x, train_y, val_x, val_y, test_x, test_y, h = divide_model.divide_data()

# 构造输入特征
input_feature_model = Feature(h)
train_data_vector = input_feature_model.construct_vector(train_x)  # 训练集的输入特征向量
val_data_vector = input_feature_model.construct_vector(val_x)  # 验证集的输入特征向量
test_data_vector = input_feature_model.construct_vector(test_x)  # 测试集的输入特征向量


train_data_vector = np.array(train_data_vector)
train_y = np.array(train_y)


# 将标签进行独热编码
train_y = to_categorical(train_y, num_classes=2)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(train_data_vector, train_y, epochs=2)

# 计算Deep SHAP值
explainer = shap.DeepExplainer(model, train_data_vector)
shap_values = explainer.shap_values(train_data_vector)

shap_values_clipped = np.clip(shap_values, -2, 2)

# 将Deep SHAP值转换为Explanation对象
expected_value = explainer.expected_value[0]
feature_names = ['a1','a2','b1','b2','b3','b4','c1','c2']
shap_values = [np.array(sv) for sv in shap_values_clipped]
shap_exp = shap.Explanation(shap_values_clipped[0], feature_names=feature_names, data=train_data_vector)

shap.plots.beeswarm(shap_exp)