#import sys
#sys.path.append('..')
import numpy
import torch
#import time
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


class Train:
    def __init__(self, model, optimizer, epochs, batch_size, print_interval):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_interval = print_interval
        self.AUC_1 = 0

    # 定义一个训练函数
    def train(self, train_x, train_y, val_x, val_y):
        # 将模型设置为训练模式
        self.model.train()
        # 创建一个训练数据集
        train_x_tensor = train_x.clone().detach().requires_grad_(True)
        train_y_tensor = torch.tensor(train_y)
        train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
        # 创建一个训练数据加载器，打乱数据，设置批量大小
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        # 循环训练轮数
        for epoch in range(self.epochs):
            # 初始化训练损失和训练准确率
            train_loss = 0.0
            train_acc = 0.0
            # 循环训练批次
            for batch_x, batch_y in train_loader:
                # 将数据和标签送入模型，得到输出
                output = self.model(batch_x)
                # 计算对数概率
                output = torch.softmax(output, dim=1)
                # 计算损失
                loss = self.model.loss_fn(output, batch_y)
                # 清空梯度
                self.optimizer.zero_grad()
                # 反向传播，计算梯度
                loss.backward()
                # 更新参数
                self.optimizer.step()
                # 累加训练损失
                train_loss += loss.item()
                # 计算预测标签，使用torch.argmax来取出最大值的索引，即为预测的类别
                pred_y = torch.argmax(output, dim=1)
                # 计算预测正确的个数，使用torch.eq来比较两个张量是否相等，然后使用torch.sum来求和
                correct = torch.sum(torch.eq(pred_y, batch_y))
                # 累加训练准确率
                train_acc += correct.item()
            # 计算平均训练损失和平均训练准确率
            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_dataset)
            # 每隔一定的轮数打印训练信息
            if (epoch + 1) % self.print_interval == 0:
                print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            #if (epoch+1) % 5 ==0:
                score = []
                label = []
                val_x_tensor = val_x.clone().detach().float()
                val_y_tensor = torch.tensor(val_y)
                test_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)
                for batch_x, batch_y in test_loader:
                    output = torch.softmax(self.model(batch_x), dim=1)
                    for i in output:
                        score.append(i[1])
                    for j in batch_y.tolist():
                        label.append(j)

                AUC_2 = roc_auc_score(label, [s.detach().numpy() for s in score])
                print(AUC_2)
                #acc = AUC_2 - self.AUC_1
                #if epoch >= 69:
                    #if acc > 0.5:
                        #self.AUC_1 = AUC_2
                    #else:
                        #break


    # 定义一个测试函数，传入模型，测试数据，测试标签，批量大小等参数
    def test(self, test_x, test_y):
        # tensor_x = torch.tensor(test_x).float()
        tensor_x = test_x.clone().detach().float()
        tensor_y = torch.tensor(test_y)
        # 设置模型为评估模式
        self.model.eval()
        # 创建一个测试数据集
        test_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        # 创建一个测试数据加载器，设置批量大小
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)
        # 创建一个AUC指标对象
        # auc = torchmetrics.AUROC(task='binary')
        # 创建一个PR指标对象
        # pr = torchmetrics.PrecisionRecallCurve(task='binary')
        # 循环测试批次
        score = []
        label = []
        for batch_x, batch_y in test_loader:
            # 将数据和标签送入模型，得到输出
            output = self.model(batch_x)
            # 计算对数概率
            output = torch.softmax(output, dim=1)

            for i in output:
                score.append(i[1])

            for j in batch_y.tolist():
                label.append(j)

            # 计算AUC值
            # auc.update(output, batch_y)
            # 计算PR曲线
            # pr.update(pred_y, batch_y)
        # 输出AUC值和PR值
        AUC_score = roc_auc_score(label, [s.detach().numpy() for s in score])
        PR_score = average_precision_score(label, [s.detach().numpy() for s in score])

        return AUC_score, PR_score

        #print("AUC:", AUC_score)
        #print("PR:", PR_score)

        # print("AUC:", auc.compute())
        # print("PR:", pr.compute())


