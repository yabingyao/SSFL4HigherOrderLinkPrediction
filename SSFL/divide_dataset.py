import random
import numpy as np
import networkx as nx
import copy

class Divide:
    def __init__(self, rate, path):
        self.rate = rate
        self.g = nx.read_edgelist(path)


    def random_triangles(self, all_triangles, train_ratio):
        test_ratio = 1 - train_ratio  # 计算选取多少比例的测试集
        test_length = int(round(test_ratio * len(all_triangles)))  # 计算测试集中需要删除三角形的数量
        test_length = test_length if test_length > 1 else 1  # 如果测试集长度小于等于1，则设为1
        random_triangles = []  # 存储随机三角形的列表
        test = []  # 初始化测试集列表
        while len(test) < test_length:
            item = random.sample(all_triangles, 1)[0]  # 从所有三角形中随机选择一个三角形
            remove_edges_list = self.remove_edges(test)  # 获取需要移除的边
            success, test = self.filter_triangles(item, test, remove_edges_list)  # 过滤三角形
            if success:
                random_triangles.append(item)
        return test


    def remove_edges(self, test):
        target_nodes = [i[0] for i in test]  # 获取待推荐节点列表
        seed_edges = [i[1] for i in test]  # 获取测试集中的种子边列表
        remove_edges = []  # 存储要移除的边列表
        for i in range(len(seed_edges)):  # 遍历种子边列表
            target_node = target_nodes[i][0]  # 获取待推荐节点
            edge_node_1, edge_node_2 = seed_edges[i]  # 获取种子边的两个端点
            remove_edge_a = [edge_node_1, target_node]
            remove_edge_b = [edge_node_2, target_node]
            remove_edges.extend([remove_edge_a, remove_edge_b])
        return list(np.unique(remove_edges, axis=0))  # 去重


    def filter_triangles(self, random_triangles, test, remove_edges_list):
        test_2 = [i[1] for i in test]  # 将测试集中的种子边存起来
        triangle_edges = [tuple(edge) for edge in [[random_triangles[0], random_triangles[1]],
                                                   [random_triangles[0], random_triangles[2]],
                                                   [random_triangles[1], random_triangles[2]]]]  # 构建三角形的边列表，并将边转换为元组形式
        remove_edges_list = [tuple(edge) for edge in remove_edges_list]  # 将要移除的边列表转换为元组形式
        edges_tt = list(set(test_2) & set(triangle_edges))  # 三角形边列表中与种子边相交的边
        ltt = len(edges_tt)
        if ltt <= 1:  # 判断与种子边相交的边的数量
            if ltt == 0:
                seed_remove_edge = list(set(triangle_edges) - set(remove_edges_list))  # 三角形中的边与要移除的边的差集
                if len(seed_remove_edge) >= 1:
                    item_edge = random.sample(seed_remove_edge, 1)[0]  # 随机选一个作为种子边
                    triangle_node = list(set(random_triangles) - set(item_edge))  # 三角形中的边与种子边的差集
                    triangle_yz = [triangle_node, item_edge]  # 构造测试集
                    test.append(triangle_yz)  # 更新测试集
                    return True, test
                else:
                    return False, test
            else:
                edges_tt = edges_tt[0]
                triangle_node = list(set(random_triangles) - set(edges_tt))  # 三角形中的边与种子边的差集
                triangle_yz = [triangle_node, edges_tt]  # 构造测试集
                test.append(triangle_yz)  # 更新测试集
            return True, test
        else:
            return False, test

    def divide_data(self):
        nodes_list = self.g.nodes()  # 获取节点列表
        edges_list = self.g.edges()  # 获取边列表
        all_cliques = nx.enumerate_all_cliques(self.g)  # 列出原网络中所有的团
        triad_cliques = [x for x in all_cliques if len(x) == 3]  # 过滤出原网络中长度为3的团，即三角形
        test_p = self.random_triangles(triad_cliques, self.rate)  # 获取测试集
        remove_edges_list = self.remove_edges(test_p)  # 获取要移除的边列表
        h = copy.deepcopy(self.g)
        h.remove_edges_from(remove_edges_list)  # 删除三角形中的楔形结构
        remain_edges_list = h.edges()  # 训练集中剩余的边列表

        train_x_n = []
        train_x_p = []

        for edge in edges_list:
            for node in nodes_list:
                if not h.has_edge(node, edge[0]) and not h.has_edge(node, edge[1]):
                    train_x_n.append([[node], edge])  # 过滤出训练集中的所有负样本

        for edge in remain_edges_list:
            for node in nodes_list:
                if h.has_edge(node, edge[0]) and h.has_edge(node, edge[1]):
                    train_x_p.append([[node], edge])  # 过滤出训练集中的所有正样本

        random.shuffle(train_x_n)
        train_x_n_1 = train_x_n[:len(train_x_p)]  # 选取和正样本相同数量的负样本
        self.train_x = train_x_p + train_x_n_1
        train_y_p = [1] * len(train_x_p)
        train_y_n = [0] * len(train_x_n_1)
        self.train_y = train_y_p + train_y_n

        train_x_n_2 = train_x_n[len(train_x_p):]  # 负样本中剩余的样本
        test_x_n = train_x_n_2[:len(test_p)]

        val_num = int(len(test_p) / 2)

        val_x_n = test_x_n[:val_num]
        val_x_p = test_p[:val_num]
        self.val_x = val_x_p + val_x_n
        val_y_n = [0] * len(val_x_n)
        val_y_p = [1] * len(val_x_p)
        self.val_y = val_y_p + val_y_n

        test_x_n = test_x_n[val_num:]
        test_x_p = test_p[val_num:]
        self.test_x = test_x_p + test_x_n
        test_y_p = [1] * len(test_x_p)
        test_y_n = [0] * len(test_x_n)
        self.test_y = test_y_p + test_y_n

        return self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y, h



