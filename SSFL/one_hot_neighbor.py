import numpy as np


class Feature:
    def __init__(self, g):
        self.g = g

    def construct_vector(self,data):

        self.vector = []

        for node_edge in data:
            node_neighbors = list(self.g.neighbors(node_edge[0][0]))  # 获取节点的邻居
            #edge_neighbors = list(set(self.g.neighbors(node_edge[1][0])) | set(self.g.neighbors(node_edge[1][1])))  # 获取边的邻居（并集）
            edge_trangle = list(set(self.g.neighbors(node_edge[1][0])) & set(self.g.neighbors(node_edge[1][1])))

            ##a,b,x分别代表seed-edge的两个端点和待推荐节点
            a = list(self.g.neighbors(node_edge[1][0]))
            b = list(self.g.neighbors(node_edge[1][1]))
            V_31 = list(set(node_neighbors) & set(a))
            V_32 = list(set(node_neighbors) & set(b))

            node_31 =len(V_31)
            node_32 = len(V_32)

            # v_1 = 0
            a1 = 0
            for i in range(len(node_neighbors)):
                ## u是我们的邻居节点
                u = node_neighbors[i]
                if self.g.has_edge(u, node_edge[1][0]) and self.g.has_edge(u, node_edge[1][1]):
                    # v_1 += 1
                    a1 += 1

            # v_2 = len(edge_trangle)
            a2 = len(edge_trangle)

            E_1 = 0
            E_2 = 0
            E_31 = 0
            E_32 = 0
            degrees = 0

            for i in range(len(node_neighbors)):  # 遍历列表中的所有节点对
                u = node_neighbors[i]
                for j in range(i + 1, len(node_neighbors)):
                    v = node_neighbors[j]
                    E_1 += self.g.number_of_edges(u, v)  # 累加它们之间的边的数量

            for i in range(len(edge_trangle)):
                u = edge_trangle[i]
                for j in range(i + 1, len(edge_trangle)):
                    v = edge_trangle[j]
                    E_2 += self.g.number_of_edges(u, v)

            for i in range(len(V_31)):
                u = V_31[i]
                for j in range(i + 1, len(V_31)):
                    v = V_31[j]
                    E_31 += self.g.number_of_edges(u, v)
                    
            for i in range(len(V_32)):
                u = V_32[i]
                for j in range(i + 1, len(V_32)):
                    v = V_32[j]
                    E_32 += self.g.number_of_edges(u, v)
                    