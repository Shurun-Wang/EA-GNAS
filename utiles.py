import os
import random
import numpy as np
import torch
import networkx as nx
from sklearn.model_selection import train_test_split
from core.train_val_test import *
from scipy.sparse.csgraph import minimum_spanning_tree


def seed_everything(seed=6767):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.mps.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_files(args):
    if not os.path.exists('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost)):
        os.makedirs('checkpoints/'+args.ioa +'/layer'+str(args.num_gnn_layers)+'/mct'+str(args.myCost))


def create_edge_index_from_adjacency_matrix(adj_matrix):
    """Creates edge index from given adjacency matrix."""
    edge_indices = np.argwhere(adj_matrix == 1)
    return torch.tensor(edge_indices, dtype=torch.long).t()

# 定义映射函数
def map_to_triangle(batch_data):
    # 获取批次大小和特征数量
    batch_size, num_features = batch_data.shape
    # 创建一个数组来保存转换后的数据
    mapped_data = np.zeros((batch_size, 140, 140))
    # 为每个样本做映射
    for i in range(batch_size):
        # 获取特征向量
        features = batch_data[i]
        # 创建一个140x140的零矩阵
        matrix = np.zeros((140, 140))
        # 使用特征向量填充矩阵的下三角部分（不包括对角线）
        matrix[np.tril_indices(140, -1)] = features
        lower_triangle = np.tril(matrix, k=-1)
        matrix = lower_triangle + lower_triangle.T
        # 保存转换后的数据
        mapped_data[i] = matrix
    return mapped_data



def mst_threshold(Co, MyCost, bin=True):
    """
    Thresholds the matrix Co based on the minimum spanning tree (MST).

    Parameters:
    - Co: Weighted matrix.
    - MyCost: Cost (in range [0,1]).
    - bin: Boolean indicating if binary (default) or weighted.

    Returns:
    - A: Thresholded matrix.
    """

    # Ensure the matrix is symmetric
    Co = (Co + Co.T) / 2
    # Set negative values to 0
    Co[Co < 0] = 0
    # Set diagonal to ones
    np.fill_diagonal(Co, 1)

    # Convert the matrix to a graph representation
    G = nx.from_numpy_array(1 / (Co+1e-7))  # We invert the matrix as we want to find the minimum spanning tree

    # Calculate the MST using Kruskal's algorithm
    MST = nx.minimum_spanning_tree(G, algorithm="kruskal")

    # Convert MST back to adjacency matrix format
    A = nx.adjacency_matrix(MST).toarray()

    # Order edges based on their weight in the Co matrix
    triu_indices = np.triu_indices_from(Co, k=1)
    weights = Co[triu_indices]
    sorted_indices = np.argsort(-weights)  # Sort in descending order

    # Grow the network based on weights in Co matrix
    t = 0
    enum = np.sum(A) / 2  # count of edges (divided by 2 to avoid double counting)
    while enum < MyCost * Co.shape[0] * (Co.shape[0] - 1) / 2:
        i, j = triu_indices[0][sorted_indices[t]], triu_indices[1][sorted_indices[t]]
        if A[i][j] == 0:
            if bin:
                A[i][j] = 1
                A[j][i] = 1
            else:
                A[i][j] = Co[i][j]
                A[j][i] = Co[i][j]
            enum += 1
        t += 1

    if not bin:
        A = Co * (A != 0)
        A = A + A.T - np.diag(np.diag(A))  # symmetrize the matrix

    A[A != 0] = 1
    return A




def warshall(A):
    n = A.shape[0]
    R = A.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                R[i][j] = R[i][j] or (R[i][k] and R[k][j])
    return R


def batch_warshall(data_3d):
    batch_size = data_3d.shape[0]
    results = np.zeros_like(data_3d)
    for i in range(batch_size):
        results[i] = warshall(data_3d[i])
    return results