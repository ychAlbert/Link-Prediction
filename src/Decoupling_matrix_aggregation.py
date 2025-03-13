import numpy as np
import torch
from scipy.sparse import coo_matrix


def coototensor(A):
    """
    Convert a coo_matrix to a torch sparse tensor
    """

    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

def adj_matrix_weight_merge(A, adj_weight):
    """
    Multiplex Relation Aggregation
    """

    N = A[0][0].shape[0]
    temp = coo_matrix((N, N))
    temp = coototensor(temp)
    
    # 根据运行的脚本和权重维度自动选择合适的聚合方式
    dataset_name = __import__('sys').argv[0].split('_')[0].split('/')[-1].split('\\')[-1]
    weight_dim = adj_weight.shape[0]
    
    if 'Alibaba' in dataset_name or ('small_alibaba_1_10' in dataset_name) or weight_dim == 4:
        # Alibaba - 4个关系
        a = coototensor(A[0][0].tocoo())
        b = coototensor(A[1][0].tocoo())
        c = coototensor(A[2][0].tocoo())
        d = coototensor(A[3][0].tocoo())
        A_t = torch.stack([a, b, c, d], dim=2).to_dense()
    elif 'DBLP' in dataset_name or weight_dim == 3:
        # DBLP - 3个关系
        a = coototensor(A[0][0].tocoo())
        b = coototensor(A[0][1].tocoo())
        c = coototensor(A[0][2].tocoo())
        A_t = torch.stack([a, b, c], dim=2).to_dense()
    elif 'Aminer' in dataset_name:
        # Aminer - 2个关系 (PAP, PTP)
        a = coototensor(A[0][0].tocoo())
        c = coototensor(A[0][2].tocoo())
        A_t = torch.stack([a, c], dim=2).to_dense()
    elif 'IMDB' in dataset_name or 'imdb' in dataset_name or weight_dim == 2:
        # IMDB - 2个关系
        a = coototensor(A[0][0].tocoo())
        b = coototensor(A[0][2].tocoo())
        A_t = torch.stack([a, b], dim=2).to_dense()
    else:
        # 默认使用Alibaba的配置
        try:
            a = coototensor(A[0][0].tocoo())
            b = coototensor(A[1][0].tocoo())
            c = coototensor(A[2][0].tocoo())
            d = coototensor(A[3][0].tocoo())
            A_t = torch.stack([a, b, c, d], dim=2).to_dense()
        except:
            # 如果默认配置失败，尝试DBLP配置
            a = coototensor(A[0][0].tocoo())
            b = coototensor(A[0][1].tocoo())
            c = coototensor(A[0][2].tocoo())
            A_t = torch.stack([a, b, c], dim=2).to_dense()

    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp, 2)

    return temp + temp.transpose(0, 1)