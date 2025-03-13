import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#鏉冮噸鐭╅樀
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#鍋忕Щ鍚戦噺
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    #
    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class MHGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(MHGCN, self).__init__()
        """
        # Multilayer Graph Convolution
        """
        self.gc1 = GraphConvolution(nfeat, out)
        self.gc2 = GraphConvolution(out, out)
        # self.gc3 = GraphConvolution(out, out)
        # self.gc3 = GraphConvolution(out, out)
        # self.gc4 = GraphConvolution(out, out)
        # self.gc5 = GraphConvolution(out, out)
        self.dropout = dropout

        """
        Set the trainable weight of adjacency matrix aggregation
        """
        # 根据数据集名称自动设置权重参数
        dataset_name = __import__('sys').argv[0].split('_')[0].split('/')[-1].split('\\')[-1]
        
        # 根据运行的脚本自动选择合适的权重维度
        if 'Alibaba' in dataset_name or 'small_alibaba_1_10' in dataset_name:
            # Alibaba
            self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
            torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        elif 'DBLP' in dataset_name:
            # DBLP
            self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
            torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        elif 'Aminer' in dataset_name:
            # Aminer
            self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
            torch.nn.init.uniform_(self.weight_b, a=0, b=1)
        elif 'IMDB' in dataset_name or 'imdb' in dataset_name:
            # IMDB
            self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
            torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        else:
            # 默认使用Alibaba的配置
            self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
            torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

    def forward(self, feature, A, use_relu=True):
        final_A = adj_matrix_weight_merge(A, self.weight_b)
        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        # Output of single-layer GCN
        U1 = self.gc1(feature, final_A)
        # Output of two-layer GCN
        U2 = self.gc2(U1, final_A)
        # U3 = self.gc3(U2, final_A)
        # U4 = self.gc4(U2, final_A)
        # U5 = self.gc5(U2, final_A)

        # Average pooling
        return (U1+U2)/2