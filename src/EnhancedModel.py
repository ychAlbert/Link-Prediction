import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from src.Decoupling_matrix_aggregation import coototensor


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (Graph Attention Layer)
    基于论文 "Graph Attention Networks" (Veličković et al., ICLR 2018)
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 线性变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力机制的参数
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # 线性变换
        h = torch.mm(input, self.W)
        N = h.size()[0]

        # 计算注意力系数
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 掩码机制：将不相连的节点对应的注意力系数设为负无穷
        zero_vec = -9e15*torch.ones_like(e)
        if isinstance(adj, torch.sparse.FloatTensor):
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        attention = torch.where(adj_dense > 0, e, zero_vec)
        
        # softmax归一化
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 聚合邻居特征
        h_prime = torch.matmul(attention, h)

        # 非线性激活
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    基于论文 "Attention Is All You Need" (Vaswani et al., NeurIPS 2017)
    """
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.6):
        super(MultiHeadAttention, self).__init__()
        assert out_features % num_heads == 0
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.dropout = dropout
        
        # 定义查询、键、值的线性变换
        self.q_linear = nn.Linear(in_features, out_features)
        self.k_linear = nn.Linear(in_features, out_features)
        self.v_linear = nn.Linear(in_features, out_features)
        
        # 输出线性变换
        self.out_linear = nn.Linear(out_features, out_features)
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, adj):
        batch_size = x.size(0)
        
        # 线性变换并分割为多头
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 掩码处理：将不相连的节点对应的注意力分数设为负无穷
        if isinstance(adj, torch.sparse.FloatTensor):
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj
        
        # 扩展邻接矩阵以适应多头注意力的形状
        adj_mask = adj_dense.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        zero_vec = -9e15*torch.ones_like(scores)
        scores = torch.where(adj_mask > 0, scores, zero_vec)
        
        # softmax归一化
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 聚合邻居特征
        out = torch.matmul(attention, v)
        
        # 重塑并连接多头的结果
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.out_features)
        
        # 最终线性变换
        out = self.out_linear(out)
        
        return out


class RelationAttention(nn.Module):
    """
    关系注意力层：动态学习不同关系的重要性
    """
    def __init__(self, relation_dim, attention_dim=64):
        super(RelationAttention, self).__init__()
        self.relation_dim = relation_dim
        self.attention_dim = attention_dim
        
        # 关系嵌入变换
        self.relation_transform = nn.Linear(relation_dim, attention_dim)
        
        # 注意力向量
        self.attention_vector = nn.Parameter(torch.FloatTensor(attention_dim, 1))
        nn.init.xavier_uniform_(self.attention_vector.data, gain=1.414)
        
        # 激活函数
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, relation_matrices):
        # relation_matrices: [batch_size, num_relations, dim]
        
        # 变换关系表示
        transformed = self.relation_transform(relation_matrices)  # [batch_size, num_relations, attention_dim]
        
        # 计算注意力分数
        attention_scores = self.leakyrelu(torch.matmul(transformed, self.attention_vector))  # [batch_size, num_relations, 1]
        
        # softmax归一化
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_relations, 1]
        
        # 加权聚合关系
        weighted_relations = relation_matrices * attention_weights  # [batch_size, num_relations, dim]
        aggregated_relation = torch.sum(weighted_relations, dim=1)  # [batch_size, dim]
        
        return aggregated_relation, attention_weights


class DynamicRelationAggregation(nn.Module):
    """
    动态关系聚合模块：使用注意力机制聚合多种关系
    """
    def __init__(self, num_relations):
        super(DynamicRelationAggregation, self).__init__()
        self.num_relations = num_relations
        
        # 关系重要性学习
        self.relation_importance = nn.Parameter(torch.FloatTensor(num_relations, 1))
        nn.init.xavier_uniform_(self.relation_importance.data, gain=1.414)
        
        # 关系特定变换
        self.relation_transforms = nn.ModuleList([
            nn.Linear(1, 1) for _ in range(num_relations)
        ])
        
        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(num_relations, 64),
            nn.ReLU(),
            nn.Linear(64, num_relations),
            nn.Softmax(dim=1)
        )
    
    def forward(self, A):
        """
        动态聚合多种关系
        A: 列表，包含多个关系的邻接矩阵
        """
        N = A[0][0].shape[0]
        device = self.relation_importance.device
        
        # 将所有关系转换为稠密张量
        relation_matrices = []
        
        # 根据数据集类型处理不同的关系结构
        if len(A) > 1:  # Alibaba类型
            for i in range(self.num_relations):
                rel_matrix = coototensor(A[i][0].tocoo()).to(device)
                relation_matrices.append(rel_matrix)
        else:  # DBLP, IMDB, Aminer类型
            for i in range(min(self.num_relations, len(A[0]))):
                rel_matrix = coototensor(A[0][i].tocoo()).to(device)
                relation_matrices.append(rel_matrix)
            
            # 如果关系数量不足，用零矩阵填充
            while len(relation_matrices) < self.num_relations:
                zero_matrix = torch.sparse_coo_tensor(
                    indices=torch.LongTensor([[0], [0]]),
                    values=torch.FloatTensor([0]),
                    size=(N, N)
                ).to(device)
                relation_matrices.append(zero_matrix)
        
        # 应用关系特定变换
        transformed_relations = []
        for i, rel_matrix in enumerate(relation_matrices):
            if rel_matrix.is_sparse:
                # 对于稀疏矩阵，我们只变换非零元素
                indices = rel_matrix._indices()
                values = rel_matrix._values()
                transformed_values = self.relation_transforms[i](values.unsqueeze(-1)).squeeze(-1)
                transformed_rel = torch.sparse_coo_tensor(
                    indices=indices,
                    values=transformed_values,
                    size=rel_matrix.size()
                )
                transformed_relations.append(transformed_rel)
            else:
                # 对于稠密矩阵，直接应用变换
                transformed_rel = self.relation_transforms[i](rel_matrix.unsqueeze(-1)).squeeze(-1)
                transformed_relations.append(transformed_rel)
        
        # 计算关系特征向量（用于注意力计算）
        relation_features = torch.stack([torch.sum(rel) for rel in transformed_relations])
        
        # 计算动态注意力权重
        attention_weights = self.attention_net(relation_features.unsqueeze(0)).squeeze(0)
        
        # 加权聚合关系
        final_adj = None
        for i, rel_matrix in enumerate(transformed_relations):
            if final_adj is None:
                final_adj = attention_weights[i] * rel_matrix
            else:
                final_adj = final_adj + attention_weights[i] * rel_matrix
        
        # 确保对称性
        if not final_adj.is_sparse:
            final_adj = (final_adj + final_adj.transpose(0, 1)) / 2
        
        return final_adj, attention_weights


class GraphTransformerLayer(nn.Module):
    """
    图变换器层：结合了图结构和Transformer的自注意力机制
    基于论文 "Graph Transformer Networks" (Yun et al., NeurIPS 2019)
    """
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.1, residual=True):
        super(GraphTransformerLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        
        # 多头自注意力
        self.attention = MultiHeadAttention(
            in_features=in_features,
            out_features=out_features,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(out_features, 4 * out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_features, out_features),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        # 自注意力机制
        attn_output = self.attention(x, adj)
        
        # 残差连接和Layer Normalization
        if self.residual and x.size(-1) == self.out_features:
            x = self.norm1(x + self.dropout(attn_output))
        else:
            x = self.norm1(attn_output)
        
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        
        # 残差连接和Layer Normalization
        x = self.norm2(x + ff_output)
        
        return x