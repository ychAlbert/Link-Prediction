import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.EnhancedModel import GraphAttentionLayer, MultiHeadAttention, RelationAttention, DynamicRelationAggregation, GraphTransformerLayer
from src.Model import GraphConvolution


class EnhancedMHGCN(nn.Module):
    """
    增强型多关系异构图卷积网络 (Enhanced Multi-relation Heterogeneous Graph Convolutional Network)
    结合了注意力机制、图变换器和对比学习框架
    """
    def __init__(self, nfeat, nhid, out, dropout=0.1, alpha=0.2, nheads=8):
        super(EnhancedMHGCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.out = out
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        
        # 根据数据集名称自动设置关系数量
        dataset_name = __import__('sys').argv[0].split('_')[0].split('/')[-1].split('\\')[-1]
        
        if 'Alibaba' in dataset_name or 'small_alibaba_1_10' in dataset_name:
            self.num_relations = 4
        elif 'DBLP' in dataset_name:
            self.num_relations = 3
        elif 'Aminer' in dataset_name or 'IMDB' in dataset_name or 'imdb' in dataset_name:
            self.num_relations = 2
        else:
            self.num_relations = 4  # 默认使用Alibaba的配置
        
        # 动态关系聚合模块
        self.relation_aggregation = DynamicRelationAggregation(self.num_relations)
        
        # 特征变换层
        self.feature_transform = nn.Linear(nfeat, nhid)
        
        # 图注意力层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha) 
            for _ in range(nheads)
        ])
        
        # 图变换器层
        self.transformer_layer = GraphTransformerLayer(
            in_features=nhid * nheads,
            out_features=out,
            num_heads=nheads,
            dropout=dropout,
            residual=True
        )
        
        # 输出层
        self.out_layer = nn.Linear(out, out)
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(out, out),
            nn.ReLU(),
            nn.Linear(out, out)
        )
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        参数初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.414)
    
    def forward(self, feature, A, return_attention=False):
        """
        前向传播
        feature: 节点特征
        A: 关系邻接矩阵列表
        return_attention: 是否返回注意力权重
        """
        # 将特征转换为张量
        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                # 如果已经是张量，则不需要转换
                if not isinstance(feature, torch.Tensor):
                    try:
                        feature = torch.tensor(feature)
                    except:
                        pass
        
        # 确保所有张量的 dtype 一致
        if isinstance(feature, torch.Tensor):
            feature = feature.float()
        A = [a.float() if isinstance(a, torch.Tensor) else a for a in A]
        
        # 动态关系聚合
        final_A, relation_attention = self.relation_aggregation(A)
        
        # 特征变换
        x = F.dropout(F.relu(self.feature_transform(feature)), self.dropout, training=self.training)
        
        # 多头图注意力
        gat_outputs = []
        for gat in self.gat_layers:
            gat_outputs.append(gat(x, final_A))
        
        # 拼接多头注意力的输出
        x = torch.cat(gat_outputs, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 图变换器层
        x = self.transformer_layer(x.unsqueeze(0), final_A).squeeze(0)
        
        # 输出层
        output = self.out_layer(x)
        
        if return_attention:
            return output, relation_attention
        else:
            return output
    
    def get_embeddings(self, feature, A):
        """
        获取节点嵌入
        """
        return self.forward(feature, A)
    
    def get_projected_embeddings(self, feature, A):
        """
        获取投影后的节点嵌入（用于对比学习）
        """
        embeddings = self.forward(feature, A)
        return self.projection_head(embeddings)


class ContrastiveLoss(nn.Module):
    """
    对比损失函数
    基于InfoNCE损失
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z_i, z_j):
        """
        计算对比损失
        z_i, z_j: 同一节点的两个不同视图的嵌入
        """
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        
        # 正样本对应的标签（对角线元素）
        labels = torch.arange(batch_size).to(z_i.device)
        
        # 计算损失（同时考虑 i->j 和 j->i 两个方向）
        loss_i = self.criterion(sim_matrix, labels)
        loss_j = self.criterion(sim_matrix.T, labels)
        
        return (loss_i + loss_j) / 2


class EnhancedMHGCNWithContrastive(EnhancedMHGCN):
    """
    带有对比学习的增强型多关系异构图卷积网络
    """
    def __init__(self, nfeat, nhid, out, dropout=0.1, alpha=0.2, nheads=8, temperature=0.5):
        super(EnhancedMHGCNWithContrastive, self).__init__(nfeat, nhid, out, dropout, alpha, nheads)
        self.contrastive_loss = ContrastiveLoss(temperature)
        
        # 特征增强器（用于生成不同视图）
        self.augmentor1 = nn.Sequential(
            nn.Linear(nfeat, nfeat),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.augmentor2 = nn.Sequential(
            nn.Linear(nfeat, nfeat),
            nn.Dropout(dropout * 1.5),  # 使用不同的dropout率生成不同视图
            nn.ReLU()
        )
    
    def forward(self, feature, A, return_attention=False, return_loss=False):
        """
        前向传播，可选择是否返回对比损失
        """
        # 标准前向传播
        if not return_loss:
            return super().forward(feature, A, return_attention)
        
        # 生成两个不同的视图
        try:
            feature_tensor = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature_tensor = torch.from_numpy(feature.toarray())
            except:
                if isinstance(feature, torch.Tensor):
                    feature_tensor = feature
                else:
                    try:
                        feature_tensor = torch.tensor(feature)
                    except:
                        feature_tensor = feature
        
        view1 = self.augmentor1(feature_tensor)
        view2 = self.augmentor2(feature_tensor)
        
        # 获取两个视图的嵌入
        z1 = self.get_projected_embeddings(view1, A)
        z2 = self.get_projected_embeddings(view2, A)
        
        # 计算对比损失
        loss = self.contrastive_loss(z1, z2)
        
        # 返回标准嵌入和对比损失
        embeddings = super().forward(feature, A, return_attention)
        
        if isinstance(embeddings, tuple):
            return embeddings[0], loss, embeddings[1]  # 嵌入、损失、注意力
        else:
            return embeddings, loss  # 嵌入、损失