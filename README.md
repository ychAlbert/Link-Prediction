# MHGCN
本仓库提供了 MHGCN 的参考实现，如论文中所述：
> 多重异构图卷积网络 (Multiplex Heterogeneous Graph Convolutional Network)
>
> 作者：Pengyang Yu, Chaofan Fu, Yanwei Yu, Chao Huang, Zhongying Zhao, Junyu Dong.
>
> 发表于 KDD'22

可在 [https://doi.org/10.1145/3534678.3539482](https://doi.org/10.1145/3534678.3539482) 查看。

## 依赖项
需要以下 Python 3 的最新版本包：
* numpy==1.21.2
* torch==1.9.1
* scipy==1.7.1
* scikit-learn==0.24.2
* pandas==0.25.0

## 数据集
### 链接
使用到的数据集可以从以下链接获取：
* Alibaba: [天池竞赛](https://tianchi.aliyun.com/competition/entrance/231719/information/)
* Amazon: [亚马逊数据](http://jmcauley.ucsd.edu/data/amazon/)
* Aminer: [Aminer 数据集](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding/tree/master/Aminer)
* IMDB: [IMDB 数据集](https://github.com/seongjunyun/Graph_Transformer_Networks)
* DBLP: [DBLP 数据集](https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0)

### 预处理
我们将数据集压缩成 mat 格式的文件，包含以下内容：
* edges: 耦合后的子网络数组，每个元素是一个子网络。
* features: 网络中每个节点的属性。
* labels: 标记点的标签。
* train: 用于节点分类任务的训练集索引。
* valid: 用于节点分类任务的验证集索引。
* test: 用于节点分类任务的测试集索引。

此外，我们还在网络中采样了正负边，并将它们分为三个文本文件：train、valid 和 test，用于链接预测任务。

## 使用方法
首先，你需要确定数据集。如果你要进行节点分类任务，需要修改 `Node_classfication.py` 中的数据集路径。如果你要进行链接预测任务，需要修改 `Link_prediction.py` 中的数据集路径。

其次，你需要修改 `Model.py` 中的权重数量。权重数量应为解耦后的子网络数量。

最后，你需要在 `Decoupling_matrix_aggregation.py` 中确定子网络及其数量。

执行以下命令运行节点分类任务：

* `python Node_Classfication.py`

执行以下命令运行链接预测任务：

* `python Link_Prediction.py`

## 引用
如果你在研究中使用了 MHGCN，请引用以下论文：

``` 
@inproceedings{yu2022multiplex,
  title={多重异构图卷积网络},
  author={Yu, Pengyang and Fu, Chaofan and Yu, Yanwei and Huang, Chao and Zhao, Zhongying and Dong, Junyu},
  booktitle={第 28 届 ACM SIGKDD 知识发现与数据挖掘会议论文集},
  pages={2377--2387},
  year={2022}
} 
```