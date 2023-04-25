## Deep Learning for Click-Through Rate Estimation

### 0. 概括

本文是一篇关于深度学习在CTR预估中的综述，文章介绍了CTR预估的重要性以及深度学习模型如何提高CTR预估的性能。


### 1. CTR背景

CTR预估是指通过分析用户的历史行为数据，预测用户是否会点击某个广告或者某个推荐内容。CTR预估任务的数据案例如下图所示：

<div align="center">
<img src=Figure/ClickInstance.png width=75% />
</div>

CTR预估的模型训练任务可构建为二分类问题，且损失函数如下：
$$ \mathcal{L}(x, y, \theta) = -y\cdot \log \sigma(f_{\theta}(x)) - (1-y)\cdot \log (1-\sigma(f_{\theta} (x))) $$


### 2. CTR模型发展趋势

<div align="center">
<img src=Figure/CTRModelTrend.png width=60% />
</div>

#### 2-1 传统模型
* **LR回归** (2007)
$$ f_{\theta}^{LR} (x) = \theta_0 + \sum_{i=1}^{m} x_i \theta_i $$
* **POLY2** (2010)
* **GBDT** (2014)
* **FM** (2010) 特征的二阶交互，$v_i$是分配给特征$i$的可学习的嵌入向量
$$ f_{\theta}^{FM} (x) = \theta_0 + \sum_{i=1}^{m} x_i \theta_i + \sum_{i=1}^{m} \sum_{j=i+1}^{m} x_i x_j v_i^{T} v_j $$
* **GBFM** (2014): Gradient Boosting FM
* **HOFM** (2016): Higher-Order FM
* **FFM** (2016) : Field-aware FM
* **FwFM** (2018): Field-weighted FM

#### 2-2 深度模型
* Wide & Deep network (2016)
$$ f_{\theta}^{W D} (x) = \theta_0 + \sum_{i=1}^{m} x_i \theta_i + \mathrm{MLP}_{\phi} ([v_1, v_2, \cdots, v_m]) $$
* DeepCross (2016) 引入残差网络


### 3. 特征交互

#### 3-1 特征交互算子

<div align="center">
<img src=Figure/InteractionOperators.png width=60% />
</div>

* **Product Operators**：捕获不同特征之间的相互作用，从而提高CTR模型的准确性。在传统CTR模型中，通常只考虑单个特征对点击率的影响，
而忽略了不同特征之间的相互作用。而Product Operator可以将多个特征组合起来，捕获它们之间的相互作用，从而更好地预测用户点击行为。PNN，KPNN，PIN，NFM。
* **Convolutional Operators**：捕获不同特征之间的局部关系，从而提高模型的准确性。
与Product Operator不同，Convolutional Operator可以考虑到不同特征之间的顺序关系，并且可以捕获更广泛的特征交互。CCPM，FGCNN，FiGNN。
* **Attention Operators**：捕获不同特征之间的重要性差异，并根据这些重要性差异来进行特征交互。
与Product Operator和Convolutional Operator不同，Attention Operator可以动态地调整不同特征之间的权重，从而更好地适应不同用户和场景。AFM，FiBiNET，AutoInt，InterHAt。

#### 3-2 DNN在DeepCTR中的作用

<div align="center">
<img src=Figure/TowerModel.png width=60% />
</div>

* **Single Tower 单塔模型**：Embedding - 特征交互 - DNN - CTR，有效捕捉高阶特征交互，但低阶特征交互的信号可能在后续的DNN中消失。NFM, PIN。
* **Dual Tower 双塔模型**：Embedding - 特征交互 & DNN 并行 - CTR，主流模型，建模能力更强。Wide & Deep，DeepFM，DCN，DCN V2，xDeepFM，Autoint+等。


### 4. 用户行为建模

<div align="center">
<img src=Figure/UserBehaviorModeling.png width=50% />
</div>

#### 4-1 Attention based Models

* DIN：学习用户对于不同消费item的偏好
* DIEN：DIN + GRU，学习兴趣的演化
* BST：将多头注意力层作为序列特征提取器，以捕获行为之间的依赖关系。
* DSIN：用户行为分多个session，session内做self-attention，session间做Bi-LSTM

#### 4-2 Memory Network based Models [长序列]

* HPMN：多层记忆网络，用于刻画长序列的用户行为。
* UIC：多通道用户兴趣网络。

#### 4-3 Retrieval based Models [序列过长，进行检索]

* UBR4CTR：从全部历史序列中，检索出topK相关item。
* SIM：提出了硬检索和软检索两种方法。对于硬检索，它使用预定义的ID（例如用户ID和类别ID）来构建索引。对于软检索，SIM使用局部敏感哈希（LSH）来检索相关行为的嵌入。


### 5. 网络结构搜索

#### 5-1 Embedding Dimension Search

_Hard Selection_
* NIS：基于强化学习的深度CTR模型，用于自动搜索混合特征嵌入维度。NIS模型首先将通用维度空间划分为几个由人类专家预定义的块，然后应用强化学习来生成选择不同特征的维度块的决策序列。
* ESAPN：预定义了不同特征的候选嵌入维度集合，并为每个字段使用策略网络动态搜索不同特征的嵌入大小。
* DNIS：引入了一个二进制指示矩阵来表示每个特征块的存在，其中特征被分组到特征块中以减少搜索空间。然后，DNIS模型提出了一个软选择层来放宽二进制指示矩阵的搜索空间为连续值。
然后使用预定义的阈值来过滤软选择层中不重要的维度。
* PEP：引入了一个连续的候选维度集合来表示每个特征块的存在。

_Soft Selection_
* AutoEmb：一种软选择策略，通过可学习的权重对候选维度进行加权求和。这些权重是通过可微分搜索算法（例如DARTS）进行训练的。
* AutoDim：一种软选择策略，通过可学习的权重对不同特征字段分配不同的嵌入大小。
* DARTS：一种可微分的神经架构搜索方法，用于自动搜索神经网络的结构。DARTS通过在搜索空间中进行梯度下降来学习网络结构，从而实现自动化的神经网络设计。


#### 5-2 Feature Interaction Search

* AutoFIS：通过枚举所有特征交互并利用一组架构参数来指示各个特征交互的重要性。
* SIF：自动搜索适合矩阵分解的交互函数。
* AutoFeature：用不同结构的micro-network来模拟特征交互的过程。
* AutoGroup：提出了一种生成特征组的方法，使得给定阶数的特征交互是有效的。
* BP-FIS：通过贝叶斯变量选择来为不同用户识别重要的特征交互。

#### 5-3 Whole Architecture Search

* AutoCTR：设计了一个两级分层搜索空间，通过将现有CTR估计架构（即MLP、点积和因子分解机）中的代表性结构抽象为虚拟块，并将这些块连接成有向无环图（DAG。
外部空间由块之间的连接组成，而内部空间由不同块中的详细超参数组成。
* AMER：以从序列特征（即用户行为）中提取顺序表示为目标，并自动同时探索非顺序特征之间的不同特征交互。
一方面，行为建模的搜索空间包括归一化、激活和层选择（例如卷积、递归、池化、注意力层）。多个架构被随机抽样以在验证集上进行评估。


### 6. 总结展望

* 主要研究方向：特征交互、用户行为建模、自动搜索
* 挑战：深度学习理论、表征学习、多模态的CTR、有策略的数据处理。
