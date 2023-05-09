# 史上最全的AI知识体系

# 目标

* **跟踪前沿技术方向。**
* **构建全知的AI知识体系。**


# 机器学习

## _一、基本概念_

**什么是机器学习**

机器学习（Machine Learning）是一种人工智能（Artificial Intelligence）的分支，也是一种通过算法和数学模型来让计算机系统自动学习新知识、提高自身性能的技术。
机器学习的核心目标就是通过从数据中自动推导出规律并预测结果。机器学习广泛应用于图像和语音识别、自然语言处理、文本分类、个性化推荐、金融风险管理、医疗研究与诊断等领域。

**机器学习的基本概念**

1. 数据集：机器学习算法需要大量的数据作为输入，这些数据被组织成一个数据集。
2. 测试集和验证集：为了评估模型的性能和泛化能力，在训练过程中需要将数据集划分为训练集、测试集和验证集。
其中，训练集用于模型的训练，测试集用于评估模型的性能，验证集用于调整模型参数和防止过拟合。
3. 特征：在数据集中，每个样本都有一组特征，用于描述该样本的属性。特征可以是数字、文本、图像等形式。
4. 标签：在监督学习中，每个样本都有一个标签或目标值，用于指示该样本所属的类别或预测值。
5. 模型：机器学习算法通过对数据集进行训练来构建一个模型，该模型可以对新的未知数据进行预测或分类。
6. 损失函数：损失函数用于衡量模型预测结果与真实结果之间的差距，并作为优化算法的目标函数。
7. 优化算法：优化算法用于调整模型参数以最小化损失函数，并提高模型的预测准确率。

**机器学习的入门课程**

* Standford CS229 - Machine Learning - Andrew Ng 
[[Syllabus]](http://cs229.stanford.edu/syllabus-autumn2018.html)
[[Youtube]](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
* UCB CS188 - Introduction to Artificial Intelligence
[[Syllabus]](https://inst.eecs.berkeley.edu/~cs188/fa18/index.html)
[[Youtube]](https://www.youtube.com/playlist?list=PL7k0r4t5c108AZRwfW-FhnkZ0sCKBChLH)
* Standford CS231 - Convolutional Neural Networks for Visual Recognition
[[Syllabus]](https://cs231n.github.io/)
[[Youtube]](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)

## _二、核心领域_

### 1. Generative AI



#### 1-A. **Generative Model**

生成模型是一种机器学习技术，可以用来自动根据历史数据创造出新的、可能的输出内容。其工作原理是通过使用数据集中出现的特征
（例如颜色、形状、位置和大小）来判断其他图片中应该存在何种对象。这些已有的特征帮助生成模型扫描图片并识别里面包含的对象。 
此外，生成模型也可以用来生成声音信号、文字信息和合成图片。

**Generative Model Survey**
* 扩散模型的缘起与今生 
[[pdf]](https://arxiv.org/pdf/2208.11970.pdf) 
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/Survey_DM.md)

**Key Paper List of Generative Model**

* **[VAE]** Auto-Encoding Variational Bayes
[[pdf]](https://arxiv.org/pdf/1312.6114.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/VAE.md)
* * **[CVAE]** Learning Structured Output Representation using Deep Conditional Generative Models
[[pdf]](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/CVAE.md)
* * **[$\beta-$VAE]** Understanding disentangling in $\beta-$VAE
[[pdf]](https://arxiv.org/pdf/1804.03599.pdf) 
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/Beta-VAE.md)
* * **[ControlVAE]** ControlVAE: Controllable Variational Autoencoder 
[[pdf]](http://proceedings.mlr.press/v119/shao20b/shao20b.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/ControlVAE.md)
* * **[VQ-VAE]** Neural Discrete Representation Learning
[[pdf]](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/VQ-VAE.md)
* **[GAN]** Generative Adversarial Networks
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3422622)
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/GAN.md)
* * **[VQ-GAN]** Taming Transformers for High-Resolution Image Synthesis
[[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/VQ-GAN.md)
* **[DDPM]** Denoising Diffusion Probabilistic Models
[[pdf]](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/DDPM.md)
* * **[LDM]** High-Resolution Image Synthesis with Latent Diffusion Models
[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) 
[[notes]](./Machine%20Learning/Generative%20AI/Generative%20Model/LDM.md)
* **[GPT-3]** Language Models are Few-Shot Learners
[[pdf]](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Pre-training%20Model/GPT3.md)


#### 1-B. Pre-training Model

预训练模型（Pre-training model）是指在机器学习算法中，使用大量训练数据，对模型进行预先训练，以获得更好的性能和更有效地学习结果。
预训练模型通常通过对模型参数进行训练来提高模型的性能，从而使模型在真实数据上有更好表现。
预训练模型的优点在于可以提高模型的性能，从而更好地满足实际应用的需求。但是，这种模型也存在一定的局限性，比如训练数据量较小时，可能无法获得有效的训练结果，或者训练模型可能会出现过拟合现象。


**Key Paper List of Pre-training Model**
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 
[[pdf]](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ)
[[notes]](./Machine%20Learning/Generative%20AI/Pre-training%20Model/BERT.md)
* Improving Language Understanding by Generative Pre-Training
[[pdf]](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Pre-training%20Model/GPT.md)
* Language Models are Unsupervised Multitask Learners
[[pdf]](https://cs.brown.edu/courses/csci1460/assets/papers/language_models_are_unsupervised_multitask_learners.pdf) 
[[notes]](./Machine%20Learning/Generative%20AI/Pre-training%20Model/GPT2.md)
* Language Models are Few-Shot Learners
[[pdf]](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Pre-training%20Model/GPT3.md)
* Training language models to follow instructions with human feedback
[[pdf]](https://arxiv.org/pdf/2203.02155.pdf?fbclid=IwAR2nZdBpdZZzvxpwI6H_bRmP4RwGOyzke9Ud63lWBe1YlyI_1BRAFhnUMUg)
[[notes]](./Machine%20Learning/Generative%20AI/Pre-training%20Model/InstructGPT.md)
* WebGPT: Browser-assisted question-answering with human feedback
[[pdf]](https://arxiv.org/pdf/2112.09332.pdf) 
[[notes]](./Machine%20Learning/Generative%20AI/Pre-training%20Model/WebGPT.md)
* LaMDA: Language Models for Dialog Applications
[[pdf]](https://arxiv.org/pdf/2201.08239.pdf) 
[[notes]](./Machine%20Learning/Generative%20AI/Pre-training%20Model/LaMDA.md)


#### 1-C In-Context Learning

情景式学习（In-Context Learning）是一种学习范式，它主要目的是使模型能够在实际情境中学习新知识。它强调以实际情境为基础，
通过反复练习，使模型能够有效地融入新知识。

* An Explanation of In-context Learning as Implicit Bayesian Inference
[[notes]](./Machine%20Learning/Generative%20AI/In-Context%20Learning/An%20Explanation%20of%20In-context%20Learning%20as%20Implicit%20Bayesian%20Inference.md)


#### 1-D Reinforcement Learning

强化学习（Reinforcement Learning）是一种机器学习方法，其目标是让一个智能体在经历不断的尝试和错误之后，学习到如何在一个环境中取得最优表现。
它是一种在环境中的自主学习，通过在环境中采取行动来获得反馈，然后根据反馈不断学习。

强化学习的过程可以描述为：智能体（Agent）从环境（Environment）中获得观察（Observation），然后根据观察选择行动（Action），
行动导致环境发生变化，变化产生新的状态和奖励（Reward），奖励可以让智能体学习到哪些行动有利于达到最终目标。

**Books**
* Reinforcement Learning: an Introduction 
[[pdf]](http://incompleteideas.net/book/RLbook2018.pdf)
* Algorithms for Reinforcement Learning 
[[pdf]](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)

**Lecture**
* UCL $\times$ DeepMind Reinforcement Learning 
[[slides]](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)

**Paper List**
* **[BasicNotes]** 
[[notes]](./Machine%20Learning/Generative%20AI/Reinforcement%20Learning/basicNote.md)
* **[DQN]** Human-level control through deep reinforcement learning
[[pdf]](https://daiwk.github.io/assets/dqn.pdf) 
[[notes]](./Machine%20Learning/Generative%20AI/Reinforcement%20Learning/DQN.md)
* **[PPO]** Proximal Policy Optimization Algorithms
[[pdf]](https://arxiv.org/pdf/1707.06347.pdf)
[[notes]](./Machine%20Learning/Generative%20AI/Reinforcement%20Learning/PPO.md)


### 2. Graph/Network

图学习（Graph Learning）是一种机器学习技术，它使用图表来表示数据，并利用图表结构中的信息来进行分析和预测。它既可以用于分类任务，
也可以用于非监督学习，因为它可以找出数据之间的相互关系，可以帮助我们看出不同的关系，从而发现有价值的信息。
它可以模拟复杂的关系，可以对大规模数据集进行有效的分析，也可以查找节点间的关系，从而更好地理解数据。

图学习可以用来构建复杂的模型，以发现数据之间的关系，从而进行精确的分类和预测。它也可以用于探查数据中的模式，有助于提高算法的准确性和效率。
图学习的应用非常广泛，可以用于计算机视觉，自然语言处理，社交网络分析，推荐系统，机器人控制等等。

#### 2-A Graph Representation Learning

图表征学习（Graph Representation Learning）是指将图中的节点、边或子图映射到低维向量空间中的过程。
这些向量可以用于表示节点、边或子图的特征，从而使得图数据可以被传统的机器学习算法所处理。
与传统的机器学习方法不同，图表征学习方法可以自动地从数据中学习节点和边的特征，而无需手动设计特征。

**Books**
* Graph Representation Learning
[[pdf]](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)

**Paper List**
1. A tutorial on spectral clustering
[[pdf]](https://arxiv.org/pdf/0711.0189.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/SpectralClustering.md)
2. **[Graphlet]** Efficient graphlet kernels for large graph comparison
[[pdf]](http://proceedings.mlr.press/v5/shervashidze09a/shervashidze09a.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/graphlet.md)
3. **[TransE]** Translating Embeddings for Modeling Multi-relational Data
[[pdf]](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/TransE.md)
4. **[TransH]** Knowledge Graph Embedding by Translating on Hyperplanes
[[pdf]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2a3f862199883ceff5e3c74126f0c80770653e05)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/TransH.md)
5. **[TransR]** Learning Entity and Relation Embeddings for Knowledge Graph Completion
[[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/9491/9350)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/TransR.md)
6. **[TransD]** Knowledge Graph Embedding via Dynamic Mapping Matrix
[[pdf]](https://aclanthology.org/P15-1067.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/TransD.md)
7. **[DeepWalk]** DeepWalk: Online Learning of Social Representations
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/2623330.2623732)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/DeepWalk.md)
8. **[LINE]** LINE: Large-scale Information Network Embedding
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/2736277.2741093)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/LINE.md)
9. **[AWE]** Anonymous Walk Embeddings
[[pdf]](http://proceedings.mlr.press/v80/ivanov18a/ivanov18a.pdf) 
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/AnonymousWalk.md)
10. **[graph2vec]** graph2vec: Learning Distributed Representations of Graphs
[[pdf]](https://arxiv.org/pdf/1707.05005.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Representation%20Learning/graph2vec.md)

#### 2-B Graph Neural Networks

图神经网络（Graph Neural Networks，GNNs）是一种用于处理图数据的深度学习模型。与传统的深度学习模型不同，GNNs可以直接处理图结构数据，
而无需将其转换为向量或矩阵形式。GNNs通过在节点和边上执行信息传递和聚合操作来学习节点和图级别的表示。

**Lecture**
* CS224W: Machine Learning with Graphs
[[slides]](http://web.stanford.edu/class/cs224w/)
* CS228: Probabilistic Graphical Models
[[syllabus]](https://ermongroup.github.io/cs228/) [[contents]](https://ermongroup.github.io/cs228-notes/)

**Paper List**
1. **[GCN]** Semi-Supervised Classification with Graph Convolutional Networks
[[pdf]](https://arxiv.org/pdf/1609.02907.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Neural%20Networks/GCN.md)
2. **[R-GCNs]** Modeling Relational Data with Graph Convolutional Networks
[[pdf]](https://arxiv.org/pdf/1703.06103.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Neural%20Networks/R-GCNs.md)
3. **[GraphSage]** Inductive Representation Learning on Large Graphs
[[pdf]](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Neural%20Networks/GraphSage.md)
4. **[GAT]** Graph Attention Networks
[[pdf]](https://arxiv.org/pdf/1710.10903.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Neural%20Networks/GAT.md)
5. **[HAN]** Heterogeneous Graph Attention Network
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3308558.3313562)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Neural%20Networks/HAN.md)
6. **[GTN]** Graph Transformer Networks
[[pdf]](https://proceedings.neurips.cc/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf)
[[notes]](./Machine%20Learning/Graph-Network/Graph%20Neural%20Networks/GTN.md)


### 3. Learning Paradigm

#### 3-A Active Learning

主动学习是一种机器学习方法，其中算法可以与人类专家进行交互，以便在训练模型时选择最有用的数据。
在主动学习中，算法会自动选择最具信息量的样本进行标记，以便在训练过程中提高模型的准确性。
这种方法通常用于大型数据集或标记成本较高的情况下，因为它可以减少需要标记的样本数量，并且可以更快地训练模型。

* Active Learning with Label Comparisons
[[pdf]](https://proceedings.mlr.press/v180/yona22a/yona22a.pdf)
[[notes]](./Machine%20Learning/Learning%20Paradigm/Active%20Learning/NbrGraphSGD.md)


#### 3-B Incremental Learning

增量学习是一种机器学习方法，其中模型可以在不重新训练的情况下逐步学习新的数据。
这种方法通常用于当新数据不断到来时需要更新模型时，以避免重新训练整个模型所需的时间和计算成本。
增量学习可以通过添加新的训练样本或类别来扩展模型，并且可以在保持先前知识的同时，逐步适应新数据。

#### 3-C Transfer Learning

迁移学习是指将已经训练好的模型应用于新任务或新领域，以提高模型性能和泛化能力的技术。
在传统机器学习中，每个任务都需要独立地进行数据收集、特征提取和模型训练，这样会浪费大量时间和资源。
而迁移学习可以利用已有的知识和经验，快速构建适用于新任务的模型。

#### 3-D Online Learning

在线学习是一种机器学习方法，其中模型可以在不断到来的数据流上进行实时训练和更新。在线学习通常用于需要快速适应新数据的场景。
与传统的批量学习不同，在线学习可以在不断到来的数据上进行增量式训练，并且可以随着时间的推移逐步改进模型。

#### 3-E Semi-Supervised Learning

半监督学习是一种机器学习方法，它结合了有标签数据和无标签数据来训练模型。
在半监督学习中，算法使用少量的有标签数据来指导模型的训练，并使用大量的无标签数据来提高模型的准确性。
这种方法通常用于当有标签数据很难获取或成本很高时，但是可以轻松获得大量未标记的数据。

* AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation
[[pdf]](https://arxiv.org/pdf/2106.04732.pdf)
[[notes]](./Machine%20Learning/Learning%20Paradigm/Semi-Supervised%20Learning/AdaMatch.md)


### 4. Bayesian Inference

**简要介绍**

贝叶斯学习（Bayesian Learning）是一种基于概率的机器学习方法，它以贝叶斯定理为基础，利用收集的数据对模型进行训练并作出预测。
它可以帮助AI在遇到新问题时根据已有信息来产生合理的预测。在Bayesian Learning中，使用不同形式的贝叶斯公式来衡量各个特征之间相关性，
从而准确估计参数。此外，Bayesian Learning还能够通过对数据所包含信念或先前证明所得到结论之间相关性进行评估来帮助AI作出决定。

**Paper List**

* Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
[[pdf]](http://proceedings.mlr.press/v48/gal16.pdf) 
[[notes]](./Machine%20Learning/Bayesian%20Inference/Dropout.md)



### 5. Optimization

#### 5-A Gradient-based Optimization

基于梯度的优化方法是一种常见的优化算法，用于在机器学习中训练模型。该方法通过计算损失函数相对于模型参数的梯度来更新模型参数，以最小化损失函数。
这些方法通常使用反向传播算法来计算梯度，并使用梯度下降等技术来更新模型参数。

* BasicNotes [[notes]](./Machine%20Learning/Optimization/GradientOpt/basisNote.md)
* SGD [[notes]](./Machine%20Learning/Optimization/GradientOpt/SGD.md)
* Momentum [[notes]](./Machine%20Learning/Optimization/GradientOpt/momentum.md)
* RMSProp [[notes]](./Machine%20Learning/Optimization/GradientOpt/RMSProp.md)
* AdaGrad [[pdf]](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) 
[[notes]](./Machine%20Learning/Optimization/GradientOpt/adaGrad.md)
* Adam: [[pdf]](https://arxiv.org/pdf/1412.6980.pdf) 
[[notes]](./Machine%20Learning/Optimization/GradientOpt/adam.md)



# 应用场景

## *一、智能机器人领域*

### 1. SLAM

SLAM（Simultaneous Localization and Mapping）指的是同时完成机器人的定位和地图构建的技术。
SLAM的核心思想是利用传感器（如激光雷达，视觉等）搜集的数据来实时构建、更新和维护机器人的位置和环境地图，从而实现机器人的自主定位、自主导航等。
SLAM的目的在于实现机器人的自主理解和规划，并且能够面对环境变化快速响应。 
SLAM的技术结构上分为前端传感器数据处理和后端建模优化两个部分。
前端传感器处理主要包括数据采集、特征提取、路径追踪、数据融合等，以实现机器人的定位；后端建模优化主要包括非线性优化、地图构建等，实现机器人的环境建模和地图更新等功能。 
SLAM技术对机器人导航技术具有重要意义，能够有效提高机器人的自主性，为机器人导航赋予智能。
SLAM技术在机器人自动化领域也有着广泛应用，如自动驾驶、服务机器人等。



## *二、计算机视觉领域*





## *三、自然语言处理领域*




## *四、推荐系统领域*

### 1. Click-Through Rate

CTR预估是指通过分析用户的历史行为数据，预测用户是否会点击某个广告或者某个推荐内容。CTR表示广告或者推荐内容被点击的概率。
CTR预估在在线广告、推荐系统、搜索引擎等个性化在线服务中扮演着核心的功能模块。
通过CTR预估，可以提高广告和推荐内容的精准度，从而提高用户的满意度和平台的收益。

**Click-Through Rate Survey**
* 深度学习与CTR预估的缘起与今生
[[pdf]](https://arxiv.org/pdf/2104.10584.pdf)
[[notes]](./Applications/Ads%20Recommendation/Click-Through%20Rate/Survey_CTR.md)

## *五、程序语言领域*

### 1. Code Generation

代码生成是一种自动生成软件代码的技术。它可以根据用户提供的设计或模型自动生成软件代码，从而大大简化编码工作，提高软件开发的效率。 
代码生成可以分为四个基本步骤：分析、转换、生成和调试。分析阶段，用户需要提供模型或设计，以便系统能够识别要实现的功能。
在转换阶段，系统将模型或设计转换成代码，这一步的转换可以是从模型到代码的直接转换或者从模型到另一个中间模型的转换，然后再从中间模型到代码的转换。
在生成阶段，系统根据转换的结果生成代码。最后，在调试阶段，用户可以通过查看代码，检查并修正生成的代码，以确保生成的代码满足用户的要求。
代码生成技术可以用来自动生成各种软件，包括桌面应用程序、移动应用程序、Web应用程序、Web服务、数据库应用程序等。
它可以帮助软件开发者以快速、高效的方式实现软件的开发，进而提高软件开发的效率。

* **[CodeBERT]** CodeBERT: A Pre-Trained Model for Programming and Natural Languages
[[pdf]](https://arxiv.org/pdf/2002.08155.pdf)
[[notes]](./Applications/Programming%20Language/Code%20Generation/CodeBERT.md)
* **[PyMT5]** PYMT5: multi-mode translation of natural language and PYTHON code with transformers
[[pdf]](https://arxiv.org/pdf/2010.03150.pdf)
[[notes]](./Applications/Programming%20Language/Code%20Generation/PyMT5.md)
* **[GraphCodeBERT]** GraphCodeBERT: Pre-training Code Representations with Data Flow
[[pdf]](https://arxiv.org/pdf/2009.08366.pdf)
[[notes]](./Applications/Programming%20Language/Code%20Generation)
* **[PLBART]** Unified Pre-training for Program Understanding and Generation
[[pdf]](https://arxiv.org/pdf/2103.06333.pdf)
[[notes]](./Applications/Programming%20Language/Code%20Generation/PLBART.md)
* **[Codex]** Evaluating Large Language Models Trained on Code
[[pdf]](https://arxiv.org/pdf/2107.03374.pdf)
[[notes]](./Applications/Programming%20Language/Code%20Generation/Codex.md)
* **[AlphaCode]** Competition-Level Code Generation with AlphaCode
[[pdf]](https://arxiv.org/pdf/2203.07814.pdf)
[[notes]](./Applications/Programming%20Language/Code%20Generation/AlphaCode.md)


### 2. Program Debugging and Fault Localization

程序调试和故障定位是指在软件开发过程中，通过识别和修复程序中的错误来提高软件质量和可靠性的过程。
调试是指识别和修复程序中的错误，例如语法错误、逻辑错误或运行时错误。
故障定位是指确定导致程序故障的根本原因，例如特定输入数据或代码路径。
这些技术通常使用断点、日志记录、单元测试等工具来帮助开发人员识别和修复程序中的错误，
并使用代码覆盖率、变异测试等技术来帮助确定故障位置。调试和故障定位是软件开发过程中不可或缺的步骤，可以提高软件质量、可靠性和安全性。

* A Survey of Algorithmic Debugging
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3106740)
[[notes]](./Programming%20Language/Program%20Debugging/csur_algorithmic_debug.md)
* Survey of Source Code Bug Detection Based on Deep Learning
[[pdf]](http://www.jos.org.cn/josen/article/abstract/6696) 
[[notes]](./Programming%20Language/Program%20Debugging/js_survey.md)




### 3. Automatic Program Repair

自动程序修复 (APR) 旨在使用来自软件工程、计算机编程和机器学习的算法和技术来自动检测和修复计算机程序中的错误。
APR可以用来修复软件bug，优化代码，提高软件的整体性能。 它还可用于检测和修复代码中的安全漏洞。

* Automatic Patch Generation by Learning Correct Code
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/2837614.2837617)
[[notes]](./Programming%20Language/Automatic%20Program%20Repair/Prophet.md)
* TBar: Revisiting Template-Based Automated Program Repair
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3293882.3330577)
[[notes]](./Programming%20Language/Automatic%20Program%20Repair/TBar.md)
* DEAR: A Novel Deep Learning-based Approach for Automated Program Repair
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3510003.3510177) 
[[notes]](./Programming%20Language/Automatic%20Program%20Repair/DEAR.md)


## *六、数据库领域*

### 1. AI4DB



# 关键技术源码 

## 一、持续更新中