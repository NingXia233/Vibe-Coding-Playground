# Notes for Week2

神经网络架构 & Graphical Model

## Generative Model & Discriminative Model

Discriminative：判别
Generative: 生成

如果有生成模型，是可以反过来实现Discriminative

Discriminative Model的数学: $P(Y|X)$，核心问题是：Y如何随着X变化
- 对应Supervised Learning的任务Task
- 物理上对应Phase Classification的任务

Generative Model的数学：$P(X,Y)$，如何通过联合概率学习到这个分布
- 根据贝叶斯：$P(X,Y)=P(Y|X)P(X)$，因此有生成模型就可以实现Discriminative模型


如何用统一的思想来理解百花齐放的生成模型？
- 第一层：Generative
- 第二层：Expilicit Density; Implicit Density
- 第三层：Tractable density, Approximate density; Direct(GAN,easy to sample)
- 第四层: Pixel-wise, Flow based; Variational approx.(VAE), Markov-chain(Boltzman machine). MCMC(GSN,not easy to sample)

Tensor Network实际上也属于Tractable density model

第一层划分是我是否可以把概率分布公式给写出来，还是只能采样
第二层划分是对于Explicit Density公式写出来后我是不是可以在多项式时间内算出来；对于Implicit Density我们的分类则是能不能直接生成样本

Graphical Model is one of the representive Generative Model, minimal model in explicit density model (both tractable and approximate)

What is graphical model?

We have a graph $G=(V,E)$,
consider $N$ binary varibale $X_1,..., X_N$,
the dimension is $2^N-1$, however, we may decompose high dimensional joint distribution in a graph.

Two kinds of graph: Directed Graphical Models（有向，比如变量间的影响，可以引入因果律causality，可以引入条件概率），Undirected Graphical Model

Directed Graphical Models(Bayesian Networks)

Undirected Graphical Model (Energy Based Model)
对于无向图我们总是能够定义出对应的能量及对应的Boltzman分布/Partition Function

有了Model之后要怎么用模型来描述实际数据？
Learning: 通过minimize KL divergence来优化参数

描述两组数据分布接近程度:KL divergence
Forward KL & Reverse KL

物理上，KL divergence代入Boltzman分布就是自由能


为了minimize KL divergence，我们需要进行求导:$\nabla_x log P_\theta(x)$

Score Matching: 为了避免对参数求导的计算，可以巧妙地直接对导数参数化。（这也就是为什么Diffusion model是approximate而不是tractable）


## Vibe-Coding Works

Hidden-Markov Model

