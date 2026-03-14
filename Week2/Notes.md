# Week 2 笔记：神经网络架构 & Graphical Model

---

## 1. 判别模型 vs 生成模型

### 1.1 基本概念

| | 判别模型 (Discriminative) | 生成模型 (Generative) |
|---|---|---|
| **数学形式** | $P(Y \mid X)$ | $P(X, Y)$ |
| **核心问题** | $Y$ 如何随 $X$ 变化 | 学习变量的联合概率分布 |
| **典型应用** | Supervised Learning、Phase Classification | 数据生成、密度估计 |

### 1.2 两者的关系

由贝叶斯公式：

$$P(X,Y) = P(Y \mid X) \cdot P(X)$$

因此，**拥有生成模型即可推导出判别模型**，反之则不成立。生成模型包含更多关于数据本身的信息。

---

## 2. 生成模型的统一分类体系

用层次化的视角理解各类生成模型：

```
Generative Models
├── Explicit Density（能写出概率密度函数）
│   ├── Tractable Density（可在多项式时间内精确计算）
│   │   ├── Pixel-wise 自回归模型（如 PixelRNN/PixelCNN）
│   │   ├── Flow-based 模型（如 RealNVP, Glow）
│   │   └── Tensor Network
│   └── Approximate Density（需要近似推断）
│       ├── Variational Approximation → VAE
│       └── Markov Chain → Boltzmann Machine
└── Implicit Density（无法写出密度函数，只能采样）
    ├── Direct Sampling → GAN（易于采样）
    └── MCMC Sampling → GSN（不易直接采样）
```

**分类准则：**

- **第一层（Explicit vs Implicit）：** 能否将概率分布的公式显式写出？还是只能通过采样间接获得？
- **第二层：**
  - Explicit 内部：写出公式后能否在多项式时间内精确计算？
  - Implicit 内部：能否直接生成样本？

---

## 3. 图模型 (Graphical Model)

### 3.1 概述

图模型是 **Explicit Density** 生成模型的代表，同时覆盖 Tractable 和 Approximate 两类。它是最基本（minimal）的显式密度模型之一。

### 3.2 基本定义

给定图 $G = (V, E)$，考虑 $N$ 个二值随机变量 $X_1, X_2, \dots, X_N$：

- 联合分布的参数空间维度为 $2^N - 1$（指数级）
- 图模型的核心思想：利用**图结构**将高维联合分布**分解**为局部因子的乘积，从而大幅降低复杂度

### 3.3 两类图模型

| | 有向图模型 (Directed) | 无向图模型 (Undirected) |
|---|---|---|
| **别名** | 贝叶斯网络 (Bayesian Network) | 能量模型 (Energy-Based Model) |
| **边的含义** | 变量间的因果/条件依赖关系 | 变量间的对称关联关系 |
| **分解方式** | 条件概率分解：$P(X) = \prod_i P(X_i \mid \text{Parents}(X_i))$ | 能量函数 + Boltzmann 分布 |
| **特点** | 天然引入因果律 (causality)、条件概率 | 可定义能量函数 $E(x)$ 及配分函数 $Z$ |

无向图模型中的概率分布：

$$P(x) = \frac{1}{Z} \exp\big(-E(x)\big), \quad Z = \sum_x \exp\big(-E(x)\big)$$

### 3.4 隐马尔可夫模型 (Hidden Markov Model, HMM)

HMM 是**有向图模型**的经典实例，其图结构为一条链，包含两类变量：

- **隐变量** $Z = (z_1, z_2, \dots, z_T)$：不可直接观测的离散状态序列
- **观测变量** $X = (x_1, x_2, \dots, x_T)$：由隐状态生成的可观测数据

```
z_1 --> z_2 --> z_3 --> ... --> z_T      (Hidden states)
 |       |       |               |
 v       v       v               v
x_1     x_2     x_3     ...    x_T      (Observations)
```

HMM 满足两个核心假设：

1. **马尔可夫性 (Markov Property)**：对 $t \geq 2$，$P(z_t \mid z_1, \dots, z_{t-1}) = P(z_t \mid z_{t-1})$
2. **观测独立性 (Observation Independence)**：$P(x_t \mid z_1, \dots, z_T) = P(x_t \mid z_t)$

由有向图的条件概率分解，HMM 的联合分布为：

$$P(X, Z) = \pi(z_1) \prod_{t=2}^{T} P(z_t \mid z_{t-1}) \prod_{t=1}^{T} P(x_t \mid z_t)$$

其中：
- $\pi(z_1) = P(z_1)$：**初始状态分布**，定义了第一个隐状态的先验概率
- $P(z_t \mid z_{t-1})$：由**转移矩阵** $A$ 定义，$A_{ij} = P(z_t = j \mid z_{t-1} = i)$
- $P(x_t \mid z_t)$：**发射分布 (Emission Distribution)**，定义了隐状态如何生成观测数据

> HMM 的三大经典问题：
> 1. **评估 (Evaluation)**：给定模型参数，计算 $P(X)$ — Forward Algorithm
> 2. **解码 (Decoding)**：给定观测序列，推断最可能的隐状态序列 — Viterbi Algorithm
> 3. **学习 (Learning)**：从数据中学习模型参数 — Baum-Welch (EM) Algorithm

---

## 4. 模型学习：最小化 KL 散度

### 4.1 学习目标

有了模型之后，如何让模型拟合真实数据？

> **核心方法：** 最小化模型分布 $P_\theta$ 与数据分布 $P_{\text{data}}$ 之间的 **KL 散度 (Kullback-Leibler Divergence)**。

### 4.2 KL 散度

KL 散度衡量两个概率分布之间的"距离"（注意：不对称）：

- **Forward KL：** $D_{\text{KL}}(P_{\text{data}} \| P_\theta)$ — 倾向于让模型覆盖数据的所有模式（mode-covering）
- **Reverse KL：** $D_{\text{KL}}(P_\theta \| P_{\text{data}})$ — 倾向于让模型集中在数据的某些模式上（mode-seeking）

### 4.3 变分推断 (Variational Inference) 与 ELBO

在许多模型（如 HMM、VAE）中，后验分布 $P(Z \mid X)$ 无法精确计算。变分推断通过引入一个**可调分布** $q(Z)$ 来近似后验。

#### 4.3.1 ELBO 推导

对数据的对数似然 $\log P(X)$ 进行分解：

$$\log P(X) = \underbrace{\mathbb{E}_{q(Z)}\big[\log P(X, Z) - \log q(Z)\big]}_{\text{ELBO}(q)} + D_{\text{KL}}\big(q(Z) \| P(Z \mid X)\big)$$

由于 KL 散度 $\geq 0$，因此 ELBO 是 $\log P(X)$ 的**下界**：

$$\text{ELBO}(q) = \mathbb{E}_{q(Z)}\big[\log P(X, Z)\big] + H(q) \leq \log P(X)$$

其中 $H(q) = -\mathbb{E}_{q}[\log q(Z)]$ 是 $q$ 的**熵**。

> **核心思想：** 最大化 ELBO 等价于最小化 $q(Z)$ 与真实后验 $P(Z \mid X)$ 之间的 KL 散度。

#### 4.3.2 变分自由能 (Variational Free Energy)

定义**变分自由能** $F$ 为 ELBO 的负数：

$$F = -\text{ELBO} = \underbrace{-\mathbb{E}_{q}[\log P(X, Z)]}_{\text{Expected Energy}} - \underbrace{H(q)}_{\text{Entropy}}$$

这与统计物理中的自由能公式 $F = E - TS$（令 $T=1$）完全对应：

| 统计物理 | 变分推断 |
|---|---|
| 自由能 $F$ | 变分自由能 $-\text{ELBO}$ |
| 内能 $E$ | $-\mathbb{E}_q[\log P(X, Z)]$ |
| 熵 $S$ | $H(q)$ |
| 平衡态 | 后验分布 $P(Z \mid X)$ |

**最小化变分自由能 $\Leftrightarrow$ 最大化 ELBO $\Leftrightarrow$ 逼近真实后验分布。**

### 4.4 均场近似 (Mean-Field Approximation)

当 $Z$ 包含多个隐变量时，直接优化 $q(Z)$ 仍然困难。均场近似假设 $q(Z)$ 可以**分解**为各变量的独立分布之积：

$$q(Z) = \prod_{t} q_t(z_t)$$

在此假设下，可通过 **坐标上升法 (Coordinate Ascent)** 逐个更新每个 $q_t$，每一步都保证 ELBO 不降：

$$\log q_t^*(z_t) \propto \mathbb{E}_{q_{-t}}\big[\log P(X, Z)\big]$$

其中 $q_{-t}$ 表示除 $q_t$ 以外所有变分因子。

> **物理类比：** 均场近似相当于将多体相互作用系统简化为每个粒子处在其他粒子的"平均场"中，独立演化。这是统计力学中处理 Ising 模型等问题的经典方法。

### 4.5 Score Matching

为了最小化 KL 散度，需要计算梯度 $\nabla_\theta \log P_\theta(x)$，但直接计算往往涉及难以处理的配分函数。

**Score Matching** 的思路：绕开对配分函数的求导，直接对 **score function**（即 $\nabla_x \log P_\theta(x)$）进行参数化建模。

> 这也解释了为什么 **Diffusion Model** 属于 Approximate Density 而非 Tractable Density —— 它通过 score matching 间接学习分布，并非直接计算精确的密度函数。

---

## 5. Vibe-Coding 实践

本周编程项目：**Hidden Markov Model (HMM)**，使用 **Mean-Field Variational Inference** 从观测数据中推断隐状态。

### 5.1 问题设定

考虑一个具有 2 个隐状态（Stable / Unstable）的 HMM：

- **转移矩阵：** 90% 概率停留在当前状态，10% 概率切换

$$A = \begin{pmatrix} 0.9 & 0.1 \\ 0.1 & 0.9 \end{pmatrix}$$

- **发射分布 (Gaussian)：**
  - Stable（状态 0）：$\mathcal{N}(0, 1)$
  - Unstable（状态 1）：$\mathcal{N}(0, 25)$（标准差 = 5）

- **任务：** 给定 $T=200$ 步的观测序列 $X$，推断隐状态序列 $Z$

### 5.2 均场变分推断应用于 HMM

将均场近似（第 4.4 节）应用于 HMM 的链式结构：

$$q(Z) = \prod_{t=1}^{T} q_t(z_t)$$

#### 变分自由能

$$F = -\sum_{k} q_1(k) \log \pi_k - \sum_{t} \sum_{k} q_t(k) \log P(x_t \mid z_t=k) - \sum_{t=2}^{T} \sum_{j,k} q_{t-1}(j)\, q_t(k) \log A_{jk} + \sum_{t} \sum_{k} q_t(k) \log q_t(k)$$

> 注：实践中若初始分布为均匀分布，$\log \pi_k$ 为常数项，可省略不影响优化。

#### 坐标上升更新规则

对于时刻 $t$ 的信念 $q_t(k)$，最优更新为：

$$\log q_t(k) \propto \underbrace{\log P(x_t \mid z_t=k)}_{\text{局部观测证据}} + \underbrace{\sum_{j} q_{t-1}(j) \log A_{jk}}_{\text{来自过去的消息}} + \underbrace{\sum_{l} q_{t+1}(l) \log A_{kl}}_{\text{来自未来的消息}}$$

然后通过 **softmax** 归一化得到概率分布。

**边界条件：**
- 当 $t=1$ 时，无"来自过去的消息"项，可用初始分布 $\log \pi_k$ 替代
- 当 $t=T$ 时，无"来自未来的消息"项，仅保留前两项

> **物理类比：** 在一维链中，每个"粒子" $z_t$ 受到三个力的作用：局部观测（类似外磁场）、左邻居的影响、右邻居的影响。均场近似让每个粒子在邻居的平均场中独立更新。

### 5.3 收敛性保证

坐标上升的每一步都保证**变分自由能单调递减**（即 ELBO 单调递增），最终收敛到局部最优。

### 5.4 代码结构总结

| 阶段 | 内容 | 对应理论 |
|---|---|---|
| Phase 1 | 模拟 HMM 数据（生成 $Z$ 和 $X$） | 第 3.4 节 HMM 联合分布 |
| Phase 2 | 定义变分自由能函数 | 第 4.3 节 ELBO / Free Energy |
| Phase 3 | 均场坐标上升迭代更新信念 | 第 4.4 节 Mean-Field + 5.2 更新规则 |
| Phase 4 | 可视化收敛曲线与信念动画 | 验证 5.3 单调递减性 |

