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

---

## 4. 模型学习：最小化 KL 散度

### 4.1 学习目标

有了模型之后，如何让模型拟合真实数据？

> **核心方法：** 最小化模型分布 $P_\theta$ 与数据分布 $P_{\text{data}}$ 之间的 **KL 散度 (Kullback-Leibler Divergence)**。

### 4.2 KL 散度

KL 散度衡量两个概率分布之间的"距离"（注意：不对称）：

- **Forward KL：** $D_{\text{KL}}(P_{\text{data}} \| P_\theta)$ — 倾向于让模型覆盖数据的所有模式（mode-covering）
- **Reverse KL：** $D_{\text{KL}}(P_\theta \| P_{\text{data}})$ — 倾向于让模型集中在数据的某些模式上（mode-seeking）

### 4.3 物理诠释

将 Boltzmann 分布代入 KL 散度，可以得到：

$$D_{\text{KL}} \longleftrightarrow \text{自由能 (Free Energy)}$$

即 **最小化 KL 散度等价于最小化自由能**，这与统计物理中的变分原理一致。

### 4.4 Score Matching

为了最小化 KL 散度，需要计算梯度 $\nabla_\theta \log P_\theta(x)$，但直接计算往往涉及难以处理的配分函数。

**Score Matching** 的思路：绕开对配分函数的求导，直接对 **score function**（即 $\nabla_x \log P_\theta(x)$）进行参数化建模。

> 这也解释了为什么 **Diffusion Model** 属于 Approximate Density 而非 Tractable Density —— 它通过 score matching 间接学习分布，并非直接计算精确的密度函数。

---

## 5. Vibe-Coding 实践

本周编程项目：**Hidden Markov Model (HMM)**

