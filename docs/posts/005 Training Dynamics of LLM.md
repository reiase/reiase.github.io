---
title: 从Training Dynamics到Outlier
date: 2025-02-20
tags:
  - LLM
  - Training
  - Training Dynamics
categories:
  - LLM
  - Training
  - Training Dynamics
slug: training_dynamics
draft: false
description: >-
  通从Training Dynamics到Outlier
---

Training Dynamics是一个未被严格定义的词，泛指模型训练过程中观测到的各种现象、变化和行为规律。我们可以从loss、泛化loss、梯度大小以及等等表现来观察模型内部的演变机制，并总结出类似涌现现象（Emergency）、Scaling Law、Double Decent和Gradient Pathologies等现象。

其中，权重与激活值的Dynamics会影响到数值表达范围，进而影响硬件运算时的精度以及量化误差。因此本文关注权重与激活值的Dynamics，并讨论其对低精度训练的影响。

## 权重、激活与梯度直观Dynamics变化

这里先给出权重与梯度的直观Dynamics变化，帮助直观理解训练过程。下图取自某开源仓库[^llm_facts]，展示了权重数值的直方分布随训练进行的变化情况：

<figure markdown="span">
![](./imgs/tdynamocs/weight_histograms_all_together_animation_mlp_h24h_800m.gif){width="500"}
</figure>

可以发现，各个block的FFN部分权重从随机初始化的高斯分布，开始时较为稳定；在2000 step左右开始剧烈变化；随后整体分布再次稳定下来。权重整体保留了高斯分布，但是存在一些不是非常大的outlier。

接下来再看一下激活值的分布变化，在训练开始后，残差激活值迅速从高斯分布转变为逻辑分布（Logistic Distribution），并且出现较大的outlier：

<figure markdown="span">
![](./imgs/tdynamics/acc.gif){width="500"}
</figure>

这种激活上的outlier会对模型量化过程产生极大的影响，因此诸如AWQ等量化方法会重点关注激活中的outlier情况，以保证模型推理时的精度。

梯度分布的变化趋势与权重类似，训练过程也未出现较大的outlier，说明梯度本身也具备较好的稳定性，存在低精度计算和存储的可能性。

<figure markdown="span">
![](./imgs/tdynamics/grad.gif){width="500"}
</figure>

<!-- more -->

## 分析方法
此处我们主要分析FFN对数值动态性的影响，Attention层本身只是对token进行加权平均，并不会本质性的改变数值分布。
### 矩阵乘法的动态性分析

矩阵运算可以分解成行列向量的内积运算：

$$
C = A \times B =
\begin{bmatrix}
a_0 \\
a_1 \\
\vdots
\end{bmatrix}
\times
\begin{bmatrix}
b_0 ;
b_1;
\dots
\end{bmatrix} =
\begin{bmatrix}
a_0 b_0 & a_0 b_1 & \dots \\
a_1 b_0 & a_1 b_1 & \dots \\
\vdots & \vdots & \ddots \\
\end{bmatrix}
= [c_{ij}] = [a_i \cdot b_j]
$$

进一步有：

$$
c_{ij} = \|a_i\| \|b_j\| \left [ \frac{a_i \cdot b_j}{\|a_i\| \|b_j\|} \right ] = \|a_i\| \|b_j\|\cos{\theta_{ij}}
$$

其中 $\theta_{ij}$ 为两个向量的夹角，若两者均值为零，则 $\cos{\theta_{ij}}$ 与两向量线性相关性 $\rho$ 相等。根据上述分析，可以将矩阵乘法分解成两部分：

$$
C = \underbrace{[\|a_i\| \|b_j\|]}_{能量矩阵} \odot \overbrace{[\cos \theta_{i,j}]}^{相关性矩阵} = \mathbf{E} \odot \mathbf{R}
$$

其中$a_i$表示的是每个token的能量，$b_j$表示权重矩阵对每个特征通道的固有缩放。两者张成的能量矩阵$\mathbf{E}$表示了输入到矩阵乘法环节的总能量分布，而相关性矩阵$\mathbf{R}$则表示了能量传输效率与信息选择。在以前的文章中我们分了过[向量相关性与浮点数量化误差的关系]((//2025/02/12/fp8_training/))。

通常来说，能量矩阵$\mathbf{E}$具有较高的动态范围，而相关性矩阵$\mathbf{R}$需要较高的计算精度。

### RMSNorm的动态性分析

RMSNorm层是当前主流大模型所使用归一化层，相较传统的LayerNorm减少了对均值的计算，进而是的归一化层的计算更加高效。RMSNorm定义如下：
$$\bar{x_i} = \frac{x_i}{RMS(x)}w_i, RMS(x) = \sqrt{\frac{1}{H}\sum_{i=0}^H} x_i^2$$
在大模型中，RMSNorm先对每个token的表达进行归一化，再对每个channel进行缩放。

## 动态性分析
### RMSNorm动态性分析
我们首先以直方图的方式，直观观察下RMSNorm层中的数值动态性，实验使用QWen2.5-1.5B Instruct模型：

![alt text](imgs/tdynamics/rmsnorm_dynamics.png)

三张图分别展示了RMSNorm层的归一化、缩放两个阶段，以及整体的数值变化。以左一为例，横轴表示输入的数值，纵轴表示归一化后的数值，图中的每个点表示归一化过程对一个数值的变换。比如某个点的座标为（200，20），则表示这个点在归一化前的数值是200，归一化后被压缩到了20。每张图上部的直方图展示的是归一化前的数值分布，而右侧的直方图展示的是归一化后的数值分布。

通过上述三张图可以直观的看到归一化与缩放两个阶段对数值范围的影响。归一化能够较好的抑制outlier，而缩放过程会放大某些重要的channel，但也会导致部分数值变得很大，反而加剧了outlier的情况。这两个阶段整体作用还是对outlier起到抑制作用，但又避免重要信息不够突出。以下是随机选择的一些RMSNorm层对动态范围压缩的情况：

??? Question "实验分析结论"
    
    - RMSNorm主要作用是对outlier进行动态范围的压缩；
    - 这种压缩作用大多数情况能成功，但部分layer只能提供很少的压缩比；

![alt text](imgs/tdynamics/image-4.png)
![alt text](imgs/tdynamics/image.png)
![alt text](imgs/tdynamics/image-1.png)
![alt text](imgs/tdynamics/image-2.png)
![alt text](imgs/tdynamics/image-3.png)

可以发现，不同的decoder layer的RMSNorm层的表现会存在差异，开始的decoder与结束的decoder中RMSNorm的数值压缩作用并不明显，但中间的decoder中RMSNorm提供了非常强的动态范围压缩能力。

### FFN动态性分析

FFN的主要计算如下：

$$
y = \left[ \text{Swish}(xW_1)\odot xW_2 \right] W_3
$$

其中三次线性变换$W_1$被称为gate projection，$W_2$被称为up projection，而$W_3$被称为down projection。我们分别分析这三次projection的能量矩阵与相关性矩阵：

![alt text](imgs/tdynamics/ffn_td.png)

上图第一行展示了输入激活值与权重的直方分布，第二行分别是激活值、能量矩阵与相关性矩阵的直方分布，这里的能量矩阵与相关性矩阵分别对应前文给出的分析方法 $C = \mathbf{E} \odot \mathbf{R}$。

最后一行分别用散点图的方式展示了能量矩阵、相关性矩阵和激活输出矩阵三者之间的数值相关性。从左一图中可以看出相关性与能量之前没有相关性，左二图中可以看到能量矩阵对输出的激活值没有直接影响，右一图说明相关性对激活值大小存在线性影响。接下来我们随机抽取一些decoder layer的ffn，观测其中矩阵乘法的dynamics：

![alt text](imgs/tdynamics/mlp/image.png)
![alt text](imgs/tdynamics/mlp/image-1.png)
![alt text](imgs/tdynamics/mlp/image-2.png)
![alt text](imgs/tdynamics/mlp/image-3.png)
![alt text](imgs/tdynamics/mlp/image-4.png)
![alt text](imgs/tdynamics/mlp/image-5.png)
![alt text](imgs/tdynamics/mlp/image-6.png)
![alt text](imgs/tdynamics/mlp/image-7.png)
![alt text](imgs/tdynamics/mlp/image-8.png)
![alt text](imgs/tdynamics/mlp/image-9.png)
![alt text](imgs/tdynamics/mlp/image-10.png)
![alt text](imgs/tdynamics/mlp/image-11.png)

[^llm_facts]: https://github.com/BerenMillidge/LM_facts_across_training
[^facts_training]: [Basic facts about language models during training](https://www.alignmentforum.org/posts/2JJtxitp6nqu6ffak/basic-facts-about-language-models-during-training-1)