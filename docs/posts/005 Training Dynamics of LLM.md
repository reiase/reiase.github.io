---
title: 从Training Dynamics到Outlier
date: 2025-02-15
tags:
  - LLM
  - Training
  - Training Dynamics
categories:
  - LLM
  - Training
  - Training Dynamics
slug: training_dynamics
draft: true
description: >-
  通从Training Dynamics到Outlier
---

Training Dynamics是一个未被严格定义的词，泛指模型训练过程中观测到的各种现象、变化和行为规律。我们可以从loss、泛化loss、梯度大小以及等等表现来观察模型内部的演变机制，并总结出类似涌现现象（Emergency）、Scaling Law、Double Decent和Gradient Pathologies等现象。

其中，权重与激活值的Dynamics会影响到数值表达范围，进而影响硬件运算时的精度以及量化误差。因此本文关注权重与激活值的Dynamics，并讨论其对低精度训练的影响。

## 权重、激活与梯度直观Dynamics变化

这里先给出权重与梯度的直观Dynamics变化，帮助直观理解训练过程。下图取自某开源仓库[^llm_facts]，展示了权重数值的直方分布随训练进行的变化情况：

![](./imgs/tdynamocs/weight_histograms_all_together_animation_mlp_h24h_800m.gif)

可以发现，各个block的FFN部分权重从随机初始化的高斯分布，开始时较为稳定；在2000 step左右开始剧烈变化；随后整体分布再次稳定下来。权重整体保留了高斯分布，但是存在一些不是非常大的outlier。

接下来再看一下激活值的分布变化，在训练开始后，残差激活值迅速从高斯分布转变为逻辑分布（Logistic Distribution），并且出现较大的outlier：

![](./imgs/tdynamocs/acc.gif)
这种激活上的outlier会对模型量化过程产生极大的影响，因此诸如AWQ等量化方法会重点关注激活中的outlier情况，以保证模型推理时的精度。

梯度分布的变化趋势与权重类似，训练过程也未出现较大的outlier，说明梯度本身也具备较好的稳定性，存在低精度计算和存储的可能性。

![](./imgs/tdynamocs/grad.gif)

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

其中$a_i$表示的是每个token的能量，$b_j$表示权重矩阵对每个特征通道的固有缩放。两者张成的能量矩阵$\mathbf{E}$表示了输入到矩阵乘法环节的总能量分布，而相关性矩阵$\mathbf{R}$则表示了能量传输效率与信息选择。

通常来说，能量矩阵$\mathbf{E}$具有较高的动态范围，而相关性矩阵$\mathbf{R}$需要较高的计算精度。

### RMSNorm的动态性分析

RMSNorm层是当前主流大模型所使用归一化层，相较传统的LayerNorm减少了对均值的计算，进而是的归一化层的计算更加高效。RMSNorm定义如下：
$$\bar{x_i} = \frac{x_i}{RMS(x)}w_i, RMS(x) = \sqrt{\frac{1}{H}\sum_{i=0}^H} x_i^2$$
在大模型中，RMSNorm先对每个token的表达进行归一化，再对每个channel进行缩放。

## 动态性分析
### RMSNorm动态性分析
我们首先以直方图的方式，直观观察下RMSNorm层中的数值动态性：
![[Pasted image 20250304132934.png]]
左一图是输入到RMSNorm的原始激活值的数值分布直方图，可以看到已经出现了比较大的outlier。左二是对激活值在token维度上norm值的直方分布，左三是channel维度上的norm值的直方分布。右一是该层的权重值的直方分布。从上述直方图上可以看到，在token和chanel维度上都出现了非常高的能量分布，接下来我们关注下归一化与缩放过程分别发生了什么。归一化后的激活值直方分布如下图所示：
![[Pasted image 20250304133546.png]]
可以明显看到，经过归一化后，激活值的整体分布得到了优化，outlier得到了一直，并且每个token和每个channel的能量都得到了压缩。接下来看看缩放过程发生了什么：

![[Pasted image 20250304133759.png]]

缩放后又再次放大了outlier。归一化和缩放过程中的压缩和缩放过程可以通过下图更加直观的来展示：

![[Pasted image 20250304134950.png]]
左一展示了归一化过程中，激活值input在token维度上的能量被压缩到归一化值output1中后token能量的映射关系，每个点的横坐标表示了激活值的每个token输入能量的数值，纵坐标表示归一化后每个token能量的数值。这个图直观的展示了规划化过程中数值变化的过程。左二展示了缩放过程中数值变化的过程，横轴是归一化数值output1，纵轴表示缩放后的数值。若将两张图合并，可以更加直观的得到整个RMSNorm对数值的压缩与缩放过程（右一）。接下来我们展示模型实际运算中，一些RMSNorm层的实际表现：
![[Pasted image 20250304135249.png]]
![[Pasted image 20250304135321.png]]
![[Pasted image 20250304135346.png]]
![[Pasted image 20250304135419.png]]
从上述表现中我们可以看到，大部分RMSNorm层能够很好的对激活层的输入能量进行压缩，但另一方面也存在一些RMSNorm层压缩失败了：
![[Pasted image 20250304135652.png]]


[^llm_facts]: https://github.com/BerenMillidge/LM_facts_across_training
[^facts_training]: [Basic facts about language models during training](https://www.alignmentforum.org/posts/2JJtxitp6nqu6ffak/basic-facts-about-language-models-during-training-1)