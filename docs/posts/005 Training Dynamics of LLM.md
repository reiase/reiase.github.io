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
  通过向量内积的信噪比分析，探讨INT8量化在LLM训练中的可行性。对比INT8与FP8在不同向量长度和相关性条件下的量化精度，
  揭示分块量化和高精度累加如何提升INT8训练性能，以及INT8相比FP8的潜在优势。
---

Training Dynamics是一个未被严格定义的词，泛指模型训练过程中观测到的各种现象、变化和行为规律。我们可以从loss、泛化loss、梯度大小以及等等表现来观察模型内部的演变机制，并总结出类似涌现现象（Emergency）、Scaling Law、Double Decent和Gradient Pathologies等现象。

其中，权重与激活值的Dynamics会影响到数值表达范围，进而影响硬件运算时的精度以及量化误差。因此本文关注权重与激活值的Dynamics，并讨论其对低精度训练的影响。

## 权重与梯度的Training Dynamics

![](./imgs/tdynamocs/weight_histograms_all_together_animation_mlp_h24h_800m.gif)

[^llm_facts]: https://github.com/BerenMillidge/LM_facts_across_training
[^facts_training]: [Basic facts about language models during training](https://www.alignmentforum.org/posts/2JJtxitp6nqu6ffak/basic-facts-about-language-models-during-training-1)