---
title: Importance Sampling
date: 2025-02-11
slug: importance_sampling
tags:
  - RL
---

在计算期望的时候，需要一个概率分布$p(x)$:

$$
\mathbb{E}_p[f(x)] = \int_x f(x)p(x) dx
$$

但是任意分布$p(x)$通常难以从随机数发生器来生成，此时需要借助一个更加容易获得的辅助分布$q(x)$来解决问题：

$$
\mathbb{E}_p[f(x)] = \int_x f(x)p(x) dx \\
 = \int_x f(x)\frac{p(x)q(x)}{q(x)} dx \\
 = \int_x \left [f(x)\frac{p(x)}{q(x)} \right ] q(x) dx
$$

其中，$\frac{p(x)}{q(x)}$为重要性权重。这样就可以通过分布$q(x)$进行采样了。