---
title: Minorize-Maximization
date: 2025-02-11
slug: minorize-maximization
tags:
  - RL
  - Optimization
---

Minorize-Maximization（MM）是一种迭代优化方法。当目标函数比较复杂难以优化时，通过引入一个近似函数来迭代求解。具体来说，对于问题

$$
    x^* = \arg\max_x f(x)
$$

可以寻找替代函数$g(x)$，并使其满足：

1. $g(x)$是$f(x)$的下界：

$$
    f(x) \ge g(x), \forall x
$$

2. 在$x_0$处，下界是紧的：

$$
    f(x_0) = g(x_0)
$$

MM 方法的优化步骤如下：

- 步骤1:在$x_t$处，构造下界函数 $g(x)$，使其满足：

$$
    f(x) \ge g(x), 且 f(x_t) = g(x_t)
$$

- 步骤2: 更新$x$:

$$
    x_{t+1} = \arg\max_x g(x)
$$

- 步骤3：重复步骤1，重新构造下界函数，迭代优化至收敛；