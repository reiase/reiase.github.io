---
title: AdamW
date: 2024-01-30
tags:
  - Training
---

LLM训练中通常使用AdamW优化器，配合grad clip参数1.0与weight decay参数0.1使用。AdamW优化器的伪代码入下图所示：
![[Pasted image 20240130090407.png]]
## weight decay
下图简单绘制了权重因weight decay随训练衰减的情况，$\theta_{t+n} = \gamma^n\theta_t$，当$\gamma$分别取0.01与0.1时，训练100个step后权重衰减如下：
![[Pasted image 20240130090343.png]]
当$\gamma=0.1$时，40个step后权重衰减为原来的1%，基本可以忽略不计。因此在实际LLM训练过程中模型的遗忘作用非常强。
### 稳态分析
若将AdamW优化器看成一个动力学系统，并假设模型收敛时权重更新近似为0，即
$$
	\theta_{t} = \theta_{t-1}
$$
带入AdamW的最后更新公式
$$
	\theta_t = \theta_{t-1} - \lambda \theta_{t-1} - \gamma \frac{m_t}{\sqrt{v_t}+\epsilon}
$$
有：
$$
	\lambda \theta_{t-1} = - \gamma\frac{m_t}{\sqrt{v_t}+\epsilon}
$$
即系统达到稳态时，权重正比于短时间内梯度的滑动平均。
### 梯度分析
若系统达到稳态，$\theta_t$ 近似不变，则可以认为模型在该权重处的梯度近似不变，此时$|m_t| \simeq \sqrt{v_t}$，则有
$$
		\lambda \theta_{t-1} = - \gamma\frac{m_t / |m_t| }{\sqrt{v_t}/|m_t|+\epsilon/|m_t|} = - \gamma\frac{sign(m_t)}{1+\epsilon/|m_t|}
$$
$$
	\theta_t = - \frac{\gamma}{\lambda} \times \frac{sign(m_t)}{1+\epsilon/|m_t|}
$$
考虑到系统达到稳态时$m_t = g_t$，则：
$$
	\theta_t = - \frac{\gamma}{\lambda} \times \frac{sign(g_t)}{1+\epsilon/|g_t|} = - \frac{\gamma}{\lambda} \times \frac{sign(\nabla L(\theta_t))}{1+\epsilon /  |\nabla L(\theta_t)|} 
$$
上述公式为一个一阶微分方程，系统稳态对应该一阶微分方程的解。