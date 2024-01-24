---
date: 2024-01-21
title: 关于Scaling Law的非正式评论
categories:
  - LLM
---
## 大模型的良好泛化性

大模型被广泛关注的起点是OpenAI发布ChatGPT，凭借优秀的对话能力与In-Context Learning的能力，吸引了整个AI圈的关注。In-Context Learning是指预测时在上下文中给出足够的背景知识和任务描述，然后直接预测[^1]，比如：

- Zero-shot（没有示例）：{==8+9=?==}
- One-shot（一个示例）：5+5=10, {==8+9=?==}
- Few-shot（多个示例）：6+7=13,6+6=12,5+5=10, {==8+9=? ==}

Open AI在2020年的大模型Scaling Law论文中发现，若将模型迁移到新的数据集，新数据集上的测试loss与训练数据集上的测试loss存在一个相对恒定的offset。随着模型规模的增大，训练数据集和新数据集上的测试loss近似同步下降。这也就意味着可以通过持续增大模型大小来降低模型在所有数据集上的loss。

![[Pasted image 20240121235222.png]]

这种优秀的泛化性给出了一种全新的探索方向：相比探索更复杂的模型结构，探索更大的模型是否能够成为深度学习的全新路径。

## 大模型的可预测性

传统深度学习的黑盒炼丹是被广为诟病的问题。虽然不乏丹术已至臻境的顶级丹师，但业务方和资本方是万万不敢在单次训练超一个月，耗资超百万的LLM训练交给这些丹师来负责的。如何预测一次训练能达到的精度，以及如何分配这种超百万级别的训练成本投入，就变成了训练LLM的必要条件。这方面比较重要的工作有两篇：一篇是OpenAI提出的Kaplan Scaling Law[^2]，主要研究了模型loss和几个主要参数：模型参数规模N、训练数据规模D以及算力投入C之间的关系；另一篇是DeepMind提出的Chinchilla Optimal[^3]，主要研究计算资源限定下，算力的分配问题，比如每个参数对应多少token。
### Kaplan Scaling Law
Kaplan Scaling Law发表于2020年，同年OpenAI也正式发表了GPT3。因此从某种程度上来说，Kaplan Scaling Law即是对GPT3模型训练的一个总结，也是OpenAI坚定投入大模型和大算力的信心所在。先简单看下Kaplan Scaling Law的几个主要结论：

1. 限定模型参数规模N，给足够大的数据集训练到收敛：
	   $L(N) = \left (\frac{N_c}{N}\right ) ^{\alpha_N}; \alpha_N \simeq 0.076, N_c \simeq 8.8 \times 10^{13}$
1. 限定数据规模D，通过early stopping 训练大模型：
	   $L(D) = \left ( \frac{D_c}{D} \right)^{\alpha_D};\alpha_D \simeq 0.095, D_c \simeq 5.4 \times 10^{13}$
1. 限定计算资源，给足够大的数据集、最优模型规模与足够小的batch size：
	$L(C_{min})=\left ( \frac{C_c^{min}}{C}\right )^{\alpha_C^{min}}; \alpha_C^{min} \simeq 0.050, C_C^{min} \simeq 3.1 \times 10^8\  (PFdays)$ 

直观一点就是说影响loss的主要因素是N、D和C。随着三者的指数增长，loss线性下降：
![[Pasted image 20240123203148.png]]
另一方面模型结构超参、学习器的超参没那么重要，整体对loss影响不大：
![[Pasted image 20240123203610.png]]
有些参数在一到两个尺度内变化，而对最终的loss影响在2%以内。而学习率只要不过于小，或者衰减过于迅速，对最终loss的影响并不大。
![[Pasted image 20240124081651.png]]
Kaplan Scaling Law最为重大的意义在于两点：
- 给出了影响模型效果的主要因素模型参数规模N和训练数据规模D，并认为其他超参是次要因素，影响不大，因此也就避免了LLM上继续黑盒炼丹；
- 给出了对Loss的预测方法，只要指数增长N、D和算力C，即可获得loss的线性改进；
根据Kaplan Scaling Law，GPT3的175B参数需要300B token训练，约每个参数1.7个token（具体参考[[kaplan vs chinchilla]]）。

### Chinchila Scaling Law
Chinchila Scaling Law由DeepMind发表于2022年，同年Google发表了PaLM（Language Modeling with Pathways[^4]）。

[^1]: 2020, OpenAI, Language Models are Few-Shot Learners
[^2]: 2020, OpenAI, Scaling Laws for Neural Language Models
[^3]: 2022, DeepMind, Training Compute-Optimal Large Language Models
[^4]: 2022, DeepMind, PaLM: Scaling Language Modeling with Pathways

