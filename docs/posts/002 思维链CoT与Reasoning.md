---
title: 思维链CoT与Reasoning
date: 2025-02-11
tags:
  - LLM
  - Training
  - RL
categories:
  - LLM
  - Training
  - RL
slug: cot_reasoning
draft: true
---

## 1. 什么是Reasoning

Reasoning 在大模型领域一般被翻译成推理能力，是指通过一系列罗辑思维从已知信息推导出未知结论的过程。常见的推理类型有 演绎推理、归纳推理和归因推理等。借助严密的推理过程，Reasoning能够极大的提升模型的数学能力与解决问题的能力。

另一方面，Reasoning将大模型的能力从生成阶段对已有知识的归纳（这是为何LLM最早一批应用是总结论文），提升到输出逻辑严密的全新知识和创新性的解决方案，能够帮助人类探索全新的知识领域。

以一个例子来说明Reasoning如何工作：

!!! note "二+零+二+五=10，那么五二零是多少"
    
    GPT4o> 根据题意，"二"表示2，"零"表示0，"五"表示5。将这些数字组合成"五二零"，即520。

    o1> 这是根据拼音字母数来计算的：
    - 二（er）有 2 个字母
    - 零（ling）有 4 个字母
    - 五（wu）有 2 个字母
    
    所以“二+零+二+五”=2+4+2+2=10；
    “五二零”（wu er ling）=2+2+4=8。

未增强Reasoning能力的模型，只能理解问题的字面意思，而o1这种基于思维链的推理模型则能够一步一步分解问题，甚至进行多种思路的尝试和验证，并给出有深入理解的结果。这种Reasoning能力使得大模型看起来更像人类，被认为是通向通用人工智能（AGI）的关键。


