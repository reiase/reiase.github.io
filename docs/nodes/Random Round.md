---
title: Random Round
date: 2024-01-24
categories:
  - LLM
---
Random Round是指浮点数量化时，随机round到相邻的数值，而不是round到最近的数值。随机round虽然增加了单个数值的量化误差，但是能够避免整体的有偏量化误差。因此，在某些训练场景下能够避免有偏量化误差带来的收敛速度慢问题，具体可见Gopher论文[^1] 的Figure A7。
![[Pasted image 20240124133211.png]]

[^1]: 2021, DeepMind, Scaling Language Models: Methods, Analysis & Insights from Training Gopher