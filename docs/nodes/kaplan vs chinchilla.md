---
date: 2024-01-20
title: Kaplan Scaling Law vs Chinchilla Optimal
tags:
  - LLM
  - Pretrain
categories:
  - LLM
---

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
```

```python
def LN(n):
    return (8.8*(10**13)/n)**0.076

def LD(d):
    return (5.4*(10**13)/d)**0.095

def ld(l):
    return 5.4*(10**13)/(l**(1/0.095))
```

GPT3的参数量为175B，预测的loss值为`LN(175*10**9)`，所需token数量为`ld(LN(175*10**9))`。