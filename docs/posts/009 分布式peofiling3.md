---
title: Probing分布式探针开发随笔（三）：分布式训练的Profiling
date: 2025-04-28
tags:
  - LLM
  - Training
  - Debug
  - Profiling
categories:
  - LLM
  - Training
slug: dist_probe_3
draft: true
description: >-
  本文深入解析Probing分布式探针系统的核心技术——动态探针注入机制。针对千卡规模LLM训练中的性能瓶颈与故障诊断难题，详解如何通过无侵入的`ptrace`注入实现运行时监控，解决CUDA/GPU性能分析、分布式通信优化、内存泄漏检测等痛点。涵盖ABI兼容性处理方案（静态链接/Zig编译）、跨平台打包策略（Python Wheel集成）及探针动态加载原理，为AI工程师提供开箱即用的大模型训练调试工具链设计指南。
---

在前两篇系列文章中，介绍了 Probing 的设计理念与核心技术，包括其应对 ABI 兼容性挑战的动态注入机制，以及基于 DataFusion 构建的可扩展查询引擎。并希望通过这些技术上的探索，开发出一个“完美”的工具，更好地解决大规模分布式异构训练系统中的性能与稳定性问题。

## 传统Profiler的困境

Profiler是性能优化工程师最为常用的工具，为了分析性能我们有形形色色的Profiler工具。大到Intel VTune和Nvidia Insight这种系列工具，有着完备的分析工具与可视化手段，很多问题都能一目了然；小到`perf top`这样简陋的调用栈采样工具，得边看边猜整个系统的行为。但是这些工具有一个共同的问题：他们都是单机工具，并不能很好的解决分布式系统中的性能问题。

PyTorch提供了`torch.profiler`，一个强大的内置性能分析工具：

```python
from torch.profiler import profile

with profile(
    activities=[
      torch.profiler.ProfilerActivity.CPU,
      torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    for step in range(steps):
      train_one_step()
      prof.step()
```

这个内置的Profiler提供了算子粒度的性能分析，并且能够追踪PyTorch中的算子执行与显存分配行为，并且可以通过Tensorboard对结果进行可视化。然而，到了大规模分布式训练场景，这个内置Profiler的局限性就显现出来了：

1. 性能开销：全量采集模式下会显著影响训练性能，导致Profiling结果不准确；
2. 数据爆炸：分布式场景中产生的数据会爆炸性增长，单卡上G的数据，千卡需要上T存储；
3. 缺乏协调：各个节点的profiling数据相互独立，难离关联分析，特别是引入模型并行之后；

## 思路转变：从Timeline到统计方法

#### 被困在Timeline分析中的性能优化工程师

单节点的性能分析中，Timeline技术备受追捧。原因无他：直观，每个阶段的执行，开销的资源与消耗的时间都可以在一个时间轴上精确展示出来。但是Timeline数据庞大，并且借助浏览器渲染时速度也欠佳。几个G的Timeline数据很快会让你的笔记本成为一个小火炉。

Timeline也存在明显的局限性：

- 一般只能分析单个节点单个Step：
  - 多节点多Step的数据，看不过来（虽然有些人在此会很倔强）；
  - 每个Step都有差异，导致难以给出结论（可以给定性结论，但结论复现存在难度）；
- 难以捕捉这个系统的随机性与不确定性：
  - 单个节点单个Step是确定的，但是一千个节点的同一个Step，充满了随机性；
  - Timeline无法刻画出整个系统性能层面的统计特性，比如耗时的99线；
- 忽略了负载不均很现象：
  - 在经典的Dense LLM中，因为模型并行会导致每个节点实际负载各不相同；
  - 在流行的MoE LLM中，专家路由也会导致计算负载不均衡问题；

上述这些问题难以在Timeline框架下解决：

在由上千节点与数千线缆构成的复杂计算集群中，节点与节点间互联必然存在随机性与不确定性，这正是分布式系统的核心挑战。因此，分布式系统的性能分析需要从单节点上精准timeline的个体样本方法转向能够描述随机性与整体特性的统计方法。

随着集群规模扩大，性能数据量也呈指数级增长。单节点的Profiling数据若直接扩展到分布式集群将给数据存储与查询带来巨大挑战。而结合集群对性能数据进行分布式采样，不仅能避免数据存储与查询瓶颈，还能提供统计层面更加直观合理的全局视角，帮助开发者更好地理解集群层面的性能特征。

此外，分布式系统中时钟同步始终是一个难题。依赖精准时间戳的timeline分析难以满足分布式系统的需求。如何在不依赖精准时间戳的情况下进行数据关联分析、识别性能异常节点(Stragglers)，也成为分布式系统性能分析的关键挑战。

被困在Timeline中的工程师只能裁剪问题到单节点，抓抓timeline，给出一些猜想，然后修改修改代码，祈祷优化在集群上管用。

#### Profiling中的统计思想

而来到分布式系统领域，性能问题的解决采用着完全不同的思路。类似opentelemetry这种分布式tracing工具成为了主导，会把一次API调用拆分成嵌套的Span，并追踪整个调用链路上的耗时与资源开销情况。这种分布式Tracing依赖于微服务调用的树状调用关系，并且需要庞大的数据系统来支撑。但分布式训练系统中并不存在树状调用关系，而是以集合通信库驱动的计算逻辑。并且千卡规模的训练系统产生的算子级别高频数据量对数据系统是个极大的挑战。

单节点的性能分析固然重要，但在动辄成百上千节点的分布式训练场景下，问题的根源往往隐藏在节点间的交互与差异中。Straggler（慢节点）、通信瓶颈、负载不均等问题，是分布式训练效率提升的主要障碍，而传统的单点 Profiling 工具对此往往束手无策。

本文将聚焦 Probing **如何解决分布式训练场景下的 Profiling 难题**。我们将探讨 Probing 如何在架构层面支持分布式数据的采集、关联与分析，特别是如何利用其核心的 SQL 查询引擎，借助统计手段从全局视角洞察分布式系统的行为，定位跨节点的性能瓶颈。


## Probing的分布式Profiling架构

Probing 通过其独特的架构来应对这些挑战，其核心思想是：**节点本地化数据采集 + 分布式查询分析**。以下是一个简单的示意图，用于说明理想情况下probing如何工作：

```mermaid
---
title: Probing 分布式 Profiling 架构
---
graph TD
    subgraph "控制平面 (用户)"
        UI[Web UI]
        CLI[命令行]
        API[SQL查询+HTTP协议]
        UI & CLI --> API
    end

    subgraph "分布式训练集群"
        direction LR
        subgraph "Node 1 (Rank 0)"
            P1[训练进程 Rank 0]
            PR1[Probe]
            H1[采集Hooks e.g., PyTorch]
            P1 --> H1 -- 本地数据 --> PR1
        end
        subgraph "Node 2 (Rank 1)"
            P2[训练进程 Rank 1]
            PR2[Probe]
            H2[采集Hooks e.g., PyTorch]
            P2 --> H2 -- 本地数据 --> PR2
        end
        subgraph "Node N (Rank N-1)"
            PN[训练进程 Rank N-1]
            PRN[Probe]
            HN[采集Hooks e.g., PyTorch]
            PN --> HN -- 本地数据 --> PRN
        end
    end

    API -- SQL查询 --> PR1;
    PR1 -- 分布式查询协调 --> PR2;
    PR1 -- 分布式查询协调 --> PRN;
    PR2 -- 本地查询/聚合 --> PR2;
    PRN -- 本地查询/聚合 --> PRN;
    PR2 -- 部分结果 --> PR1;
    PRN -- 部分结果 --> PR1;
    PR1 -- 最终聚合 --> API;

    style P1 fill:#f9f,stroke:#333
    style P2 fill:#f9f,stroke:#333
    style PN fill:#f9f,stroke:#333
    style PR1 fill:#bfb,stroke:#333
    style PR2 fill:#bfb,stroke:#333
    style PRN fill:#bfb,stroke:#333
    style H1 fill:#ccf,stroke:#333
    style H2 fill:#ccf,stroke:#333
    style HN fill:#ccf,stroke:#333
```

不同于传统的timeline方法，Probing采用基于采样的性能分析方法，具有两大优势：

- 通过调整采样率精确控制对系统的性能影响
- 借助设计良好的采样器将数据采集高效扩展到分布式系统

采样方法允许我们在每个节点上独立控制采样频率以限制性能开销，同时利用分布式系统的规模优势快速获取有代表性的性能数据。

#### 基于PyTorch钩子的采样

PyTorch提供了多种钩子（Hooks）机制，使得我们能够在训练过程中"无侵入"地采集性能数据。

```python
from torch.optim.optimizer import register_optimizer_step_post_hook

register_optimizer_step_post_hook(optimizer_step_post_hook)
```

`register_optimizer_step_post_hook` 帮我们向torch注册一个钩子函数，在每个Optimzier完成`step()`掉用后执行。，这个时机极为关键：

1. 模型已完成构建，可获取完整模型定义
2. 前向传播、反向传播与优化器都已完成预热

此时，我们能够通过Python的垃圾回收(GC)机制与反射能力识别进程中的模型结构：

```python
def get_toplevel_module():
    import gc

    import torch

    objs = [obj for obj in gc.get_objects() if isinstance(obj, torch.nn.Module)]
    is_child = set()
    for obj in objs:
        for child in obj.children():
            is_child.add(id(child))
    return [obj for obj in objs if id(obj) not in is_child]
```

通过`gc`模块我们可以获得当前进程中的全部Python对象列表，再通过反射调用`isinstance(obj, torch.nn.Module)`找出全部`torch.nn.Module`对象。最后再根据module之间的父子关系来发现顶层Module。

获取顶层Module后，我们可以注册完整的前向/反向传播钩子链：

1. Module.register_forward_pre_hook - 前向传播开始前
2. Module.register_forward_hook - 前向传播完成后
3. Module.register_full_backward_pre_hook - 反向传播开始前
4. Module.register_full_backward_hook - 反向传播完成后
5. Optimizer.register_step_pre_hook - 优化器步骤开始前
6. Optimizer.register_step_post_hook - 优化器步骤完成后

这些钩子构成了训练过程中的完整监控链，允许我们精确测量模型各组件的执行性能。

#### 结构化采样

考虑到大型模型包含大量嵌套子模块，对每个模块都执行计时操作会带来显著性能开销。我们设计了一种智能的结构化采样方法：

1. span分解：将模型执行分解为一系列span，每个module的前向和反向传播分别构成独立span
2. 层次化排序：按照嵌套关系对span进行排序
    - 粗粒度span（如整个模型的前向传播）排序靠前
    - 细粒度span（如单个卷积层的操作）排序靠后
3. 自适应采样：从粗到细逐步采样
    - 命中采样时，记录当前span计时，并移至下一个span
    - 未命中采样时，跳过计算以减少开销

这种结构化采样确保每个训练步骤只对一个特定粒度的span进行采样，使模型性能分析由粗到细逐步进行，在控制开销的同时提供全面性能视图。

#### CUDA Event精确计时

GPU上异步执行的计时通常通过CUDA Event来实现。CUDA Event能保证在CUDA Stream上的执行顺序，并且是测量GPU操作时间的最准确方式。一个CUDA Event的生命周期包括以下几个阶段：

1. 创建(Create)：通过torch.cuda.Event()或CUDA原生API创建Event对象
2. 记录(Record)：通过event.record()将Event标记到特定CUDA Stream的当前位置
3. 同步(Synchronize)：通过event.synchronize()等待Event标记的操作完成
4. 查询(Query)：通过event.query()非阻塞地检查Event是否完成
5. 计时(Elapsed Time)：通过start_event.elapsed_time(end_event)计算两个Event之间的时间差

在实际应用中，同步(Synchronize)操作会导致GPU等待并强制Stream清空，可能显著影响性能。为解决这一问题，我们采用延迟计时(Delayed Timing)策略，将时间读取推迟到优化器执行完成后进行。这种方法有效降低了计时操作对训练性能的干扰，特别适合分布式训练环境。

## 基于统计的性能/故障分析方法

在大规模分布式训练环境中，我们面临的不仅是如何采集数据，更重要的是如何有效利用这些数据发现并解决问题。Probing采用统计分析方法，将分散在各节点的性能数据转化为可操作的洞察。

#### 分布式训练中的常见性能问题
在实践中，分布式训练的性能问题通常表现为以下几种典型模式：

1. 慢节点(Straggler)问题：个别节点显著慢于集群平均水平，拖慢整体训练进度
2. 负载不均衡：计算或内存负载在节点间分布不均，导致资源利用率低下
3. 通信瓶颈：节点间数据交换速度不足，制约训练效率提升
4. 异常波动：性能指标在时间维度上出现突发性异常
5. 集群分层：性能根据硬件配置或网络拓扑自然分层，形成性能梯队利用统计数据定位问题

#### 节点性能差异分析

通过简单SQL查询，我们可以快速识别集群中的异常节点：

```SQL
-- 查找前向传播耗时异常的节点
SELECT 
    rank, 
    AVG(duration_ms) as avg_forward_time,
    COUNT(*) as sample_count,
    (AVG(duration_ms) - 
     (SELECT AVG(duration_ms) FROM torch_traces WHERE operation='forward')) 
     / (SELECT STDDEV(duration_ms) FROM torch_traces WHERE operation='forward') 
     as z_score
FROM python.torch_traces
WHERE operation = 'forward' AND step_id BETWEEN 100 AND 200
GROUP BY rank
HAVING z_score > 2.0  -- 标准差超过2倍的视为异常
ORDER BY avg_forward_time DESC;
```

这种查询允许我们立即发现性能显著偏离集群平均水平的节点，而无需手动检查每个节点的timeline。

#### 层次性能分布图

分布式训练中，模型的不同组件在不同节点上的性能表现极具研究价值。Probing通过层次性能分布图直观展示这种多维度性能数据，帮助工程师快速定位瓶颈。通过Probing可以采集如下格式的数据：

```
ts: 事件时间戳
node：节点名称
module：模块名称
stage：阶段名称，比如forward或者backward
mem_allocated: 已经分配的显存
mem_cached: 已经缓存的显存
duration：时间开销
```

通过对采集的结构化数据进行多维度聚合与可视化，我们可以构建如下分析图表：


```SQL
-- 分析每个模型层在不同节点上的性能分布
SELECT
    module,
    node,
    AVG(duration_ms) as avg_duration,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as median,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95,
    COUNT(*) as samples
FROM python.torch_traces
WHERE operation = 'forward' AND step_id BETWEEN 1000 AND 2000
GROUP BY module, node
ORDER BY module, avg_duration DESC;
```

这种查询能够生成深度学习模型中每个组件在集群不同节点上的性能热力图，通过这种热力图，我们可以立即观察到：

- 水平方向：同一节点上不同模型层的相对性能
- 垂直方向：同一模型层在不同节点上的性能差异
- 热点区域：特定节点-组件组合的性能异常

在一次实际分析中，我们通过层次性能分布图发现了某DNN模型中有趣的性能模式：

- 组件级差异：Attention层在所有节点上都比其他层耗时更长（水平模式）
- 节点级差异：特定的4个节点在处理卷积层时显著慢于其他节点（垂直模式）
- 交互效应：某些节点仅在处理特定类型的层时出现性能下降（局部热点）

#### 时间维度的性能演变分析

分布式训练的性能问题常常随时间动态变化。通过跟踪关键指标的时间序列，我们可以发现潜在问题：

```SQL
-- 分析训练过程中的性能趋势
SELECT 
    FLOOR(step_id / 50) * 50 as step_bucket,  -- 按50步为单位分桶
    AVG(duration_ms) as avg_duration,
    STDDEV(duration_ms) / AVG(duration_ms) as cv  -- 变异系数
FROM torch_traces
WHERE operation = 'forward' 
GROUP BY step_bucket
ORDER BY step_bucket;
```

通过这种分析，我们可以发现：

- 训练初期的预热效应
- 性能随时间的逐渐劣化
- 可能的内存泄漏或资源竞争问题
- 周期性波动（如系统GC或后台任务影响）

#### 分布式系统的层次化分析

在大型集群中，仅分析个体节点往往不够。Probing支持按网络拓扑、硬件型号等进行分组分析：


```SQL
 # 按网络拓扑分组分析通信性能
rack_perf = probe.sql("""
    SELECT 
        CASE 
            WHEN src_rank / 8 = dst_rank / 8 THEN 'same_node'
            WHEN src_rank / 32 = dst_rank / 32 THEN 'same_rack'
            ELSE 'cross_rack'
        END as topology,
        AVG(bytes_per_sec) as avg_bandwidth,
        COUNT(*) as sample_count
    FROM comm_events
    GROUP BY topology
""").fetchall()

for row in rack_perf:
    print(f"{row.topology}: {row.avg_bandwidth/1e9:.2f} GB/s ({row.sample_count} samples)")
# 输出:
# same_node: 87.32 GB/s (12453 samples)
# same_rack: 23.76 GB/s (8721 samples) 
# cross_rack: 11.89 GB/s (5432 samples)
```

这种分析揭示了网络拓扑对通信性能的影响，启发我们优化通信算法和数据分片策略以减少跨机架通信。

