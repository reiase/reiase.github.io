---
title: Probing分布式探针开发随笔（一）：背景与设计理念
date: 2025-03-26
tags:
  - LLM
  - Training
  - Debug
  - Profiling
categories:
  - LLM
  - Training
slug: dist_probe_1
description: >-
  探索Probing分布式探针系统如何解决千卡规模LLM训练中的性能瓶颈与故障问题。这个轻量级工具通过动态注入实现无侵入监控，支持CUDA/GPU性能分析、分布式调试、内存优化和通信分析，无需修改代码或重启应用。本文详细介绍其设计原理、SQL查询接口和在PyTorch训练环境中的应用，帮助工程师有效应对大规模分布式训练的挑战。
---

## 分布式训练系统的泥潭

在过去半年多的时间里，我一直在支持千卡规模的LLM分布式训练。坦白讲，千卡训练的过程并不愉快，尤其是在性能调优和故障排查方面。在这个过程中，我们遇到了一系列棘手的问题：训练无法正常启动、通信库突然hang住、节点性能不及预期、训练速度不稳定等等。这些问题不仅严重影响了训练效率，还大幅增加了调试的复杂度，导致我们不得不花费大量时间和精力在性能调优和故障排查上。

有人可能会说，千卡（乃至万卡）规模的稳定性问题在大厂内部已经解决得相当好了。然而，那些耗费无数人力堆砌出来的系统，往往只是在这些大厂已有的复杂基础设施上打补丁，解决眼前可见的问题，而且很多时候仅仅是在处理问题的表象。大规模分布式异构训练真正需要的是类似Hadoop、Spark、Kubernetes或TensorFlow这样具有前瞻性的系统设计，能够解决问题的本质，并提供解决问题的框架，而不仅仅是一些堆砌在特定基础设施上、不具备任何迁移性的"补丁"。我们需要一种更加系统化、可扩展的方法来应对这些挑战。

## Probing——分布式探针系统的原型探索

在解决问题的过程中，我一直思索自己到底需要什么。我需要一种能够在任何时刻动态启用，无需预先部署或插桩，在生产任务中以极低性能开销持续运行，实现实时监控与故障追溯的诊断工具。我需要一种不仅支持单机诊断，还能无缝覆盖分布式训练环境，无论集群规模如何，都能确保数据采集与故障分析的一致性的诊断工具。我需要一种能够从硬件层面的诊断数据、芯片互联状态，到框架、系统和模型各层数据的全面采集，构建完整的闭环监控系统的诊断工具。而现有的种种工具，要么需要侵入式的代码修改和预先部署，要么会严重影响性能，要么只能关注单机，无法覆盖分布式环境，要么只能关注单一维度，无法实现综合分析。

基于自己的需求，我开始尝试设计一种“探针”系统：

- 可以在任意时刻通过动态注入的方式启用，无需预先部署或插桩；
- 运行开销极低或者无开销，可以在生产任务中持续收集性能数据和故障数据；
- “寄生”在目标进程中，具有相同的内存地址空间与权限，进而实现观测和调试；
- 支持分布式，更好地覆盖大规模分布式训练环境；

这套探针系统大致用法如下：

```bash
$ probing <pid> inject # 注入探针
$ probing <pid> eval "print('Hello, Probing!')" # 在目标进程中执行代码
$ probing <pid> query "SHOW tables" # 查看可用数据
$ probing <pid> query "SELECT * FROM process.envs" # 查询进程环境变量
```

`probing`通过`query`命令提供SQL查询接口，并在这一接口下标准化了不同类型的数据，包括进程状态、硬件性能指标、网络状态、文件系统状态等，使用户无须单独学习每种数据的获取和分析方式。另一方面，SQL查询也提供和AI接入能力，用户可以借助AI生成查询与分析语句，实现自动化的性能分析与故障诊断。后续也会直接扩展SQL支持分布是查询，实现对整个集群的性能分析与故障诊断。

在接下来的一系列文章里，我将详细介绍Probing的设计与实现，包括探针机制、数据采集、分析方法等方面。希望这个探索能够为大规模分布式训练的性能分析与故障诊断提供一些启发。以下是接下来需要进行讨论的内容：

1. 如何实现探针的动态注入与运行时加载，如何规避C/C++常见的ABI兼容性问题；
2. 如何实现高频数据的采集和存储，如何实现数据的压缩和优化；
3. 如何避免跨节点时钟漂移带来的事件时间不一致问题；
