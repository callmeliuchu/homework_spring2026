# HW1 生成式策略学习强化计划

这份计划只围绕你现在已经接触到的三种方法：

- `MSE Policy`
- `Flow Matching`
- `Diffusion Policy`

目标不是“再看懂一点”，而是做到下面四件事：

1. 你能自己讲清楚三者的区别。
2. 你能不看答案写出核心公式。
3. 你能看实验结果判断为什么好或坏。
4. 你能自己完成最小实现和最小消融。

---

## 你现在的位置

你已经掌握了这些关键点：

- `MSE` 是 `state -> action chunk` 的直接回归。
- `Flow Matching` 是从噪声到动作的连续路径建模，核心是速度场。
- `Diffusion` 是从噪声到动作的离散去噪过程，核心是前向加噪和反向采样。
- `noise / x0 / v` 是不同参数化，理论相关，但训练效果可能不同。
- `schedule` 会显著影响 diffusion 的表现。

还需要强化的地方：

- 把公式变成“闭眼能写”的能力。
- 把实验观察变成“能解释为什么”的能力。
- 把 diffusion 的训练和采样流程彻底讲清楚。

---

## 学习顺序

按这个顺序做，不要跳。

### 第 1 步：口头解释三种算法

你需要能不用看代码讲清楚：

- `MSE` 在学什么
- `Flow` 在学什么
- `Diffusion` 在学什么

完成标准：

- 你能用 3 句话分别解释三者
- 每种方法都能说出输入、输出、训练目标

### 第 2 步：手写核心公式

你需要能自己写出：

```text
Flow:
x_t = (1 - t) * noise + t * x0
v_target = x0 - noise

Diffusion:
x_t = alpha * x0 + sigma * noise
v_target = alpha * noise - sigma * x0
```

完成标准：

- 不看笔记写出来
- 能解释每个符号的含义

### 第 3 步：完成填空代码题

文件：
[policy_math_fill_in.py](/Users/liuchu/codes/homework_spring2026/hw1/exercises/policy_math_fill_in.py)

你要把里面所有 `TODO` 填完。

完成标准：

- 能运行检查脚本
- 检查脚本全部通过

### 第 4 步：自己解释训练和采样为什么要对应

你需要回答：

- 为什么 diffusion 训练时要随机采样 `t`
- 为什么采样时要从噪声开始
- 为什么 schedule 会影响结果
- 为什么 `linear` 在 50 步下会差
- 为什么 `sqrt` 在当前作业里更强

完成标准：

- 每个问题都能写出 3 到 5 句话解释

### 第 5 步：做实验总结

你需要整理一个最小结论表：

| 方法 | 核心对象 | 优点 | 缺点 |
|---|---|---|---|
| MSE | 直接动作 | 简单、稳定 | 易平均化 |
| Flow | 速度场 | 目标干净、容易训 | 仍需多步采样 |
| Diffusion | 多噪声等级去噪 | 表达强、结构完整 | 更吃 schedule 和训练预算 |

完成标准：

- 你能自己补完整这个表
- 你能写一段自己的总结，不抄书

---

## 每日任务

### Day 1

- 讲清 `MSE / Flow / Diffusion` 三者区别
- 写出它们各自的输入、输出、loss

### Day 2

- 手写 `Flow` 的两个核心公式
- 手写 `Diffusion` 的两个核心公式
- 完成填空题第 1 部分

### Day 3

- 完成填空题第 2 部分
- 自己解释 `x0 / noise / v` 的关系

### Day 4

- 完成填空题第 3 部分
- 解释为什么理论等价不代表训练效果一样

### Day 5

- 写一页总结：
  `为什么这份作业里 flow 比 diffusion 更容易训好`

### Day 6

- 写一页总结：
  `为什么 linear 最差，sqrt 最好`

### Day 7

- 整体回顾
- 把三种方法完整讲一遍

---

## 代码练习

练习文件：
[policy_math_fill_in.py](/Users/liuchu/codes/homework_spring2026/hw1/exercises/policy_math_fill_in.py)

检查脚本：
[check_policy_math_fill_in.py](/Users/liuchu/codes/homework_spring2026/hw1/exercises/check_policy_math_fill_in.py)

运行方式：

```bash
cd /Users/liuchu/codes/homework_spring2026/hw1
uv run python exercises/check_policy_math_fill_in.py
```

如果你还没填完，检查脚本会失败；填完后应该全部通过。

---

## 你需要真正掌握的问题

做完以后，你要能独立回答下面这些问题：

1. 为什么 `MSE` 容易学成平均动作？
2. 为什么 `Flow Matching` 学的是速度场？
3. 为什么 `Diffusion` 比 `Flow` 更吃训练预算？
4. 为什么 `noise / x0 / v` 理论相关，但训练效果不同？
5. 为什么 `schedule` 会影响 diffusion 结果？
6. 为什么在当前作业里 `linear` 最差、`sqrt` 最好？

---

## 提交方式

你每做完一部分，就把你修改后的文件留在本地：

- [policy_math_fill_in.py](/Users/liuchu/codes/homework_spring2026/hw1/exercises/policy_math_fill_in.py)
- 你自己写的总结 markdown 或 txt

到时候我可以直接帮你检查：

- 公式是否对
- 代码是否对
- 解释是否真的讲通了

---

## 最终目标

你不是只要“会跑出一个分数”，而是要做到：

- 看见公式能知道它在做什么
- 看见代码能知道每一步为什么存在
- 看见实验结果能知道问题大概率出在哪

做到这一步，`MSE / Flow / Diffusion` 才算真正学会。
