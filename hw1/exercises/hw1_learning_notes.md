# HW1 学习笔记

对应 [LEARNING_PLAN.md](../LEARNING_PLAN.md) 第 1、4、5 步与每日任务的书面总结。

---

## 第 1 步：三种算法口头解释

### MSE Policy

- **在学什么**：给定 state，直接回归一个 action chunk（确定性映射）。
- **输入**：state `[B, D]`
- **输出**：action chunk `[B, K, A]`
- **训练目标**：`L = ||f(s) - a||^2`，最小化预测动作与专家动作的均方误差。

### Flow Matching

- **在学什么**：从噪声到真实 action 的连续路径上的速度场 `v(s, x_t, t)`。
- **输入**：state、插值点 `x_t`、时间 `t`
- **输出**：速度预测 `v_pred`
- **训练目标**：`L = ||v_pred - (x0 - noise)||^2`，让网络预测从 noise 指向 x0 的方向。

### Diffusion Policy

- **在学什么**：在不同噪声等级下，从 `x_t` 预测 v-parameterization 目标，再反推 x0/noise 做去噪。
- **输入**：state、加噪 action chunk `x_t`、timestep
- **输出**：速度/噪声参数 `v_pred`
- **训练目标**：`L = ||v_pred - (alpha * noise - sigma * x0)||^2`

---

## 第 2 步：核心公式（手写参考）

```text
Flow:
  x_t = (1 - t) * noise + t * x0
  v_target = x0 - noise

Diffusion:
  alpha = sqrt(alpha_bar)
  sigma = sqrt(1 - alpha_bar)
  x_t = alpha * x0 + sigma * noise
  v_target = alpha * noise - sigma * x0
  x0_recover = alpha * x_t - sigma * v
  noise_recover = sigma * x_t + alpha * v
```

符号含义：

- `x0`：干净 action chunk（数据）
- `noise`：标准高斯噪声
- `t` / `timestep`：路径或噪声等级索引
- `alpha_bar`：累积信号保留比例，越小越接近 pure noise
- `v`：v-parameterization，统一 x0 和 noise 两种预测方式

---

## 第 4 步：训练与采样对应关系

### 为什么 diffusion 训练时要随机采样 `t`？

训练时模型需要在**所有噪声等级**上都学会去噪。如果只固定某个 t，网络只见过一种噪声强度，采样时其他 timestep 的分布就会泛化失败。随机 t 等价于对整条 forward 扩散链做 Monte Carlo 覆盖。

### 为什么采样时要从噪声开始？

Forward 过程把 x0 逐步加噪到近似 pure noise；reverse 过程必须沿相反方向走。从 `x_T ~ N(0, I)` 出发，逐步预测 x0 并 re-noise 到更低噪声等级，才能生成新的 action sample。这是生成式建模的基本逻辑。

### 为什么 schedule 会影响结果？

`alpha_bar` 曲线决定了每个 timestep 的 signal-to-noise ratio。Schedule 不同，模型在训练时看到的 `(x_t, t)` 分布就不同，reverse 时每一步的 jump size 也不同。Schedule 与采样步数不匹配时，train/sample 分布偏移会放大。

### 为什么 `linear` 在 50 步下会差？

Linear beta schedule 的 `alpha_bar` 衰减较慢，最后一步仍保留较多信号（远非 pure noise）。但采样只有 50 步，reverse 链必须在有限步内完成去噪；训练分布与采样轨迹不对齐，尤其在高 t 区域模型很少见到足够 noisy 的样本，导致生成质量差。

### 为什么 `sqrt` 在当前作业里更强？

Sqrt schedule 直接令 `alpha_bar` 从 1.0 线性降到 0.0，50 个 timestep 均匀覆盖从干净到 pure noise 的全谱。训练和采样的噪声等级一一对应，没有 linear 那种"训练时还不够噪、采样时却要一步跳回 x0"的 mismatch。

---

## 第 5 步：方法对比表

| 方法 | 核心对象 | 优点 | 缺点 |
|---|---|---|---|
| MSE | 直接动作回归 | 简单、稳定、单步推理 | 多模态数据易平均化，表达力弱 |
| Flow | 速度场 v(s, x_t, t) | 目标干净（x0-noise）、训练稳定 | 推理需多步 Euler 积分 |
| Diffusion | 多噪声等级 v-预测 + 去噪 | 表达力强、生成框架完整 | 更吃 schedule 设计和训练预算 |

### 自己的总结

MSE 把模仿学习当成监督回归，最快但无法建模"同一 state 对应多种合理 action"。Flow 和 Diffusion 都是生成式思路：先定义 noise→data 的路径，再学路径上的向量场。Flow 路径简单（直线插值），目标恒定，所以更容易训好；Diffusion 引入 schedule 和多 timestep，灵活性更高，但 schedule 选不好就会出现 train/sample mismatch。这份作业里 sqrt 优于 linear，本质上是 schedule 与 50 步采样是否对齐的问题。

---

## 每日任务书面记录

### Day 1

- MSE：state → action，L2 loss
- Flow：学速度场，loss = ||v - (x0-noise)||²
- Diffusion：学 v-parameterization，loss = ||v - (α·noise - σ·x0)||²

### Day 2–4

- 公式已写入 `policy_math_fill_in.py` 并通过 `check_policy_math_fill_in.py`
- x0 / noise / v 关系：三者通过 alpha/sigma 线性变换互相可逆，理论等价但网络对不同参数化的梯度尺度不同

### Day 5：为什么 flow 比 diffusion 更容易训好

Flow matching 的插值路径是 `(1-t)*noise + t*x0`，目标速度恒为 `x0 - noise`，不依赖 t 的复杂 schedule。Diffusion 的 v-target 随 alpha_bar 变化，且需要正确的 schedule 才能让各 timestep 的训练分布均匀。Flow 少一个 hyperparameter 维度，优化 landscape 更平滑，所以同样预算下更容易收敛。

### Day 6：为什么 linear 最差、sqrt 最好

Linear：`alpha_bar[-1]` 仍 > 0.5，reverse 50 步时高噪声区域覆盖不足。Sqrt：`alpha_bar` 均匀从 1→0，50 步训练与 50 步采样完全对齐。Cosine 也接近 pure noise 但实现更复杂；在本作业的 50 步设定下 sqrt 最直接有效。

### Day 7：整体回顾

三种方法共享"从数据学 policy"的目标，但归纳偏置不同：MSE 假设单峰、Flow 假设直线路径、Diffusion 假设多步 Markov 去噪。选哪种取决于数据多模态程度和推理时允许的步数。

---

## 需要真正掌握的六个问题

1. **MSE 为什么易学成平均动作？** 多模态分布下 L2 最优解是条件均值，会把多个合理 action 平均成模糊中间值。
2. **Flow 为什么学速度场？** 直线路径 `(1-t)noise + t·x0` 的切向量就是 `x0 - noise`，回归它等价于学 transport map 的局部方向。
3. **Diffusion 为什么更吃训练预算？** 需要覆盖 T 个噪声等级，每步 target 随 schedule 变化，且 schedule/sampling 需联合调优。
4. **noise/x0/v 理论相关但效果不同？** 它们是线性变换关系，但 MSE loss 对不同参数的梯度尺度、数值稳定性不同，网络更容易拟合某一种。
5. **schedule 为什么影响 diffusion？** 它决定每个 t 的 SNR，即训练时 `(x_t, t)` 的分布；与采样步数不匹配会产生 distribution shift。
6. **linear 最差、sqrt 最好？** Linear 在 50 步下 `alpha_bar` 衰减不够；sqrt 均匀覆盖 1→0，train/sample 对齐最好。
