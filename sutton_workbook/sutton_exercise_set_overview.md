# Sutton 题目集总览

这份总览把当前这套 Sutton 题目集整理成一个完整结构，方便你按章节和难度系统推进。

目标不是把整本书的所有书后题全部抄出来，而是：

- 抽出最值得做的习题
- 按学习阶段分层
- 标出哪些题适合手算，哪些题适合推导
- 把前半本和后半本的学习方式区分开

---

## 一、这套题目集怎么组成

整套题目集分成两部分：

### A. 主干练习单

这部分对应 Sutton 前半本最核心的内容，已经拆成 5 张练习单：

1. [01_bandits.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/01_bandits.md)
2. [02_mdp_and_bellman.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/02_mdp_and_bellman.md)
3. [03_dynamic_programming.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/03_dynamic_programming.md)
4. [04_monte_carlo.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/04_monte_carlo.md)
5. [05_td_and_control.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/05_td_and_control.md)

这部分适合第一轮和第二轮学习，重点是把 RL 的骨架练熟。

后半本进阶部分也已经拆成 7 张练习单：

6. [06_multistep_bootstrapping.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/06_multistep_bootstrapping.md)
7. [07_planning_tabular.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/07_planning_tabular.md)
8. [08_prediction_with_approximation.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/08_prediction_with_approximation.md)
9. [09_control_with_approximation.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/09_control_with_approximation.md)
10. [10_offpolicy_with_approximation.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/10_offpolicy_with_approximation.md)
11. [11_eligibility_traces.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/11_eligibility_traces.md)
12. [12_policy_gradient.md](/Users/liuchu/codes/homework_spring2026/sutton_workbook/12_policy_gradient.md)

### B. 全书精华题清单

这部分不再追求每章题量平均，而是按“值得做”的程度抽题。

- 前半本：题多，适合大量纸笔训练
- 后半本：题少，但单题更抽象，适合推导和理解

---

## 二、前半本主干题目

### Chapter 2: Multi-armed Bandits

推荐题：

- `2.1`
- `2.2`
- `2.4`
- `2.5`

训练重点：

- 探索与利用
- 样本平均更新
- 一般步长的权重展开
- optimistic initialization
- UCB 的行为机制

### Chapter 3: Finite Markov Decision Processes

推荐题：

- `3.7`
- `3.9`
- `3.10`
- `3.11`
- `3.12`

训练重点：

- `v^π` 与 `q^π`
- Bellman expectation equation
- continuing / episodic 任务的差异
- 奖励平移

### Chapter 4: Dynamic Programming

推荐题：

- `4.1`
- `4.2`
- `4.3`
- `4.4`
- `4.6`
- `4.7`

训练重点：

- policy evaluation
- policy improvement
- policy iteration
- value iteration
- `V` 版本与 `Q` 版本的 DP

### Chapter 5: Monte Carlo Methods

推荐题：

- `5.3`
- `5.5`
- `5.7`

训练重点：

- off-policy Monte Carlo
- importance sampling
- ordinary / weighted importance sampling
- 增量更新

### Chapter 6: Temporal-Difference Learning

推荐题：

- `6.1`
- `6.5`
- `6.9`

训练重点：

- TD vs MC
- bootstrapping
- random walk 真值
- Q-learning 为什么是 off-policy

---

## 三、后半本精华题目

后半本的题目本来就少，所以不适合按“每章 5 到 10 题”这种密度去学。更合理的方式是每章抓 1 到 2 道代表题。

### Chapter 7: n-step Bootstrapping

推荐题：

- `7.1`

训练重点：

- `n-step return`
- bias / variance 权衡
- 为什么中等大小的 `n` 往往更有效

### Chapter 8: Planning and Learning with Tabular Methods

推荐题：

- `8.1`

训练重点：

- planning vs direct RL
- model-based 更新和 sample-based 更新的关系

### Chapter 9: On-policy Prediction with Approximation

推荐题：

- `9.1`
- `9.2`

训练重点：

- function approximation
- 特征表示
- 多项式 basis

### Chapter 10: On-policy Control with Approximation

推荐题：

- `10.1`

训练重点：

- control with approximation
- 为什么这一章不强调 Monte Carlo

### Chapter 11: Off-policy Methods with Approximation

推荐题：

- `11.1`

训练重点：

- off-policy + approximation 的难点
- semi-gradient 形式
- 为什么这一章容易不稳定

### Chapter 12: Eligibility Traces

推荐题：

- `12.1`

训练重点：

- `λ-return`
- eligibility trace
- 不同 `n-step return` 的加权组合

### Chapter 13: Policy Gradient Methods

推荐题：

- `13.1`

训练重点：

- policy gradient theorem
- 为什么不需要对状态分布直接求导
- 从目标函数到梯度公式的推导

---

## 四、建议学习顺序

### 第一阶段：骨架搭建

按顺序完成：

1. Chapter 2
2. Chapter 3
3. Chapter 4
4. Chapter 5
5. Chapter 6

要求：

- 每题都手写公式
- 小例子能自己算
- 核心定义能默写

### 第二阶段：主线串联

完成后回头统一比较：

- DP / MC / TD
- on-policy / off-policy
- state value / action value
- sample update / full expectation update

### 第三阶段：后半本进阶

再进入：

1. Chapter 7
2. Chapter 8
3. Chapter 9
4. Chapter 10
5. Chapter 11
6. Chapter 12
7. Chapter 13

这一阶段不要追求题量，而要追求：

- 推导是否清楚
- 每一章关键难点是否真的理解
- 能不能把新方法和前半本骨架联系起来

---

## 五、最值得反复做的题

如果你时间有限，这 12 题最值得反复做：

- `2.2`
- `2.5`
- `3.7`
- `3.9`
- `3.10`
- `3.11`
- `4.3`
- `4.6`
- `5.7`
- `6.1`
- `6.9`
- `13.1`

这 12 题基本覆盖了：

- bandits
- Bellman equations
- dynamic programming
- Monte Carlo
- TD learning
- off-policy
- policy gradient

---

## 六、怎么判断你是否学会了

如果你能做到下面这些，说明这套 Sutton 题目集基本吃透了：

1. 不看书写出 `V^π`、`Q^π`、Bellman expectation、Bellman optimality。
2. 能手算小型 MDP 里的策略评估和 value iteration。
3. 能解释 MC、TD、Q-learning 的区别。
4. 能说明 why Q-learning is off-policy。
5. 能说清楚 `n-step`、`λ-return`、policy gradient 分别在解决什么问题。

---

## 七、来源

- Sutton & Barto, *Reinforcement Learning: An Introduction (2nd ed.)*  
  [公开 PDF](https://web.stanford.edu/class/psych209/Readings/SuttonBartoRL.pdf)
