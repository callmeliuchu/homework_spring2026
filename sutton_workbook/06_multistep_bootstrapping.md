# 练习单 06：Multi-step Bootstrapping

来源：Sutton & Barto，第 7 章  
对应内容：`Ex 7.1` + 本章主节巩固题

目标：

- 理解 `n-step return`
- 理解 `TD(0)` 与 Monte Carlo 之间的连续过渡
- 理解 `n-step Sarsa`、tree backup 和 `Q(σ)` 的角色

---

## Q1. `[Ex 7.1]` 为什么中等大小的 `n` 往往更有效

书中的随机游走实验表明：在很多情况下，性能最好的并不是 `n=1`，也不是特别大的 `n`，而是中间某个 `n`。

1. 为什么 `n=1` 往往偏差较大？
2. 为什么 `n` 很大时又容易带来更高方差或更慢的学习？
3. 为什么增大状态空间后，中间 `n` 的优势通常会更明显？

### 作答区

- 

---

## Q2. 巩固题：写出 `n-step` 预测的目标

1. 写出 episodic 情形下的 `n-step return` `G_t^(n)`。
2. 说明当 `n=1` 时它退化成什么。
3. 说明当 `n` 足够大并覆盖整个 episode 时，它又退化成什么。

### 作答区

- 

---

## Q3. 巩固题：为什么 `n-step Sarsa` 仍然是 on-policy

1. 写出 `n-step Sarsa` 更新目标里最后一个 bootstrap 项的形式。
2. 为什么它学习的仍然是当前行为策略对应的动作价值？
3. 它和 Q-learning 在“目标动作来自哪里”这个问题上有什么本质区别？

### 作答区

- 

---

## Q4. 巩固题：importance sampling 与 tree backup

1. 在 `n-step off-policy` 学习里，为什么通常要引入 importance sampling？
2. `tree backup` 想绕开什么问题？
3. 从直觉上说，`Q(σ)` 为什么可以看成 sample backup 与 expectation backup 的统一？

### 作答区

- 

