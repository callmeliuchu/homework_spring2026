# 练习单 07：Planning and Learning with Tabular Methods

来源：Sutton & Barto，第 8 章  
对应内容：`Ex 8.1` + 本章主节巩固题

目标：

- 理解 model、planning、direct RL 的关系
- 理解 Dyna 的基本思想
- 理解 prioritized sweeping 和 MCTS 的定位

---

## Q1. `[Ex 8.1]` 多步 bootstrapping 能否缩小 planning 的优势

书中指出：纯非规划方法在某个对比实验里显得特别差，一个原因是它只用了 one-step 更新。

1. 如果把它改成 Chapter 7 里的多步 bootstrapping 方法，理论上会不会更好？
2. 为什么多步方法可能缩小与 planning 方法的差距？
3. 为什么它通常仍然不能完全替代显式 planning？

### 作答区

- 

---

## Q2. 巩固题：Dyna 的四个组成部分

1. Dyna 框架里同时进行了哪四类过程？
2. 其中哪一部分直接依赖真实环境交互？
3. 哪一部分利用的是学到的模型而不是真实环境？

### 作答区

- 

---

## Q3. 巩固题：模型错了会怎样

1. 如果模型学错了，planning 更新会受到什么影响？
2. 为什么说 planning 的价值依赖于 model quality？
3. 这和 model-free 方法相比，带来了什么新的风险和什么新的机会？

### 作答区

- 

---

## Q4. 巩固题：prioritized sweeping 的直觉

1. 为什么不是所有状态都值得同等频率地规划更新？
2. 什么样的状态更应该被优先回传更新？
3. 这种思想和普通 value iteration 最大的不同在哪里？

### 作答区

- 

