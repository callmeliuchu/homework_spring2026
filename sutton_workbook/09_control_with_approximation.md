# 练习单 09：On-policy Control with Approximation

来源：Sutton & Barto，第 10 章  
对应编号：`10.1` + 本章主节巩固题

目标：

- 理解半梯度控制方法
- 理解为什么这一章主角是 Sarsa 而不是 Monte Carlo
- 理解 average reward 设定与 continuing control

---

## Q1. `[Ex 10.1]` 为什么这一章没有重点讨论 Monte Carlo

1. 在带函数逼近的 control 任务里，为什么 Monte Carlo 往往不是最自然的主角？
2. 相比之下，semi-gradient Sarsa 有什么更直接的优势？
3. 这和 episode 长短、在线更新需求、方差大小分别有什么关系？

### 作答区

- 

---

## Q2. 巩固题：半梯度控制在学什么

1. episodic semi-gradient control 的更新对象是什么？
2. 为什么它需要一个可微的 `q_hat(s,a,θ)`？
3. 它与 tabular Sarsa 的核心结构有什么是一致的？

### 作答区

- 

---

## Q3. 巩固题：`n-step semi-gradient Sarsa`

1. 它相对 one-step 版本主要改了哪里？
2. 更新目标中的 bootstrap 项是什么？
3. 为什么它自然承接了 Chapter 7 的 `n-step return` 思路？

### 作答区

- 

---

## Q4. 巩固题：average reward 设定

1. continuing task 中，average reward 与 discounted return 的关注点有什么不同？
2. 为什么作者专门用一节讨论这个设定？
3. 在长期持续任务里，这个视角什么时候会更自然？

### 作答区

- 

