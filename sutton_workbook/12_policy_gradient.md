# 练习单 12：Policy Gradient Methods

来源：Sutton & Barto，第 13 章  
对应编号：`13.1` + 本章主节巩固题

目标：

- 理解 policy parameterization 的动机
- 理解 policy gradient theorem
- 理解 REINFORCE 与 actor-critic 的联系

---

## Q1. `[Ex 13.1]` 用定义和基础微积分证明策略梯度公式

根据书中的公式 `(13.7)`：

1. 从目标函数定义出发进行推导。
2. 说明为什么会出现对数梯度 `∇ log π(a|s,θ)`。
3. 这条公式为什么是后续 REINFORCE 和 actor-critic 的基础？

### 作答区

- 

---

## Q2. 巩固题：为什么要直接参数化策略

1. 与值函数法相比，直接参数化策略的一个主要优势是什么？
2. 在连续动作空间里，这种做法为什么尤其自然？
3. 为什么说它避免了“通过 `argmax` 从值函数间接取动作”的麻烦？

### 作答区

- 

---

## Q3. 巩固题：REINFORCE 的更新直觉

1. REINFORCE 的更新为什么会把高回报动作概率往上推？
2. 它为什么被称为 Monte Carlo policy gradient？
3. 这种方法的主要方差来源是什么？

### 作答区

- 

---

## Q4. 巩固题：baseline 与 actor-critic

1. baseline 的主要作用是什么？
2. 为什么减去 baseline 不改变梯度期望，但能降低方差？
3. actor-critic 中 actor 和 critic 分别在做什么？

### 作答区

- 

