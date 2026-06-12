# 练习单 11：Eligibility Traces

来源：Sutton & Barto，第 12 章  
对应编号：`12.1` + 本章主节巩固题

目标：

- 理解 `λ-return`
- 理解 `TD(λ)` 与 eligibility trace
- 理解 forward view 与 backward view 的对应关系

---

## Q1. `[Ex 12.1]` `λ` 如何控制指数加权

1. `λ-return` 为什么可以看成对不同 `n-step return` 的指数加权平均？
2. 当 `λ=0` 时它退化成什么？
3. 当 `λ=1` 时它又退化成什么？
4. 因此 `λ` 在偏差与方差之间扮演了什么角色？

### 作答区

- 

---

## Q2. 巩固题：forward view 与 backward view

1. `forward view` 是从什么角度定义更新目标的？
2. `backward view` 为什么更适合在线实现？
3. 二者想实现的本质目标是否一致？

### 作答区

- 

---

## Q3. 巩固题：为什么 eligibility trace 有记忆效果

1. eligibility trace 在算法里记录了什么信息？
2. 为什么最近访问过的状态或状态动作对会得到更大的更新资格？
3. 这和 one-step TD 相比，带来了什么传播 credit 的优势？

### 作答区

- 

---

## Q4. 巩固题：true online TD(λ) 的动机

1. 为什么作者还要专门讨论 true online 版本？
2. 它想修正普通在线实现中的什么不一致？
3. 这说明 forward/backward 等价在实践中为什么并不总是自动成立？

### 作答区

- 

