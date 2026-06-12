# 练习单 04：Monte Carlo

来源：Sutton & Barto，第 5 章习题改写  
对应编号：`5.3, 5.5, 5.7`

目标：

- 理解 off-policy Monte Carlo
- 理解 ordinary / weighted importance sampling
- 能把状态值公式改写成动作值公式

---

## Q1. `[Ex 5.3]` 把 importance sampling 从 `V` 改成 `Q`

书中给出了针对状态值 `V(s)` 的 off-policy importance sampling 形式。

1. 把它改写成针对动作价值 `Q(s,a)` 的版本。
2. 说明对应的采样起点、权重和回报分别如何定义。
3. 为什么这里要把状态和动作一起固定？

### 作答区

- 

---

## Q2. `[Ex 5.5]` one-state 例子中的 every-visit 方差

在书里的 one-state off-policy MC 例子中：

1. 如果把 first-visit 改成 every-visit，估计量的方差还会不会是无穷大？
2. 说明你的判断理由。
3. 这个例子反映了 ordinary importance sampling 的什么风险？

### 作答区

- 

---

## Q3. `[Ex 5.7]` 推导 weighted importance sampling 的增量更新

从加权平均定义出发，推导 weighted importance sampling 的递推更新式。

1. 写出累计权重 `C_n` 的更新。
2. 写出价值估计 `V_n` 或 `Q_n` 的增量形式。
3. 说明它与普通 sample average 更新看起来像在哪里。

### 作答区

- 

