# PyTorch `torch.distributions` 学习笔记（练习版）

这份笔记配合同目录的 `torch_distributions_practice.py` 使用。

---

## 1. 你最常用到的分布

- 离散动作：`torch.distributions.Categorical`
- 连续动作：`torch.distributions.Normal`

在策略梯度里，你通常会做三件事：

1. 从网络输出构造分布对象（`dist = ...`）
2. 采样动作（`action = dist.sample()`）
3. 计算动作的对数概率（`log_prob = dist.log_prob(action)`）

---

## 2. `Categorical`（离散动作）

### 构造方式

- 方式 A：传 `logits`
- 方式 B：传 `probs`

更推荐方式 A（数值更稳定）：

```python
dist = D.Categorical(logits=logits)
```

### 输入输出形状（常见）

- `logits` 形状：`(B, act_dim)`
- `action = dist.sample()` 形状：`(B,)`（每个样本一个类别索引）
- `log_prob = dist.log_prob(action)` 形状：`(B,)`

---

## 3. `Normal`（连续动作）

### 构造方式

```python
dist = D.Normal(mean, std)
```

### 输入输出形状（常见）

- `mean/std` 形状：`(B, act_dim)`
- `action = dist.sample()` 形状：`(B, act_dim)`
- `log_prob = dist.log_prob(action)` 形状：`(B, act_dim)`
- 多维动作常用：
  - `log_prob.sum(dim=-1)` -> `(B,)`

---

## 4. 常见坑

- `Categorical.log_prob` 的动作输入要是整数索引（`long`）
- 连续动作的 `log_prob` 常需要沿动作维求和
- 不要把 `torch.norm` 当分布对象使用
- `Distribution` 对象才有 `.sample() / .log_prob() / .entropy()`

---

## 5. 推荐练习路径

1. 跑一遍 `torch_distributions_practice.py`
2. 改不同 logits，观察 `sample` 频率变化
3. 改 `Normal` 的 `std`，观察 `sample` 分布变化
4. 自己写一个最小 policy-loss：`-(log_prob * advantage).mean()`

