# 强化学习数学原理练习题单

这份题单分成两部分：

- 基础练习：适合先熟悉 MDP、Bellman 方程、策略评估、值迭代
- 进阶练习：适合进一步训练理论推导、收敛性分析和策略性能分析

建议做题方式：

1. 先独立完成基础部分
2. 再开始进阶部分
3. 每题先写公式，再写文字解释
4. 遇到证明题时，优先从定义和 Bellman 方程出发

---

## 一、基础 6 题

这套题按“先概念、再计算、再推理”排序，适合纸笔练习。

记号约定：

- `S` 为状态集合
- `A` 为动作集合
- `P(s',r | s,a)` 为转移与奖励分布
- `γ` 为折扣因子
- `V^π(s)` 为策略价值函数
- `Q^π(s,a)` 为动作价值函数

### 练习 1：判断是否构成 MDP

某智能体在一个 `3x3` 网格中移动。状态只记录当前位置 `(x,y)`，动作是上下左右。每走一步奖励 `-1`，走到终点奖励 `+10` 并结束。

1. 说明这个问题的状态、动作、奖励、终止条件分别是什么。
2. 只记录当前位置是否足以构成 MDP？为什么？
3. 如果环境里有一扇门，门是否打开取决于“上一步是否按过按钮”，但状态里不记录这个信息，这时还是否是 MDP？为什么？

## answer
1. 状态是当前的位置(x,y) + 运动步数
   动作是： 上下左右
   终止条件： 到达终点
  
2. 不行还是需要走了多少步

3. 不行，当前状态只跟上一个状态相关

练习目标：理解“马尔可夫性”到底在说什么。

### 练习 2：写出 Bellman expectation equation

考虑一个固定策略 `π`，它在状态 `s` 下以概率：

- `π(a1|s)=0.7`
- `π(a2|s)=0.3`

环境满足：

- 执行动作 `a1` 后，以概率 `0.8` 转到 `s1` 并得到奖励 `2`，以概率 `0.2` 转到 `s2` 并得到奖励 `0`
- 执行动作 `a2` 后，必然转到 `s2` 并得到奖励 `1`

设 `γ=0.9`。

1. 写出 `V^π(s)` 的 Bellman expectation equation。
2. 用 `V^π(s1)` 和 `V^π(s2)` 表示 `V^π(s)`。
3. 再写出 `Q^π(s,a1)` 和 `Q^π(s,a2)` 的表达式。
4. 验证 `V^π(s) = Σ_a π(a|s) Q^π(s,a)`。

## answer


练习目标：熟练把文字条件翻成 Bellman 方程。

### 练习 3：两状态 MRP 线性方程组

考虑一个 Markov Reward Process，只有两个非终止状态 `A, B`，折扣因子 `γ=0.8`。

转移与奖励如下：

- 从 `A` 出发：
  - 以概率 `0.5` 回到 `A`，奖励 `2`
  - 以概率 `0.5` 去 `B`，奖励 `0`
- 从 `B` 出发：
  - 以概率 `1` 去 `A`，奖励 `1`

1. 写出 `V(A), V(B)` 的 Bellman equations。
2. 将它们化为二元一次方程组。
3. 解出 `V(A), V(B)`。
4. 判断哪个状态更“好”，并解释原因。

练习目标：把 Bellman equation 当作线性方程组来解。

### 练习 4：手算两轮策略评估

考虑一个小型 MDP，有三个状态：`s0, s1, T`，其中 `T` 为终止状态，且 `V(T)=0`。固定策略 `π` 如下：

- 在 `s0`，策略总是选动作 `a`
- 在 `s1`，策略总是选动作 `b`

环境为：

- 在 `s0` 执行动作 `a`：得到奖励 `1`，并转到 `s1`
- 在 `s1` 执行动作 `b`：得到奖励 `2`，并以概率 `0.5` 留在 `s1`，以概率 `0.5` 到 `T`

设 `γ=0.9`，初始取 `V_0(s0)=V_0(s1)=0`。

1. 写出同步更新的策略评估公式。
2. 计算 `V_1(s0), V_1(s1)`。
3. 计算 `V_2(s0), V_2(s1)`。
4. 根据数值直觉说明：为什么 `V(s1)` 的估计会逐步上升？

练习目标：练熟 iterative policy evaluation 的更新过程。

### 练习 5：手算两轮 Value Iteration

考虑状态 `s` 下有两个动作 `a1, a2`，折扣因子 `γ=0.9`。

- 动作 `a1`：立刻得到奖励 `3`，然后终止
- 动作 `a2`：立刻得到奖励 `1`，然后回到状态 `s`

令初值 `V_0(s)=0`。

1. 写出 Bellman optimality update：

   `V_{k+1}(s) = max_a Σ_{s',r} P(s',r|s,a)[r + γ V_k(s')]`

2. 计算 `V_1(s)`。
3. 计算 `V_2(s)`。
4. 从第几轮开始，最优动作会稳定下来？
5. 这个例子里，最优动作最终是哪一个？为什么？

练习目标：理解 value iteration 如何权衡短期奖励和长期回报。

### 练习 6：折扣因子对最优策略的影响

某状态 `s` 下有两个可选动作：

- 动作 `safe`：立即得到奖励 `4`，然后终止
- 动作 `risky`：立即得到奖励 `0`，下一步必到状态 `s'`
- 在 `s'`：立即得到奖励 `10`，然后终止

1. 分别写出选择 `safe` 和 `risky` 时的回报表达式。
2. 用 `γ` 表示两者的大小关系。
3. 求出在什么条件下，智能体会偏向 `risky`。
4. 当 `γ=0.2, 0.5, 0.9` 时，分别判断最优动作。
5. 用一句话解释：折扣因子为什么会影响策略偏好？

练习目标：理解 `γ` 的数学含义，而不是只记它在 `0` 到 `1` 之间。

### 基础题建议顺序

1. 练习 1
2. 练习 2
3. 练习 3
4. 练习 4
5. 练习 6
6. 练习 5

---

## 二、进阶 8 题

这套题比基础题高一档，重点从“会算 Bellman 方程”推进到“会证明 RL 里的关键结论”。

### P1. 奖励平移对最优策略的影响

考虑折扣 MDP，折扣因子 `γ∈(0,1)`。把所有即时奖励统一改成：

`r'(s,a,s') = r(s,a,s') + c`

其中 `c` 是常数。

1. 推导 `V'^π(s)` 与 `V^π(s)` 的关系。
2. 推导 `Q'^π(s,a)` 与 `Q^π(s,a)` 的关系。
3. 证明：最优策略集合是否改变？
4. 再讨论如果把奖励改成 `r'' = α r + c`，其中 `α>0`，最优策略是否改变？

工具：等比级数、价值函数定义、`argmax` 不变性。

### P2. 折扣 MDP 中的 Performance Difference Lemma

设 `π` 和 `π'` 是两个策略，定义优势函数：

`A^π(s,a) = Q^π(s,a) - V^π(s)`

1. 从 Bellman equation 出发，推导 `V^{π'} - V^π` 的展开式。
2. 证明：

   `J(π') - J(π) = (1 / (1-γ)) E_{s~d^{π'}, a~π'(·|s)} [A^π(s,a)]`

   其中 `d^{π'}` 是折扣占据分布。

3. 用这个结论解释：为什么“在每个状态上贪心地提升”通常会带来全局性能改进？

工具：Bellman 展开、telescoping sum、occupancy measure。

### P3. 有限时域版本的 Performance Difference Lemma

考虑时域 `H` 的 finite-horizon MDP，状态价值记为 `V_h^π(s)`，动作价值记为 `Q_h^π(s,a)`。

1. 写出时间相关的 Bellman recursion。
2. 推导 finite-horizon 下的 advantage decomposition。
3. 证明：

   `V_1^{π'}(ρ) - V_1^π(ρ) = Σ_{h=1}^H E_{s_h~d_h^{π'}, a_h~π'(·|s_h)} [A_h^π(s_h,a_h)]`

4. 比较它和折扣版本的异同。

工具：数学归纳、分步展开、trajectory decomposition。

### P4. 为什么 `γ=1` 时迭代策略评估可能不收敛

考虑一个 continuing MDP，`γ=1`。

1. 构造一个简单反例，使 iterative policy evaluation 不收敛。
2. 写出对应 Bellman operator。
3. 说明它为什么不再是压缩映射。
4. 讨论：如果是 episodic 且吸收终止状态存在，`γ=1` 时是否仍可能收敛？

工具：反例构造、fixed point、contraction mapping。

### P5. 异步价值迭代为什么还能收敛

考虑 discounted MDP，`γ<1`。每次只更新一个状态：

`V_{k+1}(s_k)=max_a Σ_{s'} P(s'|s_k,a)[r(s_k,a,s')+γ V_k(s')]`

其余状态保持不变。

1. 给出“每个状态被无限次访问”的条件。
2. 证明该异步更新仍收敛到 `V^*`。
3. 在证明中指出哪里用了 `γ<1`。
4. 解释同步 value iteration 和 totally asynchronous update 的本质共同点。

工具：sup norm、Bellman optimality operator、收敛证明。

### P6. 有限时域 Bellman optimality 的 backward recursion

考虑 `T` 步 MDP，终端值 `V^*_{T+1}(s)=0`。

1. 写出时刻 `t` 的最优值函数定义。
2. 证明：

   `V_t^*(s) = max_a Σ_{s',r} P(s',r|s,a)[r + V_{t+1}^*(s')]`

3. 由此推出最优策略可以只依赖 `(t,s)`。
4. 解释为什么 finite-horizon 下通常不需要 stationary policy。

工具：backward induction、动态规划原理。

### P7. First-visit MC 与 Every-visit MC 的理论区别

考虑给定策略 `π` 下的 Monte Carlo 价值估计。

1. 写出 first-visit MC 和 every-visit MC 对 `V^π(s)` 的估计定义。
2. 说明为什么 first-visit MC 可以直接用大数定律证明收敛。
3. 为什么 every-visit MC 中样本项不再独立同分布？
4. 这是否意味着 every-visit MC 不收敛？请给出简要讨论。

工具：大数定律、样本相关性、估计量定义。

### P8. 为什么简单 `ε`-greedy 可能做不好深度探索

考虑一个“链式 MDP”：只有一直选择正确动作，才能在很深的位置拿到大回报；一旦中途选错，就只得到小回报并结束。

1. 构造一个长度为 `H` 的 chain MDP。
2. 分析 `ε`-greedy 在该环境下找到最优轨迹的概率如何随 `H` 变化。
3. 解释为什么这个概率会指数级变小。
4. 从“乐观估计 / UCB bonus”的角度说明，为什么基于 optimism 的方法更适合这类问题。

工具：概率乘法、探索复杂度、regret 直觉。

### 进阶题建议顺序

1. `P1`
2. `P4`
3. `P6`
4. `P5`
5. `P7`
6. `P2`
7. `P3`
8. `P8`

### 进阶题难度分层

- 偏入门进阶：`P1 P4 P6`
- 标准理论训练：`P5 P7 P2`
- 更硬一点：`P3 P8`

---

## 三、简单补充 16 题（公开来源，不依赖看图）

这一组题全部来自公开教材或课程页面，尽量选择不依赖插图、不需要看曲线图也能直接完成的题目，适合空闲时做“短练习”。

### S1. `[SB 3.7]` 动作价值函数的 Bellman 方程

写出动作价值函数 `q^π(s,a)` 的 Bellman 方程，用后继状态-动作对的 `q^π(s',a')` 表示。

### S2. `[SB 3.9]` Continuing task 中奖励平移

在 continuing task 中，如果给所有奖励统一加上常数 `c`，为什么不会改变任意两个状态的相对优劣？并写出新增的常数项。

### S3. `[SB 3.10]` Episodic task 中奖励平移

如果是在 episodic task 中，把所有奖励都加上常数 `c`，结论还和上一题一样吗？为什么？给一个例子。

### S4. `[SB 3.11]` 用 `q^π` 表示 `v^π`

写出 `v^π(s)` 用 `q^π(s,a)` 表示的两个公式：一个带期望符号，一个完全展开成 `π(a|s)` 的求和式。

### S5. `[SB 4.1]` 计算两个动作价值

在 equiprobable random policy 下，求 `q^π(11, down)` 和 `q^π(7, down)`。

### S6. `[SB 4.2]` 新增状态后的状态价值

在 Example 4.1 的 `4x4` gridworld 中，先新增状态 `15`，并按题目给定方式连接；求随机策略下的 `v^π(15)`。然后再把状态 `13` 的 `down` 动作改成通往 `15`，再求一次。

### S7. `[SB 4.3]` Action-value 版本的迭代策略评估

写出 action-value 版本的迭代策略评估公式，也就是 `q^π` 的 Bellman 方程，以及 `q_0,q_1,q_2,...` 的逐次逼近形式。

### S8. `[DTU Ex8, Problem 1]` Bellman 方程的含义

用一段话解释 Bellman 方程左边和右边各代表什么，并说明它为什么能写成递归形式。

### S9. `[DTU Ex8, Problem 2]` 两状态 MRP 计算题

两状态 MRP：状态 `s1,s2` 各以 `1/2` 概率留在原状态或切到另一个状态；`s1->s1` 奖励 `1`，`s2->s2` 奖励 `2`，跨状态转移奖励 `0`。

1. 在 `v2=5/2, γ=2/3` 时算 `v1`。
2. 把一般情形写成线性方程组 `b=Av`。

### S10. `[DTU Ex8, Problem 3]` 固定策略下的线性方程组

对一个给定 MDP 和固定策略 `π`，写出状态值函数满足的线性方程组，并说明如何通过解方程得到 `v^π`。

### S11. `[DTU Ex8, Problem 4]` Iterative policy evaluation

写出 iterative policy evaluation 的核心更新，并解释它和 `Q(s,a)=E[r+γV(s')|s,a]` 的关系。

### S12. `[DTU Ex8, Problem 5]` 为什么 policy iteration 不会变差

说明 policy iteration 为什么由“策略评估 + 策略改进”两个阶段组成，并解释每一轮为什么不会让策略变差。

### S13. `[DTU Ex8, Problem 6]` Value iteration 的更新公式

写出 value iteration 的 Bellman optimality update，并解释它和 policy iteration 的主要区别。

### S14. `[SB 6.1]` TD 为什么可能优于 Monte Carlo

给出一个场景，使得 TD 更新平均上比 Monte Carlo 更新更好；可以沿用“开车回家但停车场变了”的思路解释。

### S15. `[CS188 Proj3 Q1]` 为什么 value iteration 要同步更新

说明 value iteration agent 在每轮更新时，为什么要基于“上一轮的值函数”同步更新所有状态。

### S16. `[CS188 Proj3 Q6]` `ε`-greedy 的动作选择

说明 `ε`-greedy 选动作的规则，特别是为什么“随机选动作”也可能恰好选到当前最优动作。

---

## 四、参考来源

### 基础部分

- Sutton & Barto, *Reinforcement Learning: An Introduction*  
  [PDF](https://web.stanford.edu/class/psych209/Readings/SuttonBartoRL.pdf)
- DTU Exercise 8: Bellman equations and exact planning  
  [说明页](https://www2.compute.dtu.dk/courses/02465/exercises/ex08.html)  
  [PDF](https://www2.compute.dtu.dk/courses/02465/_assets/02465ex8_Python.pdf)
- Stanford CS221 Blackjack Assignment  
  [作业页](https://web.stanford.edu/class/archive/cs/cs221/cs221.1192/assignments/blackjack/index.html)
- Stanford CS234 Assignment 1  
  [PDF](https://web.stanford.edu/class/cs234/assignments/a1/CS234_A1_Questions.pdf)

### 进阶部分

- UW CSE 542 HW1  
  [PDF](https://courses.cs.washington.edu/courses/cse542/26sp/resources/hw1.pdf)
- UW CSE 542 HW2  
  [PDF](https://courses.cs.washington.edu/courses/cse542/26sp/resources/hw2.pdf)
- Mannheim Reinforcement Learning Exercise Sheet 5  
  [PDF](https://www.wim.uni-mannheim.de/media/Lehrstuehle/wim/doering/RL/Uebungsblaetter2026/ub_05_26.pdf)
- Mannheim Reinforcement Learning Exercise Sheet 6  
  [PDF](https://www.wim.uni-mannheim.de/media/Lehrstuehle/wim/doering/RL/Uebungsblaetter2026/ub_06_26.pdf)

---

### 简单补充部分

- Berkeley CS188 Project 3  
  [项目页](https://inst.eecs.berkeley.edu/~cs188/archive/fa23/projects/proj3/)

---

## 五、可选使用方式

如果你后面想继续扩展，可以在这份文件后面继续加：

- 每题的提示
- 每题的解答草稿
- 你自己的错题总结
- 按章节整理的公式表

也可以把做完的答案单独写到另一份文件里，比如：

- `rl_math_practice_answers.md`
