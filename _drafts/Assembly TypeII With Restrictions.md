---
layout: post
title: Assembly Type-II With Restrictions
categories: [Assembly]
description: Minimizing the Cycle Time in Two-Sided Assembly Lines with Assignment Restrictions: Improvements and a Simple Algorithm
keywords: Two-Sided Assembly lines
---

双边装配线里面一个基本的问题就是找到最佳节拍，解决了这个问题相当于解决了大部分的问题，因为求解思路基本是比较相近的。本篇我们讨论加入了一些约束，有一定的复杂性，要求算法可以很好的兼容约束。基本上目前的算法都是属于贪婪(Iterated Greedy)算法，为了收缩广大的收缩域。

## Introduction

A pair of face-to-face stations like station (n,1) and station (n,2) is called **a mated-station** and one of them is **the companion** of the other. The **idle** time on each station can be divided into two types: **Sequence-dependent idle time** and **the remaining idle time** existing at the rear of the last task. There are other assignment restrictions which exist in real applications need to be considered, including zoning restriction, synchronous restriction, positional restriction, distance restriction and resource restriction. We focus on **zoning restriction**, **synchronous restriction**, **positional restriction** which are usually found in real applications. 

**Zoning restriction**: consist of positive zoning restriction and negative zoning restriction. Positive zoning restriction means that some tasks need to be assigned to the **same station**  while negative zoning restriction **prohibits** (禁止) some tasks being allocated to the **same mated-station**. 

**Synchronous restriction**: restrict two operators to **perform a pair of tasks simultaneously on both sides of the same mated-station** for collaboration. (facing directly and have the **same starting time**)

**Positional restriction**: indicates that certain tasks need to be allocated on **predetermined stations or mated-stations**.

## Mathematical Model

### Assumptions

1. A single model is taken into account and the travel time between stations is ignored.
2. Precedence diagrams are known and deterministic
3. The task with a positional restriction should be assigned to the predefined station.
4. The tasks in the positive zoning restriction should be assigned to the same station.
5. The tasks in the negative zoning restrictions are prohibited to be allocated to the same mated-station 

### Mathematical form

The objective function is as follow:
$$
\begin{align}
Fit=& w_1\sum_{j\in J}\sum_{k\in K(j)}\frac{CT-SF_{jk}}{2nm}+w_2\sum_{j\in J}\sum_{k\in K(j)}\frac{SF_{jk}-ST_{jk}}{2nm}\\
&+w_3\sqrt{\sum_{j\in J}\sum_{k\in K(j)}\frac{(CT-SF_{jk})^2}{2nm}}+w_4\sqrt{\sum_{j\in J}\sum_{k\in K(j)}\frac{(SF_{jk}-ST_{jk})^2}{2nm}}
\end{align}
$$
$nm$ 表示number of mated-stations. J表示所有配对位置，K表示所有方位(左工位和右工位)，$SF_{jk}$ and $ST_{jk}$ 分别表示该工位的最后一个任务的完工时间和所有任务的作业时间和。上面的目标函数分为四个部分：第一部分表示最小化剩余空闲时间，第二部分表示最小化序列空闲时间，第三部分和第四部分表示**各个位置之间的上述两个空闲时间的平衡**（平方项求和中最大项和最小项的比值显示了平衡程度）。

## Encoding and Decoding

采用位置编码无疑是双边装配线固定位置数目一个比较好的方法，在解码的过程中我们采用启发式规则。我们通过**调整任务的优先权重来减少序列空闲时间**。当然我们还需要考虑任务的约束，这是我们一个创新点。

### Initial Priority Value Adjustment

所有任务优先权都在区间[0,1]。比如著名的RPW 启发式算法权重定义：任务和所有后续的作业时间和。我们首先将所有任务按照一定权重降序排列，然后任务$i$ 的先验概率$PL[i]$ 可以通过下式计算：
$$
PL[i]=1-\frac{2TS_i-1}{2nt}
$$
其中$TS_i$ 表示任务i在排序中的序号，$nt$ 表示任务的数目。这样权重越大的任务先验概率越大。当然我们也可以在搜索的过程中改变任务的权重，然后根据新的权重更新先验概率。

### Task Assignment Rule

任务的分配规则是用来**降低序列空闲时间和平衡负荷**的。首先，我们寻找不会增加序列空闲时间的任务，提高这些任务的权重(减小会产生序列空闲时间的任务权重)；同样如果一个任务分配可以使得该工位的结束时间(就是这个任务的结束时间)位于区间[Cm,CT]时，我们也提高这个任务的权重。其中Cm表示位置的平均负荷，计算方法如下：$Cm=\sum_{i\in I}t_i/2nm$ . 这样负荷会比较平衡，因为所有位置的负荷都在这个区间内波动。

这个算法的顺序如下：

> 1. 计算先验概率
>
> 2. $PL[i] =PL[i]+1$ if task i satisfies **cycle time,direction,and precedence restrictions** while $j\neq nm$ or satisfies only **direction and precedence restrictions** while $j=nm$ 
>
>    (仅仅是要满足节拍、方位、先序约束)
>
> 3. if PL[i]>1 and task i in the assignable task set can begin at the earliest start time of the current station, then $PL[i] =PL[i]+1$  (需要现在就可以分配的任务，就是不会产生序列空闲时间的任务优先，如果现在左右结束时间一样，那么所有没有先序的任务权重都加1，最后一个位置不考虑节拍约束是为了让所有任务都分配，结果一定是可行的)
>
> 4. if PL[i]>1 and the finishing time of task i in the assignable task set is in [Cm,CT], then $PL[i] =PL[i]+1$ 
>
> 5. 经过以上步骤后有三种情况：$PL[i]<1$表示任务i不能分配；$1<PL[i]<2$ 表明任务i可以分配但是权重没有被提升；$PL[i]>2$ 表明任务可以分配并且优先分配

### Positional Restriction Handling

首先我们来看一下固定任务约束，if the current station is the predetermined station $(j^\ast,k^\ast)$ ，the priority of the task is increased. 为了实现这个目标，我们构建了一个矩阵TS，定义如下：
$$
\begin{align}
&TS[i][j][k]=-\psi,\; \forall j<j^\ast,k\in K(i)\\
&TS[i][j][k]=1,\; \forall j=j^\ast,k=k^\ast\\
&TS[i][j][k]=0,\;otherwise
\end{align}
$$
那么任务i的概率更改为：$PL[i]=PL[i]+TS[i][j][k]$ 这样如果在固定位置之前的配对位置，任务i的概率$PL[i]<1$ 所以任务不会被分配。如果当前位置正好是预先设定的位置，任务i的概率会被增加。 当然，也可能存在任务i并不能分配到预先定义的位置的情况，所以我们对于后面的位置不加约束，为了就是能够得到一个可行解，这种定义方便了编程操作难度。

### Positive Zoning Restriction Handling

前面有人将这样的任务看做一个整体分配，虽然降低了难度，但是这样顺序一定会带来结果上的差距，所以我们不考虑这样的方法。符合前面的规范性，我们同样定义一个矩阵TT来实现这个约束：
$$
TT[i][h]=TT[h][i]=1,\; \forall (i,h)\in PZ
$$
for task i, if PL[i]>1(首先要确认可以分配), its priority is updated as follow:
$$
PL[i]=PL[i]+\sum_{h=1}^{NT}TT[i][T_h]
$$
其中NT表示station(j,k)已经分配的任务数目，$T_h$ 表示当前工作站分配的第h个任务。这个调整表明如果当前位置**至少存在一个PZ任务**，那么这个任务的权重就会上升，当然我们也无法保证PZ任务对一定能完全符合约束，只能在最大限度上保证。

  ### Decoding Scheme

> 1. Set the initial cycle time $CT_{initial}$ 
> 2. Open a new mated-station
> 3. Choose a side with greater capacity(default select left side)
> 4. Update the priority values with task assignment rule
> 5. If task i, $i\in I$ is not in  the negative zoning restriction, go to next step. Else if PL[i]>1 and other task has been allocated to the current mated-station, then $PL[i]=Pl[i]-\psi$ (其实吧这里可以同PZ一样操作，从数学上更加简洁一点)
> 6. If PL[i]>1, then $PL[i]=PL[i]+TS[i][j][k]$ 
> 7. If PL[i]>1, then $PL[i]=PL[i]+\sum_{h=1}^{NT}TT[i][T_h]$ (这里和上一步分开操作，因为上一步如果不可行这一步就没有必要了，其实固定任务和PZ任务不会出现在一起，所以一起操作也无妨)
> 8. If task i, $i\in I$ is not in  the synchronous restriction, go to next step. Else, if task i,h in synchronous restriction can be assigned, then assign the two tasks and go to step 10; otherwise, $PL[i]=Pl[i]-\psi$ (很简单了，如果两个任务都可以分配了，赶紧分配，否则降低概率，延后等待)
> 9. 如果所有任务的概率都小于1，那么换边。重新执行步骤4-7. 如果同样没有可分配的任务，回到步骤2，而如果另外这边有可以分配的任务，选择权重最大的任务放入。
> 10. 如果任务分配还没结束，回到步骤3。否则选择最晚的结束时间作为当前的节拍，更新初始的节拍（个人备注：更新的规则很多，可以是相加取平均值等）

初始的节拍生成方式为：$CT_{initial}=\alpha\times Cm (\alpha>1)$ 本文选择的更新节拍规则为$CT_{initial}=CT_{current}-1$ 这样我们就可以逐步获得最佳的节拍。

### Cost refine

在解码的过程中，**the synchronism restriction and the negative zoning restriction are satisfied while positive zoning restriction and positional restrictions may be violated**. 所以我们更新目标损失函数如下：
$$
f=f+w_p np+w_{pz}npz
$$
其中$np$ and $npz$ 分别是违反规则的任务数目，惩罚系数待定。为了让所有任务都满足约束，我们将惩罚系数设置一个很大的数，比如1000

## Iterated Greedy Approach

解码的规则都定好了，就差一个搜索的框架了。我们使用的迭代贪婪搜索方法：首先从一个比较不错的初始解开始，然后随机拆解任务(remove some tasks) 然后用一定的方法重构解(reconstruction: reinserts tasks back). 这样我们获得了一个新的解，然后我们利用一定的规则接受新解和保留旧解，可能我们还会用一定的规则搜索局部更优解。这样一套操作确定：拆解->重构->接受域->局部搜索。我们不断循环这个步骤直到停止，下面是更加详细的介绍。

### Initialize by modified NEH Heuristic

其实这就是一个Sequence Minimize Optimization（按照的序列编码）:

> 1. generate an initial task permutation(排列) $\pi(\pi=[\pi_1,\pi_2,...,\pi_n])$ with RPW heuristic (通过权重选择，可以是随机轮流的权重)
> 2. remove the 2nd task i from $\pi$ and obtain the solution by inserting it into position 1. Then the better one  is selected.(比如可以用我们之前的目标函数)
> 3. remove the remaining tasks i(i>2) and obtain the best solution by inserting it to **all the positions before its incumbent**.

### Improved local search

我们选择一个任务k，把这个任务移出当前的位置，然后插入到所有可行的位置，如果有改进，那么以新的解代替旧的解。我们循环整个算法直到没有改进可以发生。

上面算法目的在于找到局部最优解，但是它**可能会错过局部最优解**，因为不一定所有的任务都能被遍历，而且这个算法很**耗时**，因为它需要遍历所有可以插入的位置。为了克服第一个缺点，我们按照一定顺序($\pi^{rp}$)选择所有的任务进行考察。实验证明**最佳的序列顺序比随机序列顺序策略要好**，所以我们选择最佳的序列作为参考策略。对于第二个缺点，核心在于减少无效解码，但是我们还是要保留搜索局部最优解的能力。所以我们在选择插入位置的时候应该选择满足所有先序和所有后续的位置(其实每次只改变一个任务，我们只需要满足紧邻先序和紧邻后续就可以了)。

原始的算法和改进的算法如下：
$$
\begin{text}

\end{text}
$$
