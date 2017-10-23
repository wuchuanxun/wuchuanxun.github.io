---
layout: post
title:FFT
categories: [DSP]
description: Fast  Fourier  Transform
keywords: DFT, FFT
---

​	FFT is a methods for computing the DFT efficiently. In view of the importance of the DFT in various digital signal processing applications, such as linear filtering(线性滤波), correlation analysis（相关分析）, and spectrum analysis（谱分析）, its efficient computation is a topic that has received considerable attention by many mathematicians, engineers, and applied scientists. 

## DFT

​	Basically, the computational problem for the DFT is to compute the sequence {X(k)} of N complex-valued numbers given another sequence of data {x(n)} of length N (sample length), according to the formula:
$$
k(X)=\sum _{n=0}^{N-1} n(x) W_N^{\text{kn}},    0\leq k\leq N-1\\
W_N=e^{-\frac{\text{j2$\pi $}}{N}}
$$
and $W_n^k$ for k=0,1,...,N-1 are called the $N_{th}$ *roots of unity*. They're called this because, in complex arithmetic, $(W_N^k)^N=1$ for all k. They are vertices(顶点) of a regular polygon inscribed in the unit circle of complex plane, with one vertex at (1,0). Below are roots of unity for N=2,N=4 and N=8, graphed in the complex plane.

![roots_of_unity](../images/DSP/FFT/roots_of_unity.PNG)

Powers of roots of unity are periodic with period N.

​	The sequence $A_k$ is the discrete Fourier transform of sequence $a_n$. Each is a sequence of N complex numbers. The sequence $a_n$ is the inverse discrete Fourier transform of sequence $A_K$ . The formula of the inverse DFT is:
$$
a_n=\frac{\sum _{k=0}^{N-1} A_k W_N^{-\text{kn}}}{N}
$$

## FFT

​	Direct computation of the DFT is basically inefficient primarily because it does not exploit the symmetry and periodicity properties of the phase factor $W_n$. In particular, these two properties are : Symmetry property $W_N^{k+\frac{N}{2}}=-W_N^k$ ,Periodic property $W_N^{k+N}=W_N^k$ . Then FFT is come into our eyes. 

Note the N-point DFT can be expressed in terms of the DFT's of the decimated sequences as follows: 
$$
k X=\sum _{n=0}^{N-1} n x W_N^{\text{kn}},0\leq k\leq N-1\\
=\sum _{x\in \text{even}} n(x)W_N^{\text{kn}}+\sum _{x\in \text{old}} n(x)W_N^{\text{kn}}\\
=\sum _{m=0}^{\frac{N}{2}-1} x(2 m)  W_N^{2 \text{mk}}+\sum _{m=0}^{\frac{N}{2}-1} x(2 m+1) W_N^{k (2 m+1)}
$$
But $W_N^2=e^{-\frac{2 \text{j2$\pi $}}{N}}=e^{-\frac{\text{j2$\pi $}}{0.5 N}}=W_{\frac{N}{2}}$ ,with this substitution, the equation can be expressed as:
$$
X(k)=\sum _{m=0}^{\frac{N}{2}-1} x(2 m)W_{\frac{N}{2}}^{\text{km}}+W_N^k \left(\sum _{m=0}^{\frac{N}{2}-1} x(2 m+1)W_{\frac{N}{2}}^{\text{km}}\right)\\
=F_1 (k)+F_2 (k) W_N^k
$$
Since  $F_1(k)$ and $F_2(k)$ are periodic, with period N/2, we have $F_1(k+N/2)=F_1(k)$ and $F_2(k+N/2)=F_2(k)$. In addition, the factor $W_n^{k+N/2}=-W_n^k$. Hence the equation may be expressed as:
$$
\text{X(k)}=F_1 (k)+F_2 (k) W_N^k,k=0,1,\text{...},\frac{N}{2}-1\\
\text{X(k+N/2)}=F_1 (k)-F_2 (k) W_N^k,k=0,1,\text{...},\frac{N}{2}-1\\
$$
where $F_1(k)$ and $F_2(k)$ are $N/2$ point DFTs, we can use the FFT method again and again until a two point DFTs.

​	In conclusion, the FFT algorithm decomposes the DFT into $log_2(N)$ Stages, each of which consists of N/2 butterfly computations. Each butterfly takes two complex numbers p and q and computes from them two numbers, $p+\alpha*q$ and $p-\alpha*q$ , where $\alpha$ is a complex number. Below is a diagram of butterfly operation:

<img src="../images/DSP/FFT/butterfly_operation.PNG" style="zoom:50%" />

use the diagram we can draw the flow chart of FFT, here is a 8-point FFT flow chart:

<img src="../images/DSP/FFT/8_point_fft.PNG" style="zoom:50%" />

It's obviously the inputs aren't in  the normal order. In order to see this more clearly, following is a 16-point FFT:

<img src="../images/DSP/FFT/16_point_fft.PNG" style="zoom:50%" />

we can follow the decomposition process as the figure:

<img src="../images/DSP/FFT/fft_decomposition.PNG" style="zoom:60%" />

If we use a binary tree, it easy to find the sample numbers $n_j$ is the bit-reversal of j. 

<img src="../images/DSP/FFT/sample_order.PNG" style="zoom:70%" />

## Computational complexity of FFT

​	The FFT is a fast algorithm for computing the DFT. If we take the 2-point DFT and 4-point DFT and generalize them to 8-point,16-point,...,$2^r$-point, we get the FFT algorithm. To compute the DFT of N-point sequence using original DFT would take $O(n^2)$ multiplies and adds. The FFT algorithm computes the DFT using $O(n logN)$ multiplies and adds.

## Recommend Video of FFT

Discrete Fourier Transform - Simple Step by Step:

[<img src="../images/DSP/FFT/yutube_play.jpg" style="zoom:160%"/>](https://www.youtube.com/watch?v=mkGsMWi_j4Q)

------

The FFT Algorithm - Simple Step by Step:

[<img src="../images/DSP/FFT/yutube_play.jpg" style="zoom:160%"/>](https://www.youtube.com/watch?v=htCj9exbGo0)



## Reference

1. http://www.cmlab.csie.ntu.edu.tw/cml/dsp/training/coding/transform/fft.html
2. Heckbert P. Fourier Transforms and the Fast Fourier Transform (FFT) Algorithm[J]. Comp. Graph, 1995, 2: 15-463.