# Concepts

## Bra-Ket notation

20世纪初, 物理学家对线性代数还比较陌生, Dirac横空出世, 带来了一场革命.

量子力学的基本假设(就像欧式几何的公理):

1. 量子力学系统的状态由波函数描述:$\bold {\Psi}(\bold{r},t)$.

2. 对于每一个可观测量, 都伴随着一个厄米算符(Hermitian Operator), 比如记为$\hat H$.

3. 对于可观测量的一次测量, 结果只能是上面厄米算符的特征值(本征值: eigenvalue). 而在测量之前, 系统处于叠加态, 由厄米算符的特征向量线性表示($\sum c_i \vert \lambda_i\rangle$, $c_i$为复数, $\vert \lambda_i\rangle$为特征值($\lambda_i$)对应的特征向量), 观测到某个特征值($\lambda_i$)的概率为复系数模的平方($c_i^*c_i$).

4. 可观测量的平均值(数学期望)通过上述厄米算符投影到波函数得到:

   
   $$
   \langle H \rangle = \frac{\int_{-\infty}^{+\infty} \bold{\Psi}^* \hat{H} \bf{\Psi} d\tau}{\int_{-\infty}^{+\infty} \bold{\Psi}^* \bf{\Psi} d\tau }
   $$
   

5. 随着时间变化的波函数满足含时薛定谔方程(Time-dependent Schrödinger Equation):
   $$
   \hat{H} {\bf{\Psi}} ({\bf{r}},t) = i \hbar \frac{\part \bf{\Psi}}{\part t}
   $$
   

我们用复数域的线性空间描述量子力学. 狄拉克记号基本上就是线性代数换了个括号(bra-ket)记号(左矢(bra), 右矢(ket)):  $\vert \cdot \rangle$ 称为右矢(ket), 对应到线性代数中的列向量; $\langle \cdot \vert$表示左矢(bra), 定义为右矢的共轭转置(列向量转置成行向量, 每个元素再取个复共轭).  $\langle \cdot \vert \cdot\rangle$记为两者的点乘, 为复数.

我们可以随便给状态向量取名字, 比如:$\vert +\rangle, \vert -\rangle; \vert \uparrow\rangle, \vert \downarrow\rangle$, 有的在物理里有具体化的含义, 比如光通过偏振片, 我们可以给出$\vert -\rangle, \vert | \rangle$的偏正态,分别表示水平和垂直方向的偏振方向. 如图(图自wiki):

![Vertical_polarization](.\images\Vertical_polarization.svg)

如上图的偏振片, 在实验上, 如果通过偏振片我们的光已经是垂直的偏振光了, 那么它会全部通过偏振片(光的强度在经过偏振片前后一样), 如果是水平的偏振光, 经过垂直的偏振片时, 另一边啥也没有了. 也就是说偏振光经过偏振片的强度(I)或者能量(或者振幅E(概率幅))与 光的偏振方向和偏振片的角度有关($I \propto {\mid E\mid} ^2$), 且强度满足前后之比${\mid\cos\theta \mid}^2$($\theta$为光的偏振方向与偏振片的夹角).

当光通过偏振片时, 强度变小了, 强度与能量成正比, $I \propto E^2$, 那么能量势必变小了, 而光的能量$E=\hbar f$($\hbar$为普朗克常数, 而$f$为光的频率,波长的倒数),  那么势必$f$变小了, 如果波长发生变化, 那么就会导致颜色的变化, 而实验上, 蓝光并不会变成红光, 颜色是不变的. 所以这里需要采用光子来解释, 也就是一个个携带能量的光子通过偏振片数量变少了, 光子通过偏振片是有一定概率的, 而这个概率就是$\mid\cos\theta\mid ^2$.

我们用线性代数和概率来描述这个试验结果.  

我们预先制备了垂直的偏振光($\vert | \rangle$), 通过垂直的偏振片($\langle | \vert$), 我们用两者点积表示: $\langle |\vert|\rangle$, 全部通过那么这个点积结果就是1. 水平的偏振光($\vert-\rangle$)通过垂直的偏振片点积为: $\langle |\vert - \rangle$, 全部没有通过, 那么这个点积结果就是0. 而光通过水平的偏振片($\langle-\vert$)我们有$\langle - \vert |\rangle = 0, \langle-\vert - \rangle=1$. 我们似乎可以很自然地得到一组标准正交基: $\vert |\rangle, \vert-\rangle$. 用向量表示偏振态, 我们最容易想到的当然是令 $\vert -\rangle=(0,1)^T, \vert|\rangle=(1,0)^T$. 任意夹角的方向都可以由水平方向和垂直方向的线性组合得到. 于是当我们的偏振光方向与垂直夹角为$\theta$时, 我们用$\vert |\rangle, \vert-\rangle$的线性组合来表示, $\vert\psi\rangle=\cos\theta \vert |\rangle+\sin\theta \vert-\rangle$.  将点积的结果定义为概率幅(probability amplitude), 概率幅的模的平方定义为通过的概率, 我们能很好的描述量子世界的行为. $\psi$通过垂直偏振片的概率为: $\mid \langle |\vert \psi\rangle \mid^2 = \langle |\vert \psi\rangle^*\langle |\vert \psi\rangle=(\cos\theta \langle|\vert|\rangle + \sin\theta \langle | \vert-\rangle)^*(\cos\theta \langle|\vert|\rangle + \sin\theta \langle | \vert-\rangle) = (1\cos\theta+0\sin\theta)^*(1\cos\theta+0\sin\theta) = \cos^2\theta$.

这里还不会出现虚数, 当我们的偏振光为圆偏振时(测量旋转方向), 会有虚数出现, 这个不影响我们的讨论.

这里需要注意到的是, 我们可以得到一个投影算符:$\vert \cdot\rangle\langle \cdot \vert$. 我们把$\vert b\rangle$向量投影到$\vert a\rangle$上, 可以用$\vert a\rangle\langle a\vert$作用在$b\rangle$上$\vert a\rangle\langle a\vert b\rangle = \langle a\vert b\rangle \vert a\rangle $.

接下来来找我们上述关于偏振方向的厄米算符. 我们回头看量子力学假设, 我们可以知道, 我们要测量的方向这个量对应一个算符$\hat H$, 而厄米算符的特征向量是正交的. 这里$\vert | \rangle, \vert-\rangle$是我们的特征向量, 我们的厄米算符为:
$$
\hat H = (\vert | \rangle \langle |\vert) - (\vert -\rangle\langle-\vert) = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \end{pmatrix} - \begin{pmatrix} 0 \\ 1 \end{pmatrix} \begin{pmatrix} 0 & 1 \end{pmatrix}  = \begin{pmatrix} 1 & 0\\ 0 & 0\end{pmatrix}  - \begin{pmatrix} 0 & 0 \\ 0 & 1  \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & -1  \end{pmatrix} 
$$
我们把$\hat H, \vert | \rangle, \vert -\rangle$的矩阵和向量代入可得: $\hat H \vert | \rangle = 1 \vert | \rangle, \hat H\vert-\rangle = -1 \vert - \rangle$, 我们的特征值为1和-1, 我们对光偏振方向的测量结果如果为垂直的, 那么就对应特征值1, 水平则对应特征值-1.

## Position and Momentum Operators

物理学里有两个基本可观测量:位置(position: $x$), 动量(momentum: $p$). 对应的两个算符分别为$\hat x$和$\hat p$.



测不准原理: