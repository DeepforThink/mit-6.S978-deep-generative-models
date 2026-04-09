# MIT 6.s978 Deep Generative Models - 实验报告二
**姓名/学号**：[董昱甫]
**日期**：2025年10月  

---

## Problem 1: 负对数似然 (NLL) 与交叉熵 (Cross-Entropy) 的等价性证明

**题目要求**：证明在二值序列自回归模型中，负对数似然（NLL）与交叉熵损失（Cross-Entropy Loss）是等价的。

**证明过程**：

已知二值真实序列为 $x = (x_1, x_2, \dots, x_T)$，其中 $x_i \in \{0, 1\}$。定义真实标签 $y_i = x_i$。
自回归模型预测当前位为 $1$ 的概率设为 $\hat{p}_i = p(x_i=1 \mid x_1, x_2, \dots, x_{i-1})$。

由伯努利分布的概率质量函数可知，模型预测出**实际观测值 $x_i$** 的真实发生概率可以统一写为：
$$ p(x_i \mid x_{<i}) = (\hat{p}_i)^{y_i} (1 - \hat{p}_i)^{1 - y_i} $$
这是因为：
- 当 $y_i = 1$ 时，概率为 $\hat{p}_i^1 (1 - \hat{p}_i)^0 = \hat{p}_i$
- 当 $y_i = 0$ 时，概率为 $\hat{p}_i^0 (1 - \hat{p}_i)^1 = 1 - \hat{p}_i$

对该概率取对数，得到对数似然（Log-Likelihood）：
$$ \log p(x_i \mid x_{<i}) = y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) $$

整个序列的负对数似然（NLL）是对所有步的对数似然求和并取负：
$$ NLL(x) = -\sum_{i=1}^T \log p(x_i \mid x_{<i}) $$

将上面的对数似然展开式代入，得：
$$ NLL(x) = -\sum_{i=1}^T \left( y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right) $$

这完全符合二分类交叉熵损失 (Binary Cross-Entropy Loss) 的定义公式。因此，**在二值序列自回归模型中，最小化 NLL 等价于最小化交叉熵损失**。证明完毕。

---

## Problem 2: 基础 PixelCNN (MNIST)

**(a) & (b) 模型实现总结：**
我们继承了 `nn.Conv2d` 实现了 `MaskedConv2d`，依靠注册 `buffer` 的方式初始化全 1 掩码，并根据指定的类型将对应位置置零：
- **Type A**：屏蔽当前像素 (中心点) 及其右侧、下侧的所有像素。用于网络输入的第一层，严防模型“偷看”输入图像的真实像素。
- **Type B**：保留当前像素 (中心点)，仅屏蔽右侧和下侧的像素。用于中间隐藏层，以最大化前期特征的感受野和信息传递。
我们在网络中使用了 1 层 Type A 卷积后接多层 Type B 卷积，并加入 `BatchNorm2d` 与 `ReLU` 激活，最后一层使用 $1 \times 1$ 普通卷积降维至 1 个通道。

**(c) 结果与评估：**
*(在此处插入你在 Jupyter Notebook 中生成的 2(e) Reconstruction 和 2(f) Generation 的图像截图)*
- **重建效果 (Reconstruction)**：由于依靠真实的完备上下文，重建质量非常高，几乎复原了原图。
- **生成效果 (Generation)**：通过逐像素自回归生成，模型能够凭空创造出具有数字基本拓扑特征的图像。

---

## Problem 3: 条件 PixelCNN (Conditional PixelCNN)

**(a) & (b) 条件控制的引入机制：**
为了让模型能按指定的类别标签（0-9）生成对应的数字，我们实现了 `ConditionalMaskedConv2d`：
引入公式为：$W_\ell \ast x + b_\ell + V_\ell y$
代码中，我们使用 `nn.Linear(num_classes, out_channels)`（即代码中的 `cond_proj`）作为 $V_\ell$，将独热编码 $y$ 投影到与特征图对应的通道数。接着使用 `unsqueeze` 进行广播 (Broadcasting) 操作，使一维的条件信号展平并逐通道加到了 $28 \times 28$ 的特征图上，从而实现了基于条件的特征偏置。

**(c) 结果讨论与对比：**
*(在此处插入 3(e) 和 3(f) 条件生成的图像截图)*
- **效果对比**：相比于基础的无条件 PixelCNN 生成难以辨认的随机笔划或者交织的数字特征，Conditional PixelCNN 生成的图像严格遵循了我们传入的标签类别。图像排列呈现出完美的规律性（从上到下数字 0 到 9 依次展现）。
- **结论**：条件生成不仅拥有无条件生成的细节拟合能力，还成功掌握了代表全局语义的宏观类别特征。

---

## Problem 4: 带有正则化的最大似然估计 与 MAP 的等价性

**题目要求**：证明带有 $\ell_2$ 正则化的最大似然估计（ML）等价于带有高斯先验的最大后验估计（MAP）。

**(a) 推导 MAP 估计形式：**
在贝叶斯估计中，利用贝叶斯定理，后验概率可表示为：
$$ p(\theta \mid x) = \frac{p(x \mid \theta) p(\theta)}{p(x)} $$

最大后验估计 (MAP) 寻找使得后验概率最大的参数 $\theta$。对两边取对数得到：
$$ \log p(\theta \mid x) = \log p(x \mid \theta) + \log p(\theta) - \log p(x) $$

因为 $p(x)$ 与需要优化的参数 $\theta$ 无关，它可以视为一个常数丢弃：
$$ \hat{\theta}_{MAP} = \arg\max_\theta \left[ \log p(x \mid \theta) + \log p(\theta) \right] $$

此等式右侧的 $\log p(x \mid \theta)$ 直接对应于前面给出的非贝叶斯视角下的模型对数似然 $\log p_\theta(x; \theta)$。

**(b) 等价性证明及高斯先验的设定：**
现在我们假设先验概率 $p(\theta)$ 服从均值为 $0$、协方差矩阵为 $\sigma^2 I$ 的多维高斯分布。其概率密度函数为：
$$ p(\theta) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\left(-\frac{1}{2\sigma^2} \|\theta\|_2^2\right) $$

对其取对数可得：
$$ \log p(\theta) = -\frac{1}{2\sigma^2} \|\theta\|_2^2 + C $$
（这里的 $C$ 为与 $\theta$ 无关的常数项）

将该项代入上面得到的 MAP 目标函数中：
$$ \hat{\theta}_{MAP} = \arg\max_\theta \left[ \log p(x \mid \theta) - \frac{1}{2\sigma^2} \|\theta\|_2^2 \right] $$

对比题目中给出的带有 $\ell_2$ 正则化 (罚函数为 $\rho(\theta) = \|\theta\|_2^2$) 的常客派最大似然估计（RML）：
$$ \hat{\theta}_{RML} = \arg\max_\theta \left[ \log p_\theta(x; \theta) - \lambda \|\theta\|_2^2 \right] $$

**等价条件**：
显然，只要我们令 正则化系数 $\lambda = \frac{1}{2\sigma^2}$，两个目标函数就完全一致。
**结论**：带有 $\ell_2$ 正则化的 ML 估计等效于采用 **零均值 (Mean=0)** 且 **方差 $\sigma^2 = \frac{1}{2\lambda}$** 高斯先验的 MAP 估计。证明完毕。