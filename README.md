# MIT 6.S978: Deep Generative Models 🧠

> MIT 6.S978 深度生成模型课程作业与学习记录。

本仓库包含了我在学习 **MIT 6.S978: Deep Generative Models** 课程时的所有编程作业（Problem Sets）、实验报告以及课程幻灯片（Slides）。该课程深入探讨了现代深度生成模型的核心理论与实践。

## 📁 目录结构

主要内容按作业（Assignments）和部分划分：

* **`assignment1/`**：变分自编码器 (VAE) 相关实验，包含模型构建、潜在空间插值（Torus Interpolation）以及 KL 散度与 SGVB 的对比。
* **`assignment2/`**：课程作业 2（包含中文实验报告 `pset2_report_zh.md` / `.tex`）。
* **`assignment3/`**：课程作业 3（可能包含自回归模型 Autoregressive Models 等大型 Notebook 文件）。
* **`assignment4/`**：生成对抗网络 (GAN) 相关作业，包含最终生成的样本结果 (`final_generated_samples.png`)。
* **`assignment5/`**：课程作业 5。
* **`assignment6/`**：课程作业 6。
* **`slides/`**：课程核心幻灯片课件
  * `lec2_vae.pdf`: Variational Autoencoders (VAE)
  * `lec3_ar.pdf`: Autoregressive Models (AR)
  * `lec4_gan.pdf`: Generative Adversarial Networks (GAN)
  * `lec5_diffusion.pdf`: Diffusion Models

## 🚀 运行环境与使用方法

所有作业主要依赖 `Jupyter Notebook` 和 `PyTorch` 进行实现。

1. **克隆仓库**
   ```bash
   git clone https://github.com/DeepforThink/mit-6.S978-deep-generative-models.git
   cd mit-6.S978-deep-generative-models
   ```

2. **安装依赖**
   建议使用 Python 3.8+ 及以上版本，安装好 Jupyter 环境以及深度学习基础库：
   ```bash
   pip install torch torchvision numpy matplotlib jupyterlab
   ```

3. **运行代码**
   各作业文件夹下均包含 `.ipynb` 文件，请直接使用 Jupyter 启动：
   ```bash
   jupyter lab
   ```

## 📝 备注

* 仓库已配置 `.gitignore` 去除了原始数据集（如 MNIST `*.ubyte` 文件）以节省空间。若需重新运行完整流程，部分代码（如 `assignment1/download_images.py`）可用于自动下载数据集。
* 部分 Notebook 文件体积较大。
