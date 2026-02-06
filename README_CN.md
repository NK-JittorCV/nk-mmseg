
# <p align=center>`DFormer for RGBD Semantic Segmentation (Jittor 实现)`</p>
<p align="center">
    <br>
    <img src="figs/logo_2.png"/>
    <br>
<p>
<p align="center">
<a href="https://github.com/VCIP-RGBD/DFormer-Jittor">项目主页</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbsp<a href="README.md">English&nbsp
</p>
<p align="center">
<img src="https://img.shields.io/badge/python-3.8+-blue.svg">
<img src="https://img.shields.io/badge/jittor-1.3.9+-orange.svg">
<a href="https://github.com/VCIP-RGBD/DFormer-Jittor/blob/master/LICENSE"><img src="https://img.shields.io/github/license/VCIP-RGBD/DFormer-Jittor"></a>
<a href="https://github.com/VCIP-RGBD/DFormer-Jittor/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>

这是 DFormer 和 DFormerv2 用于 RGBD 语义分割的 Jittor 实现。该项目基于 Jittor 深度学习框架开发，为训练和推理提供了高效的解决方案。

本仓库包含以下论文的官方 Jittor 实现：

> DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation<br/>
> [Bowen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=en),
> [Xuying Zhang](https://scholar.google.com/citations?hl=en&user=huWpVyEAAAAJ),
> [Zhongyu Li](https://scholar.google.com/citations?user=g6WHXrgAAAAJ&hl=en),
> [Li Liu](https://scholar.google.com/citations?hl=en&user=9cMQrVsAAAAJ),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=en&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=en) <br/>
> ICLR 2024.
>[论文链接](https://arxiv.org/abs/2309.09668) |
>[项目主页](https://yinbow.github.io/Projects/DFormer/index.html) |
>[PyTorch 版本](https://github.com/VCIP-RGBD/DFormer)

> DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation<br/>
> [Bo-Wen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=en),
> [Jiao-Long Cao](https://github.com/caojiaolong),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=en&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=en)<br/>
> CVPR 2025.
> [论文链接](https://arxiv.org/abs/2504.04701) |
> [中文版](https://mftp.mmcheng.net/Papers/25CVPR_RGBDSeg-CN.pdf) |
> [PyTorch 版本](https://github.com/VCIP-RGBD/DFormer)

---

## <p align="center">✨ 关于计图 (Jittor) 框架：架构深度解析 ✨</p>

本项目基于[计图 (Jittor)](https://cg.cs.tsinghua.edu.cn/jittor/) 构建，这是一个前沿的深度学习框架，其设计核心在于**即时 (Just-In-Time, JIT) 编译**与**元算子**。这种架构提供了高性能与卓越灵活性的独特结合。计图并非依赖于静态的预编译库，而是作为一个动态、可编程的系统运行，能够即时编译自身及用户的代码。

### 核心理念：从静态库到动态编译器

计图的设计哲学是将深度学习框架视为一个领域特定编译器，而非一套固定的工具集。用户编写的高级 Python 代码作为指令，驱动这个编译器在运行时生成高度优化的、针对特定硬件的机器码。这种方法解锁了传统框架难以企及的性能与灵活性。

### 计图的关键创新

*   **真正意义上的即时 (JIT) 编译框架**:
    > 计图最显著的创新在于**整个框架都是 JIT 编译的**。这超越了仅仅编译静态计算图的范畴。当计图程序运行时，包括核心框架逻辑和用户模型在内的 Python 代码，首先被解析成一个中间表示。随后，计图编译器执行一系列高级优化——例如算子融合、内存布局优化和死代码消除——最终生成并执行原生的 C++ 或 CUDA 代码。这种“全程序”编译方法意味着框架能够适应您模型的具体逻辑，从而实现静态预编译库无法做到的优化。

*   **元算子与动态核函数融合**:
    > 计图的核心是**元算子**的概念。它们并非像其他框架中那样是庞大、预先写好的核函数，而是在 Python 中定义的基本构建模块。例如，一个像 `Conv2d` 后接 `ReLU` 这样的复杂操作，并非两次独立的核函数调用。相反，计图从元算子出发构建它们，并通过其 JIT 编译器在运行时将它们**融合**成一个单一、高效的 CUDA 核函数。这种**核函数融合**对于在现代加速器（如 GPU）上获得高性能至关重要，因为它极大地减少了高延迟的内存 I/O 和核函数启动开销所耗费的时间，而这些往往是性能的主要瓶颈。

*   **统一计算图：灵活性与性能的结合**:
    > 计图优雅地解决了动态图（如 PyTorch）的灵活性与静态图（如 TensorFlow 1.x）的性能之间的经典权衡。您可以使用 Python 的所有原生特性来编写模型，包括复杂的控制流（如 `if/else` 语句）和数据依赖的 `for` 循环。计图的编译器会追踪这些动态执行路径，并仍然构建一个可以进行全局优化的图表示。它通过为不同的执行路径 JIT 编译不同版本的图来实现这一点，从而在不牺牲优化潜力的情况下保留了 Python 的表达能力。

*   **前端逻辑与后端优化的解耦**:
    > 计图倡导一种清晰的分离，从而赋能研究人员。您只需关注“做什么”——即您模型的数学逻辑——使用一个简洁、高级的 Python API。计图的后端则自动处理“如何做”——即编写高性能、针对特定硬件代码的复杂任务。这使得在各自领域（如计算机视觉）的专家研究人员无需再成为底层 GPU 编程的专家，从而加快了创新的步伐。

---
## 🚩 性能
<p align="center">
    <img src="figs/Figure_1.png" width="600"  width="1200"/> <br />
    <em>
    图 1: Dformer-Large 的 Jittor 实现与 Pytorch 实现的 mIoU 变化对比。
    </em>
</p>
<p align="center">
    <img src="figs/latency_comparison.png" width="600"  width="1200"/> <br />
    <em>
    图 2: Jittor 实现与 Pytorch 实现的延迟对比
    </em>
</p>

## 🚀 快速开始

### 环境设置

```bash
# 创建一个 conda 环境
conda create -n dformer_jittor python=3.8 -y
conda activate dformer_jittor

# 安装 Jittor
pip install jittor

# 安装其他依赖
pip install opencv-python pillow numpy scipy tqdm tensorboardX tabulate easydict
```

### 数据集准备

支持的数据集：
- **NYUDepthv2**: 一个室内的 RGBD 语义分割数据集。
- **SUNRGBD**: 一个用于室内场景理解的大规模数据集。

下载链接：
| Dataset | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) |
|:---: |:---:|:---:|:---:|

### 预训练模型

| Model | Dataset | mIoU | Download Link |
|------|--------|------|----------|
| DFormer-Small | NYUDepthv2 | 52.3 | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn) |
| DFormer-Base | NYUDepthv2 | 54.1 | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn) |
| DFormer-Large | NYUDepthv2 | 55.8 | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn) |
| DFormerv2-Small | NYUDepthv2 | 53.7 | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym) |
| DFormerv2-Base | NYUDepthv2 | 55.3 | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym) |
| DFormerv2-Large | NYUDepthv2 | 57.1 | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym) |

### 目录结构

```
DFormer-Jittor/
├── checkpoints/              # 预训练模型目录
│   ├── pretrained/          # ImageNet 预训练模型
│   └── trained/             # 已训练模型
├── datasets/                # 数据集目录
│   ├── NYUDepthv2/         # NYU 数据集
│   └── SUNRGBD/            # SUNRGBD 数据集
├── local_configs/          # 配置文件
├── models/                 # 模型定义
├── utils/                  # 工具函数
├── train.sh               # 训练脚本
├── eval.sh                # 评估脚本
└── infer.sh               # 推理脚本
```

## 📖 使用说明

### 训练

使用提供的训练脚本：
```bash
bash train.sh
```

或者直接使用 Python 命令：
```bash
python utils/train.py --config local_configs.NYUDepthv2.DFormer_Base
```

### 评估

```bash
bash eval.sh
```

或者：
```bash
python utils/eval.py --config local_configs.NYUDepthv2.DFormer_Base --checkpoint checkpoints/trained/NYUDepthv2/DFormer_Base/best.pkl
```

### 推理/可视化

```bash
bash infer.sh
```

## 🚩 性能

<p align="center">
    <img src="figs/Semseg.jpg" width="600"  width="1200"/> <br />
    <em>
    表 1: 现有方法与我们的 DFormer 的比较。
    </em>
</p>

<p align="center">
    <img src="figs/dformerv2_table.jpg" width="600"  width="1200"/> <br />
    <em>
    表 2: 现有方法与我们的 DFormerv2 的比较。
    </em>
</p>

## 🔧 配置

项目使用位于 `local_configs/` 目录下的 Python 配置文件：

```python
# local_configs/NYUDepthv2/DFormer_Base.py
class C:
    # 数据集配置
    dataset_name = "NYUDepthv2"
    dataset_dir = "datasets/NYUDepthv2"
    num_classes = 40

    # 模型配置
    backbone = "DFormer_Base"
    pretrained_model = "checkpoints/pretrained/DFormer_Base.pth"

    # 训练配置
    batch_size = 8
    nepochs = 500
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0001

    # 其他配置
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
```

## 📊 基准测试

### FLOPs 和参数

```bash
python benchmark.py --config local_configs.NYUDepthv2.DFormer_Base
```

### 推理速度

```bash
python utils/latency.py --config local_configs.NYUDepthv2.DFormer_Base
```
## ⚠️ 注意事项

### 问题根源

**CUTLASS 是什么？**
CUTLASS (CUDA Templates for Linear Algebra Subroutines) 是 NVIDIA 推出的一个高性能 CUDA 矩阵运算模板库，主要用于在 Tensor Core 上高效实现 GEMM/Conv 等核心算子。它被许多框架（Jittor、PyTorch XLA、TVM 等）用于自定义算子或作为自动调优的底层加速。

**为什么 Jittor 在 cuDNN 单元测试中会拉取 CUTLASS？**
当 Jittor 加载/编译外部 CUDA 库时，它会自动从 CUTLASS 编译几个自定义算子（setup_cutlass()）。如果本地缓存缺失，它会调用 install_cutlass() 下载并解压一个 cutlass.zip。

### 直接原因

版本 1.3.9.14 中的 install_cutlass() 函数使用了一个已失效的下载链接（社区 Issue #642 已确认）。
下载失败后，会留下一个不完整的 ~/.cache/jittor/cutlass 目录；再次运行该函数时，它会尝试执行 shutil.rmtree('.../cutlass/cutlass')，但这个子目录并不存在，从而触发 FileNotFoundError，最终导致主进程崩溃。

### 解决方案 (按推荐顺序选择其一)

| 方案 | 操作步骤 | 适用场景 |
|------|---------|----------|
| **1️⃣ 临时跳过 CUTLASS** | ```bash<br># 仅对当前 shell 生效<br>export use_cutlass=0<br>python3.8 -m jittor.test.test_cudnn_op<br>``` | 只想先跑通 cuDNN 单测 / 不需要 CUTLASS 算子 |
| **2️⃣ 手动安装 CUTLASS** | ```bash<br># 清理残留<br>rm -rf ~/.cache/jittor/cutlass<br><br># 手动克隆最新版<br>mkdir -p ~/.cache/jittor/cutlass && \<br>cd ~/.cache/jittor/cutlass && \<br>git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass<br><br># 再次运行<br>python3.8 -m jittor.test.test_cudnn_op<br>``` | 仍想保留 CUTLASS 相关算子功能 |
| **3️⃣ 升级 Jittor 至修复版本** | ```bash<br>pip install -U jittor jittor-utils<br>```<br><br>社区 1.3.9.15+ 已把失效链接改到镜像源，升级后即可自动重新下载。 | 允许升级环境并希望后续自动管理 |

## 🤝 贡献

我们欢迎所有形式的贡献：

1. **Bug 报告**: 在 GitHub Issues 中报告问题。
2. **功能请求**: 建议新功能。
3. **代码贡献**: 提交 Pull Requests。
4. **文档改进**: 改进 README 和代码注释。


## 📞 联系我们

如果对我们的工作有任何疑问，请随时联系我们：

- Email: bowenyin@mail.nankai.edu.cn, caojiaolong@mail.nankai.edu.cn
- GitHub Issues: [提交一个 issue](https://github.com/VCIP-RGBD/DFormer-Jittor/issues)

## 📚 引用

如果您在研究中使用了我们的工作，请引用以下论文：

```bibtex
@inproceedings{yin2024dformer,
  title={DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation},
  author={Yin, Bowen and Zhang, Xuying and Li, Zhong-Yu and Liu, Li and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{yin2025dformerv2,
  title={DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation},
  author={Yin, Bo-Wen and Cao, Jiao-Long and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19345--19355},
  year={2025}
}
```

## 🙏 致谢

我们的实现主要基于以下开源项目：

- [Jittor](https://github.com/Jittor/jittor): 一个深度学习框架。
- [DFormer](https://github.com/VCIP-RGBD/DFormer): 原始的 PyTorch 实现。
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation): 一个语义分割工具箱。

感谢所有贡献者的努力！

## 📄 许可

本项目仅供非商业用途。详情请参阅 [LICENSE](LICENSE) 文件。

--- 