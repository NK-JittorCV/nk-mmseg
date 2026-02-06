# <p align=center>`DFormer for RGBD Semantic Segmentation (Jittor Implementation)`</p>
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
<!-- <p align="center">
    <img src="https://img.shields.io/badge/Framework-Jittor-brightgreen" alt="Framework">
    <img src="https://img.shields.io/badge/License-Non--Commercial-red" alt="License">
    <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
</p> -->

This is the Jittor implementation of DFormer and DFormerv2 for RGBD semantic segmentation. Developed based on the Jittor deep learning framework, it provides efficient solutions for training and inference.

This repository contains the official Jittor implementation of the following papers:

> DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation<br/>
> [Bowen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=en),
> [Xuying Zhang](https://scholar.google.com/citations?hl=en&user=huWpVyEAAAAJ),
> [Zhongyu Li](https://scholar.google.com/citations?user=g6WHXrgAAAAJ&hl=en),
> [Li Liu](https://scholar.google.com/citations?hl=en&user=9cMQrVsAAAAJ),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=en&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=en) <br/>
> ICLR 2024. 
>[Paper Link](https://arxiv.org/abs/2309.09668) |
>[Homepage](https://yinbow.github.io/Projects/DFormer/index.html) |
>[PyTorch Version](https://github.com/VCIP-RGBD/DFormer)

> DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation<br/>
> [Bo-Wen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=en),
> [Jiao-Long Cao](https://github.com/caojiaolong),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=en&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=en)<br/>
> CVPR 2025. 
> [Paper Link](https://arxiv.org/abs/2504.04701) |
> [Chinese Version](https://mftp.mmcheng.net/Papers/25CVPR_RGBDSeg-CN.pdf) |
> [PyTorch Version](https://github.com/VCIP-RGBD/DFormer)

---

## <p align="center">✨ About the Jittor Framework: An Architectural Deep Dive ✨</p>

This project is built upon [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/), a cutting-edge deep learning framework that pioneers a design centered around **Just-In-Time (JIT) compilation** and **meta-operators**. This architecture provides a unique combination of high performance and exceptional flexibility. Instead of relying on static, pre-compiled libraries, Jittor operates as a dynamic, programmable system that compiles itself and the user's code on the fly.

### The Core Philosophy: From Static Library to Dynamic Compiler

Jittor's design philosophy is to treat the deep learning framework not as a fixed set of tools, but as a domain-specific compiler. The high-level Python code written by the user serves as a directive to this compiler, which then generates highly optimized, hardware-specific machine code at runtime. This approach unlocks a level of performance and flexibility that is difficult to achieve with traditional frameworks.

### Key Innovations of Jittor

*   **A Truly Just-in-Time (JIT) Compiled Framework**:
    > Jittor's most significant innovation is that **the entire framework is JIT compiled**. This goes beyond merely compiling a static computation graph. When a Jittor program runs, the Python code, including the core framework logic and the user's model, is first parsed into an intermediate representation. The Jittor compiler then performs a series of advanced optimizations—such as operator fusion, memory layout optimization, and dead code elimination—before generating and executing native C++ or CUDA code. This "whole-program" compilation approach means that the framework can adapt to the specific logic of your model, enabling optimizations that are impossible when linking against a static, pre-compiled library.

*   **Meta-Operators and Dynamic Kernel Fusion**:
    > At the heart of Jittor lies the concept of **meta-operators**. These are not monolithic, pre-written kernels (like in other frameworks), but rather elementary building blocks defined in Python. For instance, a complex operation like `Conv2d` followed by `ReLU` is not two separate kernel calls. Instead, Jittor composes them from meta-operators, and its JIT compiler **fuses** them into a single, efficient CUDA kernel at runtime. This **kernel fusion** is critical for performance on modern accelerators like GPUs, as it drastically reduces the time spent on high-latency memory I/O and kernel launch overhead, which are often the primary bottlenecks.

*   **The Unified Computation Graph: Flexibility Meets Performance**:
    > Jittor elegantly resolves the classic trade-off between the flexibility of dynamic graphs (like PyTorch) and the performance of static graphs (like TensorFlow 1.x). You can write your model using all the native features of Python, including complex control flow like `if/else` statements and data-dependent `for` loops. Jittor's compiler traces these dynamic execution paths and still constructs a graph representation that it can optimize globally. It achieves this by JIT-compiling different graph versions for different execution paths, thus preserving Python's expressiveness without sacrificing optimization potential.

*   **Decoupling of Frontend Logic and Backend Optimization**:
    > Jittor champions a clean separation that empowers researchers. You focus on the "what"—the mathematical logic of your model—using a clean, high-level Python API. Jittor's backend automatically handles the "how"—the complex task of writing high-performance, hardware-specific code. This frees researchers who are experts in their domain (e.g., computer vision) from needing to become experts in low-level GPU programming, thus accelerating the pace of innovation.

---
## 🚩 Performance
<p align="center">
    <img src="figs/Figure_1.png" width="600"  width="1200"/> <br />
    <em> 
    Chart 1: Comparison of mIoU changes between Jittor implementation and Pytorch implementation of Dformer-Large.
    </em>
</p>
<p align="center">
    <img src="figs/latency_comparison.png" width="600"  width="1200"/> <br />
    <em> 
    Chart 2: Comparisons of lantency between Jittor implementation and Pytorch implementation
    </em>
</p>
<p align="center">
    <img src="figs/multi_model_eval_comparison.png" width="600"  width="1200"/> <br />
    <em> 
    Chart 3: Comparisons of evaluation time between Jittor implementation and Pytorch implementation
    </em>
</p>
<p align="center">
    <img src="figs/multi_model_scaling_analysis.png" width="600"  width="1200"/> <br />
    <em> 
    Chart 4: Comparisons of Model Size and Evaluation Time between Jittor implementation and Pytorch implementation
    </em>
</p>

## 🚀 Getting Started

### Environment Setup

```bash
# Create a conda environment
conda create -n dformer_jittor python=3.8 -y
conda activate dformer_jittor

# Install Jittor
pip install jittor

# Install other dependencies
pip install opencv-python pillow numpy scipy tqdm tensorboardX tabulate easydict
```

### Dataset Preparation

Supported datasets:
- **NYUDepthv2**: An indoor RGBD semantic segmentation dataset.
- **SUNRGBD**: A large-scale dataset for indoor scene understanding.

Download links:
| Dataset | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) | 
|:---: |:---:|:---:|:---:|

### Pre-trained Models

| Model | Dataset | mIoU | Download Link |
|------|--------|------|----------|
| DFormer-Small | NYUDepthv2 | 52.3 | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn) |
| DFormer-Base | NYUDepthv2 | 54.1 | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn) |
| DFormer-Large | NYUDepthv2 | 55.8 | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn) |
| DFormerv2-Small | NYUDepthv2 | 53.7 | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym) |
| DFormerv2-Base | NYUDepthv2 | 55.3 | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym) |
| DFormerv2-Large | NYUDepthv2 | 57.1 | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym) |

### Directory Structure

```
DFormer-Jittor/
├── checkpoints/              # Directory for pre-trained models
│   ├── pretrained/          # ImageNet pre-trained models
│   └── trained/             # Trained models
├── datasets/                # Directory for datasets
│   ├── NYUDepthv2/         # NYU dataset
│   └── SUNRGBD/            # SUNRGBD dataset
├── local_configs/          # Configuration files
├── models/                 # Model definitions
├── utils/                  # Utility functions
├── train.sh               # Training script
├── eval.sh                # Evaluation script
└── infer.sh               # Inference script
```

## 📖 Usage

### Training

Use the provided training script:
```bash
bash train.sh
```

Or use the Python command directly:
```bash
python utils/train.py --config local_configs.NYUDepthv2.DFormer_Base
```

### Evaluation

```bash
bash eval.sh
```

Alternatively:
```bash
python utils/eval.py --config local_configs.NYUDepthv2.DFormer_Base --checkpoint checkpoints/trained/NYUDepthv2/DFormer_Base/best.pkl
```

### Inference/Visualization

```bash
bash infer.sh
```

## 🚩 Performance

<p align="center">
    <img src="figs/Semseg.jpg" width="600"  width="1200"/> <br />
    <em> 
    Table 1: Comparisons between the existing methods and our DFormer.
    </em>
</p>

<p align="center">
    <img src="figs/dformerv2_table.jpg" width="600"  width="1200"/> <br />
    <em> 
    Table 2: Comparisons between the existing methods and our DFormerv2.
    </em>
</p>

## 🔧 Configuration

The project uses Python configuration files located in the `local_configs/` directory:

```python
# local_configs/NYUDepthv2/DFormer_Base.py
class C:
    # Dataset configuration
    dataset_name = "NYUDepthv2"
    dataset_dir = "datasets/NYUDepthv2"
    num_classes = 40
    
    # Model configuration
    backbone = "DFormer_Base"
    pretrained_model = "checkpoints/pretrained/DFormer_Base.pth"
    
    # Training configuration
    batch_size = 8
    nepochs = 500
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0001
    
    # Other configurations
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
```

## 📊 Benchmarking

### FLOPs and Parameters

```bash
python benchmark.py --config local_configs.NYUDepthv2.DFormer_Base
```

### Inference Speed

```bash
python utils/latency.py --config local_configs.NYUDepthv2.DFormer_Base
```
## ⚠️ Note

### Root Cause of the Issue

**What is CUTLASS?**  
CUTLASS (CUDA Templates for Linear Algebra Subroutines) is a high-performance CUDA matrix operation template library launched by NVIDIA, primarily used for efficiently implementing core operators like GEMM/Conv on Tensor Cores. It is utilized by many frameworks (Jittor, PyTorch XLA, TVM, etc.) for custom operators or as a low-level acceleration for Auto-Tuning.

**Why does Jittor pull CUTLASS in cuDNN unit tests?**  
When Jittor loads/compiles external CUDA libraries, it automatically compiles several custom operators from CUTLASS (setup_cutlass()). If the local cache is missing, it will call install_cutlass() to download and extract a cutlass.zip.

### Direct Cause of the Crash

The install_cutlass() function in version 1.3.9.14 uses a download link that has become invalid (confirmed by community Issue #642).  
After the download fails, a partial ~/.cache/jittor/cutlass directory is left behind; when running the function again, it attempts to execute shutil.rmtree('.../cutlass/cutlass'), but this subdirectory does not exist, triggering a FileNotFoundError and ultimately causing the main process to core dump.

### 解决方案 (按推荐顺序选择其一)

| 方案 | 操作步骤 | 适用场景 |
|------|---------|----------|
| **1️⃣ 临时跳过 CUTLASS** | ```bash<br># 仅对当前 shell 生效<br>export use_cutlass=0<br>python3.8 -m jittor.test.test_cudnn_op<br>``` | 只想先跑通 cuDNN 单测 / 不需要 CUTLASS 算子 |
| **2️⃣ 手动安装 CUTLASS** | ```bash<br># 清理残留<br>rm -rf ~/.cache/jittor/cutlass<br><br># 手动克隆最新版<br>mkdir -p ~/.cache/jittor/cutlass && \<br>cd ~/.cache/jittor/cutlass && \<br>git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass<br><br># 再次运行<br>python3.8 -m jittor.test.test_cudnn_op<br>``` | 仍想保留 CUTLASS 相关算子功能 |
| **3️⃣ 升级 Jittor 至修复版本** | ```bash<br>pip install -U jittor jittor-utils<br>```<br><br>社区 1.3.9.15+ 已把失效链接改到镜像源，升级后即可自动重新下载。 | 允许升级环境并希望后续自动管理 |

## 🤝 Contributing

We welcome all forms of contributions:

1. **Bug Reports**: Report issues in GitHub Issues.
2. **Feature Requests**: Suggest new features.
3. **Code Contributions**: Submit Pull Requests.
4. **Documentation Improvements**: Improve README and code comments.


## 📞 Contact

If you have any questions about our work, feel free to contact us:

- Email: bowenyin@mail.nankai.edu.cn, caojiaolong@mail.nankai.edu.cn
- GitHub Issues: [Submit an issue](https://github.com/VCIP-RGBD/DFormer-Jittor/issues)

## 📚 Citation

If you use our work in your research, please cite the following papers:

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

## 🙏 Acknowledgements

Our implementation is mainly based on the following open-source projects:

- [Jittor](https://github.com/Jittor/jittor): A deep learning framework.
- [DFormer](https://github.com/VCIP-RGBD/DFormer): The original PyTorch implementat
ion.
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation): A semantic segmentation toolbox.

Thanks to all the contributors for their efforts!

## 📄 License

This project is for non-commercial use only. See the [LICENSE](LICENSE) file for details.

---

