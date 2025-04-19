# Behavior Importance-Aware Graph Neural Architecture Search for Cross-Domain Recommendation

<p align="center">
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/33274">
    <img src="https://img.shields.io/badge/AAAI_2025-Paper-blue?style=for-the-badge&logo=readthedocs" alt="AAAI 2025 Paper">
  </a>
</p>

Official implementation of our **AAAI 2025 Oral** paper:  
**"Behavior Importance-Aware Graph Neural Architecture Search for Cross-Domain Recommendation"**  
Feel free to star ⭐ this repo if you find it helpful!

---

## 🚀 Quick Start

### 1. Create Environment

```bash
conda create -n bignas python=3.10 -y
conda activate bignas
```

### 2. Install PyTorch with CUDA

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 3. Install PyTorch Geometric

```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
     -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

### 4. Install Other Requirements

```bash
pip install -r requirements.txt
```

---

## 🧪 Run Experiments

We provide an example of how to run the full BiGNAS training and evaluation pipeline.  
Please simply run:

```bash
bash run.sh
```

The dataset will be automatically downloaded.  
You can easily replace the `--categories` and `--target` arguments in the script to reproduce results for other task pairs.

---

## 📄 Citation

If you use this code or find our work useful, please cite:

```bibtex
@article{ge2025behavior,
  title   = {Behavior Importance-Aware Graph Neural Architecture Search for Cross-Domain Recommendation},
  author  = {Ge, Chendi and Wang, Xin and Zhang, Ziwei and Qin, Yijian and Chen, Hong and Wu, Haiyang and Zhang, Yang and Yang, Yuekui and Zhu, Wenwu},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {39},
  number  = {11},
  pages   = {11708--11716},
  year    = {2025}
}
```

---

## 📬 Contact

If you have any questions, feel free to open an issue or contact the first author at `gcd23@mails.tsinghua.edu.cn`.

---

## 🪪 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.