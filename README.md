![intro](./assets/intro.png)
<h1 align="center">AutoScaler</h1>

<p align="center">
  <strong>[PR 2026] AutoScaler: Self Scale Alignment for Handwritten Mathematical Expression Recognition</strong><br>
</p>
<p align="center">
  <a href="https://github.com/W-Geong">
    <img src="https://img.shields.io/badge/GitHub-WGeong-blueviolet?logo=github" alt="GitHub WGeong">
  </a>
  <a href="https://github.com/HCIILAB">
    <img src="https://img.shields.io/badge/GitHub-DLVC-success?logo=github" alt="GitHub DLVC">
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python 3.10">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-1.8.1-yellowgreen" alt="PyTorch 1.8.1">
  </a>
  <a href="https://www.pytorchlightning.ai/">
    <img src="https://img.shields.io/badge/PyTorch%20Lightning-1.4.9-orange" alt="PyTorch Lightning 1.4.9">
  </a>
</p>


![overall](./assets/main.png)

## 🛠️ Installation

* Clone this repo:
```bash
git clone https://github.com/SCUT-DLVCLab/AutoScaler.git
cd AutoScaler
```

* Create a virtual environment:
```bash
conda create -n autoscaler python=3.10 -y
conda activate autoscaler
```

* Install the required packages:
```bash
pip install -r requirements.txt
```

## 🔥 Training
* Download and process the dataset into the `dataset` folder.
* Generate the corresponding dictionary file `dic.txt`.
* Set the CUDA device in `train.py` and run 🚀:
```bash
cd AutoScaler
python train.py
```

## 🧠 Inference
* Set the checkpoint path in `test.py` and run:
```bash
python test.py
```
## 💐 Acknowledgements
This repo is modified based on [BTTR](https://github.com/Green-Wood/BTTR). Special thanks for their contributions to the community.
## 🧰 Other Awesome Repos
- [WAP](https://github.com/JianshuZhang/WAP) [![PR Badge](https://img.shields.io/badge/PR-2017-brightgreen)](https://www.sciencedirect.com/science/article/abs/pii/S0031320317302376)

- [DWAP-TD](https://github.com/JianshuZhang/TreeDecoder) [![ICML Badge](https://img.shields.io/badge/ICML-2020-green)](https://proceedings.mlr.press/v119/zhang20g.html)

- [BTTR](https://github.com/Green-Wood/BTTR) [![ICDAR Badge](https://img.shields.io/badge/ICDAR-2021-yellowgreen)](https://link.springer.com/chapter/10.1007/978-3-030-86331-9_37)

- [TSDNet](https://github.com/zshhans/TSDNet) [![ACM Badge](https://img.shields.io/badge/ACM_MM-2022-yellow)](https://dl.acm.org/doi/10.1145/3503161.3548424)

- [ABM](https://github.com/XH-B/ABM) [![AAAI Badge](https://img.shields.io/badge/AAAI-2022-yellow)](https://ojs.aaai.org/index.php/AAAI/article/view/19885)

- [SAN](https://github.com/tal-tech/SAN) [![CVPR Badge](https://img.shields.io/badge/CVPR-2022-orange)](https://openaccess.thecvf.com/content/CVPR2022/html/Yuan_Syntax-Aware_Network_for_Handwritten_Mathematical_Expression_Recognition_CVPR_2022_paper.html)

- [CoMER](https://github.com/Green-Wood/CoMER) [![ECCV Badge](https://img.shields.io/badge/ECCV-2022-red)](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_23)

- [CAN](https://github.com/LBH1024/CAN) [![ECCV Badge](https://img.shields.io/badge/ECCV-2022-blue)](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_12)

- [PosFormer](https://github.com/SJTU-DeepVisionLab/PosFormer) [![ECCV Badge](https://img.shields.io/badge/ECCV-2024-yellow)](https://link.springer.com/chapter/10.1007/978-3-031-72670-5_8)

- [TAMER](https://github.com/qingzhenduyu/TAMER) [![AAAI Badge](https://img.shields.io/badge/AAAI-2025-yellow)](https://ojs.aaai.org/index.php/AAAI/article/view/33190)

## 📜License
The code should be used and distributed under [ (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial research purposes.

## 📧Contacts

If you have any questions, feel free to contact the author at [wente_young@foxmail.com](wente_young@foxmail.com).
