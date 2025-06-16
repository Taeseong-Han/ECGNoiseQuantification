# Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection

Paper: [arXiv:2506.11815](https://arxiv.org/abs/2506.11815)

## 🔍 Overview

This repository contains the official implementation of  
**"Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection"**  
Han *et al.*, arXiv:2506.11815

We propose a label-free, reconstruction-based framework that quantifies ECG noise using diffusion models.  
Trained only on clean signals, the model reconstructs ECG scalograms and estimates noise severity via reconstruction errors, measured by PSNR and Wasserstein-1 distance ($W_1$).

**Key features:**
- ✅ Label-free anomaly detection without synthetic noise
- 🚀 Lightweight inference (3-step DDIM)
- 📊 Superior $W_1$ performance across PTB-XL, BUT QDB, CinC, NSTDB
- 🧠 Real-world clinical applications: arrhythmia detection, long-term ECG monitoring

## 🖼️ Framework Overview

![Framework Overview](figures/Framework_overview.jpg)

## 🧪 Example Usage

## 📄License and Citation

The software is licensed under the Apache License 2.0.  
Please cite the following paper if you use this code:

```bibtex
@misc{han2025diffusionbasedelectrocardiographynoisequantification,
  title={Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection}, 
  author={Tae-Seong Han and Jae-Wook Heo and Hakseung Kim and Cheol-Hui Lee and Hyub Huh and Eue-Keun Choi and Dong-Joo Kim},
  year={2025},
  eprint={2506.11815},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2506.11815}
}
