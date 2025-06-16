# Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection

## 🔍 Overview

This repository contains the official implementation of  
**"Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection"**  
Han *et al.*, arXiv:2506.11815

We propose a label-free, reconstruction-based framework that quantifies ECG noise using diffusion models.  
Trained only on clean signals, the model reconstructs ECG scalograms and estimates noise severity based on reconstruction errors.

Reconstruction quality is measured using **Peak Signal-to-Noise Ratio (PSNR)**, providing a continuous estimate of local signal degradation. To assess model performance in distinguishing clean from noisy inputs—even under mislabeled or ambiguous conditions—we use the **Wasserstein-1 distance ($W_1$)** between reconstruction error distributions as a robust, distribution-level evaluation metric.

**Key features:**
- ✅ Label-free anomaly detection without synthetic noise
- 🚀 Lightweight inference (3-step DDIM)
- 📊 Superior $W_1$ performance across PTB-XL, BUT QDB, CinC, NSTDB
- 🧠 Real-world clinical applications: arrhythmia detection, long-term ECG monitoring

📄 Paper: [arXiv:2506.11815](https://arxiv.org/abs/2506.11815)

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
