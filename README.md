# Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection

This repository contains the official implementation of  
**"Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection"**  
[![arXiv](https://img.shields.io/badge/arXiv-2506.11815-b31b1b.svg)](https://arxiv.org/abs/2506.11815)


## 🔍 Introduction

This study introduces a diffusion-based framework for ECG noise quantification using reconstruction-based anomaly detection.  The model is trained solely on clean ECG signals, learning to reconstruct clean representations from potentially noisy inputs. **Reconstruction errors, measured by Peak Signal-to-Noise Ratio (PSNR), serve as a proxy for noise levels** without requiring explicit noise labels during inference.

To address label inconsistencies and improve generalizability, we adopt a **distributional evaluation** strategy using the **Wasserstein-1 distance ($W_1$)**.  
By comparing the reconstruction error distributions of clean and noise-labeled ECGs, the model can:
- Identify optimal architectures and sampling configurations
- Detect mislabeled samples
- Refine the training dataset through low-error selection

Our final model achieves robust noise quantification with only **three reverse diffusion steps**, enabling efficient and scalable deployment in real-world settings.

### 🖼️ Framework Overview
![Framework Overview](figures/Framework_overview.jpg)

## 🔄 Superlet Transform on PTB-XL Data

To convert raw PTB-XL ECG recordings into superlet scalograms, run the following command from the project root:

```bash
python -m preprocessing.superlet_transform_ptbxl --ptbxl_raw_path [PTBXL_RAW_PATH]
```

Replace [PTBXL_RAW_PATH] with the full path to your downloaded PTB-XL dataset.
Example:
```bash
--ptbxl_raw_path ~/Database/physionet.org/files/ptb-xl/1.0.3
```

📥 Download PTB-XL Dataset
You can download the raw PTB-XL dataset from PhysioNet:

🔗 https://physionet.org/content/ptb-xl/1.0.3/


## 🧪 Example Usage

## 📄 License and Citation

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
```