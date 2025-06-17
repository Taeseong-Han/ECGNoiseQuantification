# Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection

This repository contains the official implementation of  
**"Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection"**  
[![arXiv](https://img.shields.io/badge/arXiv-2506.11815-b31b1b.svg)](https://arxiv.org/abs/2506.11815)

## 🔍 Introduction

This study introduces a diffusion-based framework for ECG noise quantification using reconstruction-based anomaly
detection. The model is trained solely on clean ECG signals, learning to reconstruct clean representations from
potentially noisy inputs. **Reconstruction errors, measured by Peak Signal-to-Noise Ratio (PSNR), serve as a proxy for
noise levels** without requiring explicit noise labels during inference.

To address label inconsistencies and improve generalizability, we adopt a **distributional evaluation** strategy using
the **Wasserstein-1 distance ($W_1$)**.  
By comparing the reconstruction error distributions of clean and noise-labeled ECGs, the model can:

- Identify optimal architectures and sampling configurations
- Detect mislabeled samples
- Refine the training dataset through low-error selection

Our final model achieves robust noise quantification with only **three reverse diffusion steps**, enabling efficient and
scalable deployment in real-world settings.

<br>

#### 🖼️ Framework Overview

![Framework Overview](figures/Framework_overview.jpg)

<br>
<br>

## 🔄 Superlet Transform on PTB-XL Data

To convert raw PTB-XL ECG recordings into superlet scalograms, run the following command from the project root:

```bash
python -m preprocessing.superlet_transform_ptbxl \
  --ptbxl_raw_path [PTBXL_RAW_PATH]
```

Replace [PTBXL_RAW_PATH] with the full path to your downloaded PTB-XL dataset.

Example:

```bash
--ptbxl_raw_path ~/Database/physionet.org/files/ptb-xl/1.0.3
```

📥 **Download PTB-XL Dataset**

You can download the raw PTB-XL dataset from PhysioNet:

🔗 https://physionet.org/content/ptb-xl/1.0.3/

<br>
<br>

## 🏋️Train Model

To train the autoencoder on the superlet-transformed PTB-XL dataset, run the following command from the project root:

```bash
python -m train.train_autoencoder
```

This will launch training with default settings using discretized superlet scalograms. You can customize training using
additional arguments.
This autoencoder acts as the feature compressor for the latent diffusion model and must be trained before training the
latent diffusion model.

> ⚠️ The training script structure is unified across this project.  
> The same CLI pattern applies to training:
>
> - `autoencoder`
> - `latent_diffusion`
> - `vanilla_diffusion`
>
> Simply execute:
>
> ```bash
> python -m train.[script_name] [options]
> ```
> View all available arguments using:
> ```bash
> python -m train.[script_name] --help
> ```

<br>
<br>

## 🧪 Example Usage

### 🔗 Pretrained Model

You can download the pretrained latent diffusion model from 🤗 Hugging Face:

👉 [Download pretrained model](https://huggingface.co/Taeseong-Han/ECGNoiseQuantification/blob/main/pretrained_ldm.pth)

### 💻 Inference Example

To run ECG noise quantification,
see [demo.ipynb](https://github.com/Taeseong-Han/ECGNoiseQuantification/blob/main/demo.ipynb) or use the following code
snippet:
> The input ECG is automatically segmented into 10-second windows, each of which is converted into a time-frequency
> representation via superlet transform. These scalograms are then fed into the pretrained diffusion model for
> reconstruction-based anomaly detection.
>
>The output includes segment-level original and denoised scalograms, along with the corresponding PSNR (Peak
> Signal-to-Noise Ratio) values for each segment. A higher PSNR indicates better signal quality (i.e., lower noise level).

```python
from utils.inference import ecg_noise_quantification

checkpoint_path = "[YOUR_PATH]/pretrained_ldm.pth"

output = ecg_noise_quantification(
    ecg=ecg,  # numpy array of shape (leads, timepoints)
    sampling_freq=500,  # sampling frequency in Hz
    checkpoint_path=checkpoint_path,
)

output.original_image: np.ndarray  # shape: (leads, segments, H, W)
output.cleaned_image: np.ndarray  # shape: (leads, segments, H, W)
output.psnr: np.ndarray  # shape: (leads, segments)
```

<br>
<br>

## 📄 License and Citation

The software is licensed under the MIT License 2.0.  
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