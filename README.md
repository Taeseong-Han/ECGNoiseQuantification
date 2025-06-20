# Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection

This repository contains the official implementation of  
**"Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection"**  
[![arXiv](https://img.shields.io/badge/arXiv-2506.11815-b31b1b.svg)](https://arxiv.org/abs/2506.11815)

## 🔍 Introduction

This study introduces a diffusion-based framework for ECG noise quantification using reconstruction-based anomaly
detection. The model is trained solely on clean ECG signals, learning to reconstruct clean representations from
potentially noisy inputs. **Reconstruction errors serve as a proxy for noise levels** without requiring explicit noise
labels during inference.

- **Noise Quantification**: Peak Signal-to-Noise Ratio (**PSNR**), a robust metric for quantifying distortion in the
  time-frequency domain, is employed. PSNR is inversely proportional to noise levels:
    - **Lower PSNR** 📉💥 indicates **higher noise**.
    - **Higher PSNR** indicates **better signal quality**.

Our optimized model achieves robust noise quantification with only **three reverse diffusion steps**, ensuring efficient
and scalable real-world applicability.

### 🖼️ Limitations of Human Labels in ECG Noise Assessment

![External validation](figures/external_validation.jpg)

**Figure** demonstrates the practical utility of PSNR-based segment analysis and Wasserstein-1 distance (**$W_1$**)
-based evaluation (leveraging PSNR distributions):

- **(a)** Examples from the BUT QDB dataset highlight discrepancies where a segment labeled as "Clean" by humans
  exhibited a lower PSNR (higher noise) compared to another segment labeled "Moderate Noise." Specifically,
  noise-labeled samples with high PSNR appeared visually clean, whereas clean-labeled samples with low PSNR clearly
  exhibited noise artifacts.

- **(b)** Analysis of data from the **CinC Challenge 2011** dataset revealed significant overlap in PSNR distributions
  between human-labeled clean and noisy segments. Further visual inspection (Figure (b)) confirmed these overlapping
  segments indeed exhibited similar noise characteristics, underscoring the limitations of human annotations.

<br>

## 🧪 Example Usage

#### 🔗 Pretrained Model

You can download the pretrained latent diffusion model from 🤗 Hugging Face:

👉 [Download pretrained model](https://huggingface.co/Taeseong-Han/ECGNoiseQuantification/blob/main/pretrained_ldm.pth)

#### 💻 Inference Example

Higher PSNR values indicate better signal quality, corresponding to lower noise levels.
In practice, a PSNR threshold of approximately 24 effectively distinguishes severely degraded ECG segments from
acceptable ones.

see [demo.ipynb](https://github.com/Taeseong-Han/ECGNoiseQuantification/blob/main/demo.ipynb) or use the following code
snippet:

```python
from utils.inference import ecg_noise_quantification

checkpoint_path = "[YOUR_PATH]/pretrained_ldm.pth"

output = ecg_noise_quantification(
    ecg=ecg,  # numpy array of shape (leads, timepoints)
    sampling_freq=500,  # sampling frequency in Hz
    checkpoint_path=checkpoint_path,
    return_images=True,
)

output.psnr: np.ndarray  # shape: (leads, segments)
output.original_image: np.ndarray  # shape: (leads, segments, H, W)
output.cleaned_image: np.ndarray  # shape: (leads, segments, H, W)
```

> The input ECG is automatically segmented into 10-second windows, each of which is converted into a time-frequency
> representation via superlet transform. These scalograms are then fed into the pretrained diffusion model for
> reconstruction-based anomaly detection.
>
>The output includes segment-level original and denoised scalograms, along with the corresponding PSNR values for each
> segment.
>

<br>
<br>

## 🧬 Reproducibility: Data Preprocessing and Training

The following sections provide instructions for:

- Transforming raw PTB-XL ECG data using superlet transform
- Training the autoencoder used for latent diffusion
- Running full training pipelines for diffusion models

---

#### 🖼️ Framework Overview

![Framework Overview](figures/Framework_overview.jpg)

### 🔄 Superlet Transform on PTB-XL Data

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

### 🏋️Train Model

To train the autoencoder on the superlet-transformed PTB-XL dataset, run the following command from the project root:

```bash
python -m train.train_autoencoder --discretization
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

### 📊 Noise Level Quantification

Quantify ECG signal noise using a pretrained diffusion model. The model reconstructs clean
versions of input scalograms, and PSNR (or other metrics) is used to quantify the reconstruction error,
which serves as a proxy for noise.

```bash
python -m evaluation.run_noise_quantification \
  --checkpoint [YOUR MODEL PATH] \
  --timestep 250 \
  --noise_scheduler_type ddim \
  --step_interval 1 \
  --batch_size 32 \
  --discretize \
  --output_dir ./evaluation/results
```

If you wish to run quantification across all 10 folds (not just test set), add:

```bash
--include_all_folds
```

### 📈 Performance Evaluation Across Combinations

Compare multiple experiment results (e.g., different sampling settings or model types) 
by computing Wasserstein-1 distances between clean vs noisy segment distributions.

```bash
python -m evaluation.eval_models \
 --keyword --output_dir
```
**Options:**
- --keyword: Substring to filter result files (e.g., 'ddpm', 't250', etc.)
Output will display W₁ distances across noise types (e.g., clean vs. static, burst, baseline), 
enabling objective comparison across configurations.



### 🧹 Refining Dataset and Retraining Model

To improve training quality, we refine the dataset by selecting only high-confidence clean segments 
— based on reconstruction metrics, not human labels.

#### Step 1. Quantify Noise Across All Folds
Run noise quantification on the full PTB-XL dataset to evaluate every segment, 
including clean-labeled ones:

```bash
python -m evaluation.run_noise_quantification \
  --checkpoint ./checkpoints/pretrained_ldm.pth \
  --timestep 250 \
  --noise_scheduler_type ddpm \
  --include_all_folds \
  --discretize \
  --output_dir ./evaluation/results

```
This provides segment-level PSNR (or other metrics) used to reassess clean sample quality.

#### Step 2: Select High-Confidence Clean Segments and Retraining Model
Using the model that showed strong sensitivity to static and burst noise (high W₁-distance), 
extract the top-N% clean segments least likely to be mislabeled:

```bash
python -m train.retrain_autoencoder \
  --static_file ./evaluation/results/static_ddpm_t250_psnr.csv \
  --burst_file ./evaluation/results/burst_ddpm_t250_psnr.csv \
  --metric psnr \
  --static_percentage 0.5 \
  --burst_percentage 0.5 \
  --discretization \
  --save_path ./output/ae_model_refined

```
This refines the training set to include only segments that are consistently high-quality 
under both static and burst noise evaluations, enabling robust model retraining.

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