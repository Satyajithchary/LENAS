<img width="2213" height="582" alt="LENAS drawio (1) (1)" src="https://github.com/user-attachments/assets/5a8e2103-6bc2-42fc-ad47-ee54723363ac" /># LENAS

This repository provides the official implementation of **LENAS: Learning from Explainable and Navigated Attention for Annotation-Free Medical Image Segmentation**.

## Key Features

- **Annotation-Free Segmentation**: Achieves high-fidelity segmentation without pixel-level supervision or manual bounding boxes, relying solely on image-level labels.
- **Differential BiomedCLIP**: A novel backbone that explicitly models pathology-context separation to isolate subtle pathological features from background noise.
- **Navigated Attention**: Transforms Explainable AI (XAI) maps into geometric spatial priors to automatically prompt the Segment Anything Model (SAM).
- **Self-Correcting Mechanism**: Features an iterative loop with a confidence-based reward system and Snake-based contour optimization to refine masks in real-time.

## Details

Universal foundation models for pathological segmentation often suffer from a profound gap between visual recognition and medical understanding. While models like SAM achieve robust performance on natural images, medical imaging is constrained by scarce expert annotations and subtle lesion variations.

In this work, we propose **LENAS** (Learning from Explainable and Navigated Attention for Annotation-Free Medical Image Segmentation). LENAS provides annotation-free segmentation as a unified vision–language challenge solved using navigated attention. The images are applied through navigated attention that guides the model’s focus to pathology-relevant regions under minimal supervision.

Our key contributions are:
1. A new **Differential BiomedCLIP** architecture whose differential attention layers excel at isolating subtle pathological features.
2. An **XAI-to-prompt pipeline** that translates classifier-derived representations into spatial priors for prompt-efficient segmentation with SAM.
3. An **iterative loop of self-evaluation** for classifier’s confidence on the resulting mask which serves as a real-time reward signal for prompt refinement.

<div align="center">
    <img width="1000px" height="auto" src="assets/LENAS.drawio (1) (1).png">
</div>

**Differential BiomedCLIP Backbone**

<div align="center">
    <img width="800px" height="auto" src="assets/Differential_BiomedCLIP (1).png">
</div>

## Quantitative Results

LENAS achieves state-of-the-art performance among annotation-free and low-label baselines. Below is the comparison on **Kvasir-SEG** (Polyp) and **BUSI** (Breast Ultrasound) datasets.

| Method | Kvasir-SEG (Dice) | Kvasir-SEG (IoU) | BUSI (Dice) | BUSI (IoU) |
| :--- | :---: | :---: | :---: | :---: |
| GradCAM | 0.1034 | 0.0600 | -- | -- |
| ScribbleUNet | 0.2522 | 0.1565 | -- | -- |
| U-Net (1% labels) | 0.1760 | 0.2760 | 0.0019 | 0.0010 |
| U-Net (5% labels) | 0.1760 | 0.2760 | 0.1745 | 0.1076 |
| **LENAS (Ours)** | **0.3459** | **0.2612** | **0.3491** | **0.2731** |

## Qualitative Results

**Visualizations of pathological lesion localization and segmentation**

The figure below demonstrates the LENAS pipeline: generating fused saliency maps, deriving initial SAM masks, applying Snake refinement, and producing the final prediction.

<div align="center">
    <img width="1000px" height="auto" src="assets/fig5_results.png">
</div>

## Get started

**Installation**

```shell
# create a new conda environment
conda create -n LENAS python=3.9
conda activate LENAS

# install torch (adjust cuda version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirements.txt
