# 🧠 Neonatal Vision–Language and Diffusion Training Pipelines

This repository contains two complementary deep learning pipelines for neonatal chest X-ray research:

1. **BLIP Transformer Fine-Tuning** — for **gestational age (GA) prediction** and clinical text generation.  
2. **Stable Diffusion UNet Training** — for **text-conditioned neonatal X-ray synthesis** with mask-guided supervision.

---

## 🚼 **Code 1: `GA_prediction_using_medblip_finetuning.py`**

This script fine-tunes the **BLIP (Bootstrapped Language–Image Pretraining)** model — a **transformer-based vision–language framework** — for **gestational age (GA) prediction** from neonatal chest X-ray images and clinical text prompts.  
It jointly optimizes **text generation** and **regression** objectives to learn both descriptive radiology-style captioning and continuous GA estimation.

### 🔹 Key Features
- **Transformer-based multimodal encoder–decoder architecture (BLIP)** integrating a **Vision Transformer (ViT)** and **text transformer** for cross-modal understanding.  
- **Custom CXR dataset class** with on-the-fly augmentations and tokenized text prompts.  
- **Dual-loss optimization** combining:  
  - Weighted cross-entropy loss (upweighting digits for GA relevance).  
  - MSE regression loss from a new `ga_regressor` head.  
- **Z-score normalization** of GA values for stable regression.  
- **Evaluation metrics:** BLEU, ROUGE-L, BERTScore, and entity-level F1.  
- **Outputs:** generated reports, GA predictions, and evaluation metrics saved as CSV.

### 🧩 Model Base
`nathansutton/generate-cxr` — BLIP architecture fine-tuned for medical imaging and report generation.

### ⚙️ Dependencies
`transformers`, `torch`, `evaluate`, `spacy`, `pandas`, `scikit-learn`, `Pillow`, `numpy`

---

## 🌈 **Code 2: `Stable_diffusion_model_training_finetuning.py`**

This script fine-tunes and trains a **Stable Diffusion UNet** model for **neonatal chest X-ray image generation** from clinical text prompts.  
It integrates **mask-region supervision** with **diffusion-based reconstruction** to enhance both spatial fidelity and semantic alignment between text and generated images.

### 🔹 Key Features
- **Custom per-class dataset** supporting multi-label prompts and corresponding region masks.  
- **Dual-arm training** (per-class and union-mask datasets) for balanced learning of local and global image structures.  
- **Mask-weighted loss function** combining:  
  - Base pixel-wise MSE loss.  
  - Weighted mask MSE (λ = 0.5) emphasizing anatomical or device regions.  
- **Automatic checkpointing and resume** for UNet and optimizer states.  
- **Integrated evaluation** after each epoch with image and text metrics:  
  - Image: **FID (Frechet Inception Distance)** and **SSIM (Structural Similarity Index)**.  
  - Text: **ROUGE-L**, **BERTScore**, **BLEU**, **CIDEr**, **METEOR** via BLIP caption comparison.  
- **Final test generation** step producing synthetic X-rays conditioned on text prompts.  
- **Deterministic, reproducible training** using fixed seeds and controlled logging.

### 💾 Outputs
- **Checkpoints:** `checkpoints/unet_epoch_*.pt` and `optimizer_epoch_*.pt`  
- **Metrics:** stored in `generated_images_masks_july6/metrics.csv`  
- **Generated Images:** saved under `generated_images_masks_july6/testset_gen/`

### 🧩 Model Base
- **Stable Diffusion backbone:** `/home/sshome/roentgen_project/roentgen`  
- **Fine-tuned BLIP captioner for neonatal CXRs:** `/home/sshome/roentgen_project/blip_finetuned/final`

### ⚙️ Dependencies
`torch`, `diffusers`, `transformers`, `torchvision`, `lpips`, `scikit-image`,  
`pandas`, `numpy`, `rouge`, `bert-score`, `pycocoevalcap`, `nltk`, `tqdm`, `Pillow`
