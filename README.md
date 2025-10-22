**Code 1 : GA_prediction_using_medblip_finetuning.py**

This script fine-tunes the BLIP (Bootstrapped Language–Image Pretraining) model for gestational age (GA) prediction from neonatal chest X-ray images and clinical text prompts. It integrates text generation and regression objectives to jointly learn descriptive captioning and numerical GA estimation.

**Key Features:**

Custom CXR dataset class with on-the-fly augmentations and tokenized text prompts.

Dual-loss optimization combining:

Weighted cross-entropy loss (upweighting digits for GA relevance)

MSE regression loss from a new ga_regressor head.

Evaluation with BLEU, ROUGE-L, BERTScore, and entity-level F1 metrics.

Uses z-score normalized GA values for stable regression.

Outputs generated reports, GA predictions, and evaluation metrics to CSV.

Model base: nathansutton/generate-cxr

Dependencies: transformers, torch, evaluate, spacy, PIL, pandas, scikit-learn
**
Code 2 :  Stable_diffusion_model_training_finetuning.py**

This script fine-tunes and trains a Stable Diffusion UNet model for neonatal chest X-ray image generation from clinical text prompts. It combines masked-region supervision with diffusion-based reconstruction to improve spatial fidelity and semantic alignment between prompts and generated images.

**Key Features**

Custom per-class dataset supporting multi-label prompts and corresponding region masks.

Dual-arm training (per-class and union-mask datasets) for balanced learning of class-specific and global features.

Mask-weighted loss function combining:

Base pixel-wise MSE loss.

Weighted mask MSE (λ = 0.5) to emphasize anatomical or device regions.

Automatic checkpointing and resume support for UNet and optimizer states.

Integrated evaluation after each epoch with image and text metrics:

FID (Frechet Inception Distance) and SSIM (Structural Similarity Index).

Text alignment metrics: ROUGE-L, BERTScore, BLEU, CIDEr, METEOR via BLIP caption comparison.

Final test generation stage producing synthetic X-ray images conditioned on textual prompts.

Deterministic and reproducible setup (global seeds, controlled logging).

Outputs

Epoch-wise checkpoints: checkpoints/unet_epoch_*.pt and optimizer_epoch_*.pt.

Evaluation metrics stored in generated_images_masks_july6/metrics.csv.

Generated test images saved under generated_images_masks_july6/testset_gen/.

Model Base

Stable Diffusion backbone: /home/sshome/roentgen_project/roentgen

Fine-tuned BLIP captioner for neonatal CXRs: /home/sshome/roentgen_project/blip_finetuned/final

Dependencies

torch, diffusers, transformers, torchvision, lpips, scikit-image,
pandas, numpy, rouge, bert-score, pycocoevalcap, nltk, tqdm, Pillow
