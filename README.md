Code 1 : GA_prediction_using_medblip_finetuning.py

This script fine-tunes the BLIP (Bootstrapped Language–Image Pretraining) model for gestational age (GA) prediction from neonatal chest X-ray images and clinical text prompts. It integrates text generation and regression objectives to jointly learn descriptive captioning and numerical GA estimation.

Key Features:

Custom CXR dataset class with on-the-fly augmentations and tokenized text prompts.

Dual-loss optimization combining:

Weighted cross-entropy loss (upweighting digits for GA relevance)

MSE regression loss from a new ga_regressor head.

Evaluation with BLEU, ROUGE-L, BERTScore, and entity-level F1 metrics.

Uses z-score normalized GA values for stable regression.

Outputs generated reports, GA predictions, and evaluation metrics to CSV.

Model base: nathansutton/generate-cxr

Dependencies: transformers, torch, evaluate, spacy, PIL, pandas, scikit-learn

Code 2 : 
