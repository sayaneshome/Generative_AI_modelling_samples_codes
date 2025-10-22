#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import glob
import logging
import warnings
import csv

import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch_fidelity import calculate_metrics
from rouge import Rouge
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

from torchvision import transforms, utils as vutils
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    BlipProcessor,
    BlipForConditionalGeneration
)
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

# ── Config & Logging ───────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Hyperparams & Paths ────────────────────────────────────────────────────────
MODEL_PATH       = "/home/sshome/roentgen_project/roentgen"
CHECKPOINT_DIR   = "./checkpoints"
SAVE_DIR         = "./generated_images"
TEST_CSV         = "test.csv"

SAMPLE_SIZE      = 50
SSIM_THRESHOLD   = 0.4
MAX_REPORTS      = SAMPLE_SIZE
BATCH_SIZE       = 10

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────────────────────────────
def epoch_from_path(path):
    m = re.search(r'unet_epoch_(\d+)\.pt$', path)
    return int(m.group(1)) if m else -1

rouge_evaluator  = Rouge()
cider_evaluator  = Cider()
meteor_evaluator = Meteor()
smoothie         = SmoothingFunction().method4

def clean_report(rpt: str) -> str:
    rpt = re.sub(r"(?i)\bpostop chest\b.*?(\.|\Z)", "", rpt, flags=re.DOTALL)
    rpt = re.sub(r"(?i)\bi, the attending\b", "", rpt)
    rpt = re.sub(r"(?i)\binterpreted by attending radiologist\b.*?(\.|\Z)", "", rpt, flags=re.DOTALL)
    rpt = re.sub(r"\s{2,}", " ", rpt).strip()
    if rpt and not rpt.endswith("."):
        rpt += "."
    return rpt

def evaluate_text_metrics(reports, references):
    rouge_scores, bert_scores, bleu_scores, cider_scores, meteor_scores = [], [], [], [], []
    for r, ref in zip(reports, references):
        rouge_scores.append(rouge_evaluator.get_scores(r, ref)[0]['rouge-l']['f'])
        _, _, f1 = bert_score([r], [ref], lang='en', verbose=False)
        bert_scores.append(f1.mean().item())
        bleu_scores.append(sentence_bleu([ref.split()], r.split(), smoothing_function=smoothie))
        gts, res = {0: [ref]}, {0: [r]}
        c_score, _ = cider_evaluator.compute_score(gts, res)
        m_score, _ = meteor_evaluator.compute_score(gts, res)
        cider_scores.append(c_score)
        meteor_scores.append(m_score)
    return (
        np.mean(rouge_scores),
        np.mean(bert_scores),
        np.mean(bleu_scores),
        np.mean(cider_scores),
        np.mean(meteor_scores),
    )

# ── BLIP Captioning Setup ─────────────────────────────────────────────────────
processor     = BlipProcessor.from_pretrained("nathansutton/generate-cxr")
caption_model = BlipForConditionalGeneration.from_pretrained(
    os.path.join(MODEL_PATH, "../blip_finetuned/final"),
    torch_dtype=torch.float16
)

def generate_report(image_tensor, device):
    caption_model.to(device)
    img = transforms.ToPILImage()(image_tensor.squeeze(0)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    out    = caption_model.generate(
        **inputs,
        max_length=100,
        num_beams=5,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=3
    )
    return processor.decode(out[0], skip_special_tokens=True)

# ── Dataset & Collation ────────────────────────────────────────────────────────
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

class MedicalImageDataset(Dataset):
    def __init__(self, df, image_dir, tokenizer):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row['image_filename'])).convert("RGB")
        pix = image_transform(img)
        tok = self.tokenizer(
            row['Training_Prompt'], padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        return {
            "pixel_values": pix,
            "input_ids": tok.input_ids.squeeze(0),
            "attention_mask": tok.attention_mask.squeeze(0),
            "prompt_text": row['Training_Prompt'],
            "filename": row['image_filename']
        }

def custom_collate(batch):
    pix       = torch.stack([b['pixel_values']    for b in batch])
    ids       = torch.stack([b['input_ids']        for b in batch])
    att       = torch.stack([b['attention_mask']   for b in batch])
    texts     = [b['prompt_text']  for b in batch]
    filenames = [b['filename']     for b in batch]
    return {
        "pixel_values": pix,
        "input_ids": ids,
        "attention_mask": att,
        "prompts": texts,
        "filenames": filenames
    }

# ── Core Evaluation ────────────────────────────────────────────────────────────
def evaluate_model(unet, vae, text_encoder, test_df, tokenizer, epoch, save_dir, device, max_reports):
    unet.eval(); vae.eval(); text_encoder.eval()
    ds     = MedicalImageDataset(test_df, "/remote/home/sshome/arranged_images_512x512", tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    sched  = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    real_dir = os.path.join(save_dir, f"dyn_ckpt{epoch}_reals")
    gen_dir  = os.path.join(save_dir, f"dyn_ckpt{epoch}_gens")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    ssim_scores, reports, references = [], [], []
    passing_files = []
    valid_count = 0

    for batch in loader:
        pix = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        enc_hid = text_encoder(input_ids=ids, attention_mask=att)[0]

        bs = pix.size(0)
        lat = torch.randn(
            (bs, unet.config.in_channels, 64, 64),
            dtype=torch.float16, device=device
        ) * sched.init_noise_sigma

        for t in sched.timesteps:
            with torch.no_grad(), torch.cuda.amp.autocast():
                out = unet(lat, int(t), enc_hid)
            lat = sched.step(out.sample, int(t), lat).prev_sample

        with torch.no_grad():
            z   = (lat / 0.18215).to(torch.float16)
            dec = vae.decode(z).sample.clamp(-1,1)
            dec = (dec / 2 + 0.5).clamp(0,1).detach()
        real = ((pix / 2) + 0.5).clamp(0,1)

        for i in range(bs):
            cur_ssim = float(np.mean([
                ssim(
                    dec[i].permute(1,2,0).detach().cpu().numpy()[:,:,c],
                    real[i].permute(1,2,0).cpu().numpy()[:,:,c],
                    data_range=1.0
                ) for c in range(3)
            ]))
            if cur_ssim >= SSIM_THRESHOLD and valid_count < max_reports:
                logging.info(f"[CKPT{epoch} | #{valid_count+1}] SSIM={cur_ssim:.4f}")
                vutils.save_image(real[i], os.path.join(real_dir, f"{valid_count}.png"))
                vutils.save_image(dec[i],  os.path.join(gen_dir,  f"{valid_count}.png"))
                rpt = clean_report(generate_report(dec[i].unsqueeze(0), device))
                reports.append(rpt)
                references.append(batch["prompts"][i])
                ssim_scores.append(cur_ssim)
                passing_files.append(batch["filenames"][i])
                valid_count += 1
            if valid_count >= max_reports:
                break
        if valid_count >= max_reports:
            break

    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_rouge, avg_bert, avg_bleu, avg_cider, avg_meteor = (
        evaluate_text_metrics(reports, references) if ssim_scores else (0,0,0,0,0)
    )
    fid = calculate_metrics(input1=real_dir, input2=gen_dir, fid=True)['frechet_inception_distance']
    logging.info(
        f"[RESULT ckpt{epoch}] SSIM={avg_ssim:.4f}, FID={fid:.4f}, "
        f"ROUGE-L={avg_rouge:.4f}, BERT={avg_bert:.4f}, "
        f"BLEU={avg_bleu:.4f}, METEOR={avg_meteor:.4f}"
    )

    # (Optionally) add t-SNE & distance saving here...

    return (avg_ssim, fid, avg_rouge, avg_bert, avg_bleu, avg_meteor), passing_files

# ── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(MODEL_PATH, "tokenizer"))
    vae       = AutoencoderKL.from_pretrained(
                  MODEL_PATH, subfolder="vae", torch_dtype=torch.float16
                ).to(device)
    text_enc  = CLIPTextModel.from_pretrained(
                  MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.float16
                ).to(device)
    caption_model.to(device)
    torch.cuda.empty_cache()

    # prepare data sample
    df = pd.read_csv(TEST_CSV).sample(n=SAMPLE_SIZE, random_state=42)

    # prepare CSV outputs
    dyn_csv = os.path.join(SAVE_DIR, "metrics_dynamic.csv")
    fix_csv = os.path.join(SAVE_DIR, "metrics_fixed.csv")
    if not os.path.exists(dyn_csv):
        with open(dyn_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","SSIM","FID","ROUGE-L","BERT","BLEU","METEOR"])
    if not os.path.exists(fix_csv):
        with open(fix_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","SSIM","FID","ROUGE-L","BERT","BLEU","METEOR"])

    # gather and filter checkpoints
    all_ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "unet_epoch_*.pt"))
    # drop any file ≤1 KB (likely empty/corrupted)
    ckpts = [p for p in all_ckpts if os.path.getsize(p) > 1024]
    ckpts = sorted(ckpts, key=epoch_from_path)
    if not ckpts:
        raise FileNotFoundError(f"No valid UNet checkpoints in {CHECKPOINT_DIR}")

    fixed_list = None

    for idx, ckpt in enumerate(ckpts):
        epoch = epoch_from_path(ckpt)
        logging.info(f"→ Attempting to load UNet epoch {epoch} from {ckpt}")

        # robust load
        try:
            state = torch.load(ckpt, map_location=device)
        except (EOFError, RuntimeError) as e:
            logging.warning(f"⚠️  Skipping epoch {epoch}: failed to load {ckpt}: {e}")
            continue

        unet = UNet2DConditionModel.from_pretrained(
                   MODEL_PATH, subfolder="unet", torch_dtype=torch.float16
               ).to(device)
        unet.load_state_dict(state)

        # dynamic evaluation
        (s_d, f_d, r_d, b_d, l_d, m_d), passing = evaluate_model(
            unet, vae, text_enc, df, tokenizer,
            epoch, SAVE_DIR, device, MAX_REPORTS
        )
        with open(dyn_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, s_d, f_d, r_d, b_d, l_d, m_d])

        # capture fixed list on first successful epoch
        if fixed_list is None:
            fixed_list = passing

        # fixed evaluation on same subset
        fixed_df = df[df['image_filename'].isin(fixed_list)]
        (s_f, f_f, r_f, b_f, l_f, m_f), _ = evaluate_model(
            unet, vae, text_enc, fixed_df, tokenizer,
            epoch, SAVE_DIR, device, len(fixed_list)
        )
        with open(fix_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, s_f, f_f, r_f, b_f, l_f, m_f])

if __name__ == "__main__":
    main()
