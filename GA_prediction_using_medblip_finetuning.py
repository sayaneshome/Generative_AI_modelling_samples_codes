#!/usr/bin/env python3
import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
import evaluate
import spacy

# ── Hyperparameters ──────────────────────────────────────────────────────────
GA_HEAD_LR      = 1e-3    # learning rate for the GA regressor head
DIGIT_WEIGHT    = 2.0     # upweight digit tokens in CE
GA_LOSS_WEIGHT  = 20.0    # upweight GA regression loss
NUM_EPOCHS      = 10      # number of training epochs
NUM_TEST_IMAGES = 5       # unique‐MRN test samples

# ── Globals for z‐score normalization ─────────────────────────────────────────
GA_MEAN = 0.0
GA_STD  = 1.0

# ── Metrics & NLP Setup ───────────────────────────────────────────────────────
nlp = spacy.load("en_ner_bc5cdr_md")
bleu_metric      = evaluate.load("bleu")
rouge_metric     = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

# ── Raw GA extraction ──────────────────────────────────────────────────────────
GA_PATTERN = re.compile(r"(\d+)\s+days gestational age")
def extract_ga_raw(prompt: str) -> float:
    m = GA_PATTERN.search(prompt)
    return float(m.group(1)) if m else 0.0

def extract_ga_norm(prompt: str) -> float:
    # z‐score normalize using globals
    return (extract_ga_raw(prompt) - GA_MEAN) / GA_STD

# ── Dataset ───────────────────────────────────────────────────────────────────
class CXRBlipDataset(Dataset):
    def __init__(self, df, image_dir, processor, max_length=128):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(0.1, 0.1),
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row["image_filename"])).convert("RGB")
        img = self.augment(img)
        pv = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

        prompt = row["Training_Prompt"]
        tok = self.processor.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids      = tok.input_ids.squeeze(0)
        attention_mask = tok.attention_mask.squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        ga = torch.tensor(extract_ga_norm(prompt), dtype=torch.float)
        return {
            "pixel_values":   pv,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "ga":             ga,
        }

# ── Optimizer & Scheduler ────────────────────────────────────────────────────
def get_optimizer_and_scheduler(model, args):
    # separate LR for ga_regressor
    regressor_params = list(model.ga_regressor.parameters())
    other_params = [p for n,p in model.named_parameters() if "ga_regressor" not in n]
    optimizer = AdamW([
        {"params": regressor_params, "lr": GA_HEAD_LR},
        {"params": other_params,   "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=1, threshold=1e-3, min_lr=1e-7)
    return optimizer, scheduler

# ── Custom Trainer ───────────────────────────────────────────────────────────
class TrainerWithPlateau(Seq2SeqTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps):
        self.optimizer, self.lr_scheduler = get_optimizer_and_scheduler(self.model, self.args)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels         = inputs["labels"]
        pixel_values   = inputs["pixel_values"]
        attention_mask = inputs["attention_mask"]
        input_ids      = inputs["input_ids"]
        ga_targets     = inputs["ga"]

        out = model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            input_ids=input_ids,
            labels=None,
            return_dict=True,
        )
        logits = out.logits

        # weighted CE
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        raw = loss_fct(logits.view(-1, logits.size(-1)),
                       labels.view(-1)).view(labels.size())
        mask = torch.ones_like(labels, dtype=torch.float, device=labels.device)
        for d in self.args.digit_token_ids:
            mask = mask + (labels == d).float() * (DIGIT_WEIGHT - 1.0)
        ce_loss = (raw * mask).sum() / (labels != -100).sum()

        # GA regression
        vout   = model.vision_model(pixel_values)
        pooled = getattr(vout, "pooler_output", vout.last_hidden_state.mean(dim=1))
        ga_pred = model.ga_regressor(pooled).squeeze(-1)
        mse_loss = nn.functional.mse_loss(ga_pred, ga_targets.to(ga_pred.device))

        loss = ce_loss + GA_LOSS_WEIGHT * mse_loss
        return (loss, out) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = {k: v for k,v in inputs.items() if k != "ga"}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        lr = self.optimizer.param_groups[0]["lr"]
        self.log({"learning_rate": lr})
        if "eval_loss" in metrics:
            self.lr_scheduler.step(metrics["eval_loss"])
        return metrics

# ── Text Metrics ──────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds  = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_clean   = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_labels = processor.tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

    bleu_res  = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_res = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    bert_res  = bertscore_metric.compute(predictions=decoded_preds,
                                         references=decoded_labels, lang="en")

    tp=fp=fn=0
    for p,t in zip(decoded_preds, decoded_labels):
        pents = {e.text.lower() for e in nlp(p).ents}
        tents = {e.text.lower() for e in nlp(t).ents}
        tp += len(pents & tents)
        fp += len(pents - tents)
        fn += len(tents - pents)
    prec = tp/(tp+fp+1e-8)
    rec  = tp/(tp+fn+1e-8)
    ent_f1 = 2*prec*rec/(prec+rec+1e-8)

    return {
        "bleu": bleu_res["bleu"],
        "rougeL": rouge_res["rougeL"],
        "bertscore_f1": np.mean(bert_res["f1"]),
        "entity_precision": prec,
        "entity_recall":    rec,
        "entity_f1":        ent_f1,
    }

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    global GA_MEAN, GA_STD

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    df = pd.read_csv("train.csv")

    # hold out unique‐MRN test set
    idxs = list(range(len(df))); random.shuffle(idxs)
    seen, test_idxs = set(), []
    for i in idxs:
        mrn = df.iloc[i]["MRN"]
        if mrn not in seen:
            seen.add(mrn)
            test_idxs.append(i)
            if len(test_idxs) == NUM_TEST_IMAGES:
                break
    test_df     = df.iloc[test_idxs].reset_index(drop=True)
    trainval_df = df.drop(index=test_idxs).reset_index(drop=True)
    test_df.to_csv("test.csv", index=False)
    print(f"✔️ Test set ({NUM_TEST_IMAGES}) → test.csv")

    # compute GA z‐score stats from trainval
    raw_gas = trainval_df["Training_Prompt"].apply(extract_ga_raw).values
    GA_MEAN, GA_STD = raw_gas.mean(), raw_gas.std()

    # train/val split by MRN
    gss = GroupShuffleSplit(test_size=0.20, random_state=42)
    t_idx, v_idx = next(gss.split(trainval_df, groups=trainval_df["MRN"]))
    train_df = trainval_df.iloc[t_idx].reset_index(drop=True)
    val_df   = trainval_df.iloc[v_idx].reset_index(drop=True)
    print(f"✔️ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # model & processor
    processor = BlipProcessor.from_pretrained("nathansutton/generate-cxr")
    processor.tokenizer.padding_side = "left"
    model = BlipForConditionalGeneration.from_pretrained("nathansutton/generate-cxr")
    hidden_size = model.vision_model.config.hidden_size
    model.ga_regressor = nn.Linear(hidden_size, 1)
    for p in model.vision_model.parameters(): p.requires_grad = False
    model.config.decoder_attention_dropout = 0.2
    model.config.dropout                = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    digit_ids = [processor.tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]

    # datasets
    train_ds = CXRBlipDataset(train_df, "/remote/home/sshome/arranged_images_512x512", processor)
    val_ds   = CXRBlipDataset(val_df,   "/remote/home/sshome/arranged_images_512x512", processor)

    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="./blip_full",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=3e-5,
        weight_decay=0.05,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
    )
    training_args.digit_token_ids = digit_ids

    trainer = TrainerWithPlateau(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # train + save
    trainer.train()
    trainer.save_model("./blip_full/final")

    # final test eval
    test_ds  = CXRBlipDataset(test_df, "/remote/home/sshome/arranged_images_512x512", processor)
    test_res = trainer.predict(test_ds)
    print("Test metrics:", test_res.metrics)

    # dump outputs
    decoded  = processor.tokenizer.batch_decode(test_res.predictions, skip_special_tokens=True)
    ga_targs = [extract_ga_raw(p) for p in test_df["Training_Prompt"]]
    ga_preds = []
    model.eval()
    with torch.no_grad():
        for fn in test_df["image_filename"]:
            img = Image.open(f"/remote/home/sshome/arranged_images_512x512/{fn}").convert("RGB")
            pv  = processor(images=img, return_tensors="pt").pixel_values.to(device)
            vo  = model.vision_model(pv)
            pooled = getattr(vo, "pooler_output", vo.last_hidden_state.mean(dim=1))
            norm   = model.ga_regressor(pooled).squeeze().item()
            ga_preds.append(norm * GA_STD + GA_MEAN)

    rouges = [
        rouge_metric.compute(predictions=[g], references=[r])["rougeL"]
        for g, r in zip(decoded, test_df["Training_Prompt"])
    ]

    out_df = pd.DataFrame({
        "image_filename":    test_df["image_filename"],
        "reference_prompt":  test_df["Training_Prompt"],
        "generated_prompt":  decoded,
        "rougeL_score":      rouges,
        "ga_target_days":    ga_targs,
        "ga_pred_days":      ga_preds,
    })
    out_df.to_csv("./blip_full/test_outputs.csv", index=False)
    print("✔️ Saved test outputs → ./blip_full/test_outputs.csv")

if __name__ == "__main__":
    main()
