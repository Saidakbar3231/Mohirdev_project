# =============================================================================
# PHASE 1: Speech-to-Text (STT) Fine-Tuning
# Model: openai/whisper-small
# Dataset: Custom Uzbek audio dataset
# =============================================================================

import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset as HFDataset
import evaluate
from sklearn.model_selection import train_test_split

# ───────────────── 
# CONFIG 
# ─────────────────
AUDIO_DIR   = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\datas\audios"
METADATA    = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\datas\metadata_clean.csv"
OUTPUT_DIR  = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\models\whisper-uz-finetuned"

MODEL_NAME  = "openai/whisper-small"
LANGUAGE    = "uz"
TASK        = "transcribe"
SAMPLE_RATE = 16000

TEST_SIZE   = 0.1
SEED        = 42

TRAIN_BATCH = 4
EVAL_BATCH  = 4
GRAD_ACCUM  = 2
LR          = 1e-5
WARMUP      = 100
MAX_STEPS   = 500
FP16        = torch.cuda.is_available()

# ───────────────── 
# DATA LOADING 
# ─────────────────
print("Dataset yuklanmoqda...")

df = pd.read_csv(METADATA)

df["audio_path"] = df["file_name"].apply(
    lambda x: str(Path(AUDIO_DIR) / x.replace(".wav", "_noise.wav"))
)

df = df[df["audio_path"].apply(os.path.exists)].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ───────────────── MODEL ─────────────────
print(" Model yuklanmoqda...")

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# NEW STYLE GENERATION CONFIG
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK
)
model.generation_config.suppress_tokens = []

print(" Model tayyor")

# ───────────────── 
# PREPROCESSING
# ─────────────────
def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio

def prepare_dataset(df_subset):
    records = []
    for _, row in df_subset.iterrows():
        audio = load_audio(row["audio_path"])

        input_features = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features[0]

        labels = processor.tokenizer(
            row["transcription"]
        ).input_ids

        records.append({
            "input_features": input_features,
            "labels": labels,
        })
    return records

train_records = prepare_dataset(train_df)
test_records  = prepare_dataset(test_df)

hf_train = HFDataset.from_list(train_records)
hf_test  = HFDataset.from_list(test_records)

# ───────────────── 
# DATA COLLATOR 
# ─────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]):

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

# ───────────────── 
# METRIC 
# ─────────────────
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str  = [p.lower().strip() for p in pred_str]
    label_str = [l.lower().strip() for l in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4)}

# ───────────────── 
# BASE WER 
# ─────────────────
print("Base WER hisoblanmoqda...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

base_preds, base_refs = [], []

for rec in test_records[:50]:
    with torch.no_grad():
        input_feat = rec["input_features"].clone().detach().unsqueeze(0).to(device)
        pred_ids = model.generate(input_feat, max_length=225)

    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    base_preds.append(pred_text.lower().strip())

    ref_text = processor.tokenizer.decode(
        rec["labels"],
        skip_special_tokens=True
    )
    base_refs.append(ref_text.lower().strip())

base_wer = wer_metric.compute(predictions=base_preds, references=base_refs)
print(f"Base WER: {base_wer:.4f}")

# ───────────────── 
# TRAINING
# ─────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=WARMUP,
    max_steps=MAX_STEPS,
    fp16=FP16,
    eval_strategy="steps",          # <-- MUHIM O‘ZGARISH
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=hf_train,
    eval_dataset=hf_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

trainer.train()

# ───────────────── 
# FINAL WER 
# ─────────────────
print("Fine-tuned WER hisoblanmoqda...")
eval_results = trainer.evaluate()
finetuned_wer = eval_results["eval_wer"]

print("\n========== NATIJA ==========")
print(f"Base WER      : {base_wer*100:.2f}%")
print(f"Fine-tuned WER: {finetuned_wer*100:.2f}%")
print(f"Improvement   : {(base_wer-finetuned_wer)*100:.2f}%")

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("Phase 1 tugadi")