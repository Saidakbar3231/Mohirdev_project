# =============================================================================
# phase2_uz_finetuning.py
# O'zbekcha dataset bilan Flan-T5 fine-tuning
# Dataset: uz_summary_dataset.csv (Gemini bilan yaratilgan)
# =============================================================================

import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME   = "google/flan-t5-small"
OUTPUT_DIR   = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\models\flan-t5-uz-summarizer"
DATASET_CSV  = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\datas\uz_summary_dataset.csv"

MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 128
PREFIX = "summarize: "

TRAIN_BATCH = 8
EVAL_BATCH  = 8
GRAD_ACCUM  = 2
LR          = 3e-4
NUM_EPOCHS  = 5        # kichik dataset — ko'proq epoch
WARMUP      = 50
FP16        = False    # flan-t5 bilan o'chirilgan
SEED        = 42
TEST_SIZE   = 0.1      # 10% test → ~72 ta

print(f"Device: {'CUDA (RTX 3060)' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
# DATASET YUKLASH
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("O'zbekcha dataset yuklanmoqda...")
print("=" * 60)

df = pd.read_csv(DATASET_CSV, encoding="utf-8")
df = df.dropna(subset=["transcription", "summary"])
df = df[df["transcription"].str.strip() != ""]
df = df[df["summary"].str.strip() != ""].reset_index(drop=True)
print(f"Jami: {len(df)} ta juftlik")

# Train / Val / Test split
train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
train_df, val_df      = train_test_split(train_val_df, test_size=0.1, random_state=SEED)

print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"\nNamuna:")
print(f"  Transcript : {train_df.iloc[0]['transcription'][:80]}...")
print(f"  Summary    : {train_df.iloc[0]['summary']}")

# HuggingFace Dataset formatiga o'tkazish
def df_to_hf(df_subset):
    return Dataset.from_dict({
        "transcription": df_subset["transcription"].tolist(),
        "summary":       df_subset["summary"].tolist(),
    })

hf_train = df_to_hf(train_df)
hf_val   = df_to_hf(val_df)
hf_test  = df_to_hf(test_df)

# ─────────────────────────────────────────────
# TOKENIZER VA MODEL
# ─────────────────────────────────────────────
print(f"\n{MODEL_NAME} yuklanmoqda...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print(f"Model yuklandi! Parametrlar: {model.num_parameters():,}")

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(examples):
    inputs  = [PREFIX + t for t in examples["transcription"]]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LEN, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LEN, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("\n Tokenization...")
tok_train = hf_train.map(preprocess, batched=True, remove_columns=hf_train.column_names)
tok_val   = hf_val.map(preprocess, batched=True, remove_columns=hf_val.column_names)
tok_test  = hf_test.map(preprocess, batched=True, remove_columns=hf_test.column_names)
print("Tokenization yakunlandi!")

# ─────────────────────────────────────────────
# DATA COLLATOR
# ─────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model,
    label_pad_token_id=-100, pad_to_multiple_of=None,
)

# ─────────────────────────────────────────────
# ROUGE METRIC
# ─────────────────────────────────────────────
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds  = np.clip(preds, 0, tokenizer.vocab_size - 1).astype(np.int32)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int32)
    decoded_preds  = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
    decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

# ─────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────
print("\n BASE MODEL baholanmoqda...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

base_preds, base_refs = [], []
for i in range(min(30, len(hf_test))):
    sample = hf_test[i]
    inputs = tokenizer(
        PREFIX + sample["transcription"],
        return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True
    ).to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_TARGET_LEN)
    base_preds.append(tokenizer.decode(generated[0], skip_special_tokens=True).strip())
    base_refs.append(sample["summary"].strip())

base_rouge = rouge_metric.compute(predictions=base_preds, references=base_refs, use_stemmer=True)
print(f"Base ROUGE-1: {base_rouge['rouge1']*100:.2f}%")
print(f"Base ROUGE-2: {base_rouge['rouge2']*100:.2f}%")
print(f"Base ROUGE-L: {base_rouge['rougeL']*100:.2f}%")

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
print("\n Fine-tuning boshlanyapti (O'zbekcha dataset)...")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=WARMUP,
    weight_decay=0.01,
    fp16=False,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    report_to=["none"],
    push_to_hub=False,
    seed=SEED,
    dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
print("\n FINE-TUNED MODEL baholanmoqda...")
test_results = trainer.predict(tok_test)
ft_rouge = test_results.metrics

print(f" Fine-tuned ROUGE-1: {ft_rouge.get('test_rouge1', 0):.2f}%")
print(f" Fine-tuned ROUGE-2: {ft_rouge.get('test_rouge2', 0):.2f}%")
print(f" Fine-tuned ROUGE-L: {ft_rouge.get('test_rougeL', 0):.2f}%")

# Namuna natijalar
print("\n NAMUNA NATIJALAR:")
model.eval()
for i in range(3):
    sample = hf_test[i]
    inputs = tokenizer(
        PREFIX + sample["transcription"],
        return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True
    ).to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_TARGET_LEN, num_beams=4)
    pred = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\n  [{i+1}] Transcript : {sample['transcription'][:70]}...")
    print(f"       Reference  : {sample['summary']}")
    print(f"       Predicted  : {pred}")

# ─────────────────────────────────────────────
# SAQLASH
# ─────────────────────────────────────────────
print(f"\n Model saqlanmoqda: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

results = {
    "model":            MODEL_NAME,
    "dataset":          "uz_summary_dataset.csv (Gemini generated)",
    "train_size":       len(train_df),
    "base_rouge1":      round(base_rouge["rouge1"] * 100, 2),
    "base_rouge2":      round(base_rouge["rouge2"] * 100, 2),
    "base_rougeL":      round(base_rouge["rougeL"] * 100, 2),
    "finetuned_rouge1": round(ft_rouge.get("test_rouge1", 0), 2),
    "finetuned_rouge2": round(ft_rouge.get("test_rouge2", 0), 2),
    "finetuned_rougeL": round(ft_rouge.get("test_rougeL", 0), 2),
}
with open(os.path.join(OUTPUT_DIR, "rouge_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print(" NATIJALAR XULOSASI")
print("=" * 60)
print(json.dumps(results, indent=2, ensure_ascii=False))
print(" Phase 2 (O'zbekcha) yakunlandi!")