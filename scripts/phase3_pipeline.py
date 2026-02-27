#!/usr/bin/env python3
# =============================================================================
# pipeline.py — Phase 3: Full STT + Summarization Pipeline
#
# Usage:
#   python pipeline.py --audio path/to/audio.wav
#   python pipeline.py --audio path/to/audio.mp3
# =============================================================================

import os
import sys
import argparse
import torch
import librosa
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# DEFAULT MODEL PATHS (Phase 1 va 2 dan keyin)
# ─────────────────────────────────────────────
DEFAULT_STT_MODEL = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\models\whisper-uz-finetuned"
DEFAULT_LLM_MODEL = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\models\flan-t5-uz-summarizer"

# Agar fine-tuned model yo'q bo'lsa, base modellarni ishlatish
FALLBACK_STT = "openai/whisper-small"
FALLBACK_LLM = "google/flan-t5-small"

SAMPLE_RATE = 16000
MAX_SUMMARY_LEN = 128
NUM_BEAMS = 4

# ─────────────────────────────────────────────
# 1. MODEL YUKLASH
# ─────────────────────────────────────────────
def load_stt_model(model_path: str):
    """Whisper STT model va processori yuklash."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    # Fine-tuned model bor yoki yo'qligini tekshirish
    if os.path.exists(model_path):
        print(f" Fine-tuned STT model yuklanmoqda: {model_path}")
        src = model_path
    else:
        print(f" Fine-tuned model topilmadi. Base model ishlatilmoqda: {FALLBACK_STT}")
        src = FALLBACK_STT

    processor = WhisperProcessor.from_pretrained(src)
    model = WhisperForConditionalGeneration.from_pretrained(src)
    return processor, model


def load_llm_model(model_path: str):
    """Flan-T5 summarization model va tokenizeri yuklash."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if os.path.exists(model_path):
        print(f"Fine-tuned LLM model yuklanmoqda: {model_path}")
        src = model_path
    else:
        print(f"Fine-tuned model topilmadi. Base model ishlatilmoqda: {FALLBACK_LLM}")
        src = FALLBACK_LLM

    tokenizer = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSeq2SeqLM.from_pretrained(src)
    return tokenizer, model

# ─────────────────────────────────────────────
# 2. AUDIO → TRANSCRIPT
# ─────────────────────────────────────────────
def transcribe(audio_path: str, processor, stt_model, device: str) -> str:
    """Audio faylni matnga aylantirish."""
    # Audio yuklash (wav yoki mp3)
    audio_path = str(audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio fayl topilmadi: {audio_path}")

    print(f"  Audio yuklanmoqda: {Path(audio_path).name}")
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    print(f"   Davomiyligi: {len(audio)/SAMPLE_RATE:.1f}s | Sample rate: {SAMPLE_RATE}Hz")

    # 30 sekunddan uzun audiolarni bo'laklash
    chunk_size = 30 * SAMPLE_RATE  # 30s chunks
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

    transcripts = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) < 1000:  # Juda qisqa chunk — o'tkazib yuborish
            continue
        input_features = processor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = stt_model.generate(input_features)

        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcripts.append(chunk_text.strip())
        if len(chunks) > 1:
            print(f"   Chunk {idx+1}/{len(chunks)}: {chunk_text[:50]}...")

    full_transcript = " ".join(transcripts).strip()
    return full_transcript

# ─────────────────────────────────────────────
# 3. TRANSCRIPT → SUMMARY
# ─────────────────────────────────────────────
def summarize(text: str, tokenizer, llm_model, device: str) -> str:
    """Matnni qisqartirish."""
    if not text or len(text.split()) < 10:
        return "Matn juda qisqa — qisqartirish talab etilmadi."

    # Instruction prefix (flan-t5 style)
    input_text = f"summarize: {text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        generated = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_SUMMARY_LEN,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    summary = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    return summary

# ─────────────────────────────────────────────
# 4. ASOSIY PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(
    audio_path: str,
    stt_model_path: str = DEFAULT_STT_MODEL,
    llm_model_path: str = DEFAULT_LLM_MODEL,
):
    """To'liq pipeline: Audio → Transcript → Summary."""
    print("\n" + "=" * 65)
    print(" ML PIPELINE: STT + SUMMARIZATION")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Device: {device.upper()}")

    # Modellarni yuklash
    print("\n Modellar yuklanmoqda...")
    stt_processor, stt_model = load_stt_model(stt_model_path)
    llm_tokenizer, llm_model = load_llm_model(llm_model_path)

    stt_model.to(device).eval()
    llm_model.to(device).eval()

    # Phase 1: Speech → Text
    print("\n[PHASE 1] NUTQ → MATN...")
    transcript = transcribe(audio_path, stt_processor, stt_model, device)

    # Phase 2: Text → Summary
    print("\n[PHASE 2] MATN → QISQACHA...")
    summary = summarize(transcript, llm_tokenizer, llm_model, device)

    # Natija
    print("\n" + "=" * 65)
    print("NATIJA")
    print("=" * 65)
    print(f"\n Original Transcript:\n{transcript}\n")
    print(f" Summary:\n{summary}\n")
    print("=" * 65)

    return {
        "audio_file": str(Path(audio_path).name),
        "transcript": transcript,
        "summary": summary,
    }

# ─────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STT + Summarization Pipeline"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Audio fayl yo'li (.wav yoki .mp3)",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=DEFAULT_STT_MODEL,
        help="Fine-tuned Whisper model yo'li",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="Fine-tuned Flan-T5 model yo'li",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Natijani output.json ga saqlash",
    )

    args = parser.parse_args()

    result = run_pipeline(
        audio_path=args.audio,
        stt_model_path=args.stt_model,
        llm_model_path=args.llm_model,
    )

    if args.save_output:
        import json
        out_file = "output.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f" Natija saqlandi: {out_file}")
