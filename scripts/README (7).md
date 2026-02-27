# ML Pipeline: Speech-to-Text + Summarization

> O'zbek tilidagi audio fayllarni matnga aylantiruvchi va qisqacha xulosa chiqaruvchi to'liq ML pipeline.  
> **Whisper (STT)** + **Flan-T5 (Summarization)** modellarini o'zbek tili uchun fine-tune qilindi.

---

##  Project Structure

```
Mohirdev_STT_LLM/
│
├── datas/
│   ├── audios/                          # 750 ta o'zbek audio (.wav, _noise suffix)
│   ├── metadata.csv                     # file_name + transcription (original)
│   ├── metadata_clean.csv               # Tozalangan CSV
│   └── uz_summary_dataset.csv           # Gemini bilan yaratilgan (720 juftlik)
│         
│
├── models/
│   ├── whisper-uz-finetuned/            # Fine-tuned STT model
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── training_args.bin
│   │   └── checkpoint-*/               # Oraliq checkpointlar
│   │
│   └── flan-t5-uz-summarizer/           # Fine-tuned Summarization model
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       └── checkpoint-*/               # Oraliq checkpointlar
│
└── scripts/
    ├── phase1_stt_finetuning.py         #  Whisper fine-tuning notebook
    ├── phase2_llm_finetuning.py         #  Flan-T5 fine-tuning notebook
    ├── phase3_pipeline.py               #  To'liq inference pipeline
    ├── create_summary_csv.py            # Gemini API bilan dataset yaratish
    ├── README(7).md
    └── requirements.txt
```

---

##  Model & Dataset Choices

### Phase 1 — STT: `openai/whisper-small`

| Sabab | Tafsilot |
|-------|----------|
| **Ko'p tillik** | 96+ til, o'zbek tili allaqachon qo'llab-quvvatlanadi |
| **Optimal hajm** | 244M parametr — GPU va CPU da ham ishlaydi |
| **Arxitektura** | Encoder-Decoder Transformer — audio spektrogram → matn |
| **Ochiq manba** | MIT License, tijorat loyihalarida ham ishlatish mumkin |

> Alternativlar: `whisper-tiny` (39M, RAM cheklangan holda), `whisper-medium` (769M, yuqori aniqlik uchun)

### Phase 2 — LLM: `google/flan-t5-small`

| Sabab | Tafsilot |
|-------|----------|
| **Yengil** | 77M parametr — Laptop GPU da 2-3 daqiqada train tugaydi |
| **Instruction-tuned** | `"summarize:"` prefix bilan vazifani tushunadi |
| **Seq2Seq** | Matn → Matn arxitekturasi summarization uchun optimal |
| **O'zbek uchun** | Gemini yaratgan o'zbekcha dataset bilan qayta train qilindi |

> Alternativlar: `mt5-small` (ko'p tilli), `flan-t5-base` (250M, aniqroq)

### Datasets

| Phase | Dataset | Hajm | Tanlash sababi |
|-------|---------|------|----------------|
| **STT** | Custom Uzbek audio + `metadata.csv` | 750 audio (720 valid) | Real o'zbek nutqi, so'zlashuv tili va aksent |
| **LLM** | Gemini 2.5 Flash bilan yaratilgan | 720 transcript-summary juftligi | HuggingFace da o'zbekcha summarization dataset yo'q edi |

---

##  Phase 1 — WER Results (STT Fine-tuning)

**WER (Word Error Rate)** — past bo'lsa yaxshi. Formula: `(S + D + I) / N`

| Model | WER (%) |
|-------|---------|
| `whisper-small` — base model | 146.02% |
| `whisper-small` — fine-tuned | **68.83%** |
| **Yaxshilanish** | **⬇ 77.19%** |

### Training Progress (WER per epoch)

| Epoch | WER (%) | Holat |
|-------|---------|-------|
| 0 (base) | 146.02% | Boshlang'ich |
| 1.2 | 109.9% | — |
| 2.5 | 77.9% | — |
| 3.7 | 76.6% | — |
| 4.9 | 71.8% | — |
| **6.2** | **68.8%** |  Best model |

- **Train loss:** 8.47 → **0.41** (model yaxshi o'rgandi)
- **Overfitting:** eval_loss 4-5 epoch da ozgina oshdi, lekin WER kamayishda davom etdi
- **Keyingi qadam:** 32k dataset bilan WER → **20-30%** gacha tushishi kutiladi

---

##  Phase 2 — ROUGE Results (LLM Fine-tuning)

**ROUGE** — yuqori bo'lsa yaxshi. Generated vs Reference summary o'rtasidagi o'xshashlik.

| Metric | Base Model | Fine-tuned | Yaxshilanish |
|--------|-----------|------------|--------------|
| **ROUGE-1** (unigram) | 41.82% | 43.70% | **+1.88%** |
| **ROUGE-2** (bigram) | 15.96% | 19.96% | **+4.00%** |
| **ROUGE-L** (LCS) | 32.48% | 36.08% | **+3.60%** |

- **Training vaqti:** atigi **2 daqiqa 27 soniya** — RTX 3060 Laptop bilan
- **Dataset:** 583 train / 65 val / 72 test

---

##  Phase 3 — Pipeline Demo

```
    Audio fayl (.wav)
        ↓
    Whisper (fine-tuned)  →  O'zbek transcript
        ↓
    Flan-T5 (fine-tuned)  →  Qisqa xulosa
```

### Namuna natija

**Audio:** `chunk_001563_noise.wav` (14.1 soniya)

```
    Original Transcript:
Aha. Oli Oli mali motli Oli mali motli bo'lib, keyin sekin o'sha Ispaniyadami yo galandiyagam borib, dokument topshirib, ishga kirib, oylik olib, nalog to'lab,
  Summary:
Oli mali motli bo'lib, keyin sekin o'sha Ispaniyadami yo galandiyagi olib, dokument topshirib, ishga kirib.
```

---


##  Challenges & Solutions

### 1.  CSV Parsing Error — `EOF inside string`
**Muammo:** `metadata.csv` da transcription ichida vergul va ko'p qatorli matn bor edi. `pd.read_csv()` 750-qatorda crash qilardi.  
**Yechim:** Regex bilan `.wav` fayl nomlarini anchor sifatida ishlatib, ko'p qatorli transcriptionlarni to'g'ri o'qidik:
```python
pattern = re.compile(r'([^\n,]+?\.wav),"(.*?)"', re.DOTALL)
```

### 2.  Audio File Path Mismatch
**Muammo:** CSV da `chunk_001545.wav`, lekin real fayl `chunk_001545_noise.wav`.  
**Yechim:** Path yaratishda suffix qo'shdik:
```python
x.replace(".wav", "_noise.wav")
```

### 3.  Python 3.13 Incompatibility
**Muammo:** PyTorch CUDA wheels Python 3.13 uchun chiqmagan — GPU ko'rinmadi.  
**Yechim:** Python 3.10 bilan alohida virtual environment yaratildi.

### 4. FP16 NaN Gradient (Flan-T5)
**Muammo:** `fp16=True` da `grad_norm: nan` — gradient overflow.  
**Yechim:** `fp16=False` — model stabil ishladi, tezlik ozgina kamaydi.

### 5.  OverflowError in `compute_metrics`
**Muammo:** Prediction tensorida manfiy token IDlar — `batch_decode` crash qildi.  
**Yechim:**
```python
preds = np.clip(preds, 0, tokenizer.vocab_size - 1).astype(np.int32)
```

### 6. 🇺🇿 O'zbek Summarization Dataset yo'q edi
**Muammo:** HuggingFace da o'zbekcha summarization dataset topilmadi.  
**Yechim:** Gemini 2.5 Flash API bilan 720 ta transcriptiondan avtomatik summary yaratildi. Har 50 ta dan checkpoint saqlandi — uzilishdan himoya.

### 7.  transformers API Changes
**Muammo:** Yangi `transformers` versiyasida parameter nomlari o'zgardi.  
**Yechim:**
```python
# Eski → Yangi
evaluation_strategy="epoch"  →  eval_strategy="epoch"
Seq2SeqTrainer(tokenizer=tokenizer)  →  processing_class=tokenizer
```

### 8.  WER hali ham 68%
**Sabab:** Train dataset juda kichik — 648 ta sample.  
**Keyingi qadam:** 32k dataset bilan qayta train — WER **20-30%** gacha tushishi kutiladi.

---

##  Hardware & Environment

| | |
|--|--|
| **GPU** | NVIDIA RTX 3060 Laptop (6.4GB VRAM) |
| **Python** | 3.10.11 |
| **OS** | Windows 11 |
| **CUDA** | 11.8 |
| **PyTorch** | 2.5.1+cu118 |
| **Transformers** | ≥4.40.0 |

---

##  Future Improvements

- [ ] 32k dataset bilan Whisper qayta train (WER → 20-30%)
- [ ] `whisper-large-v3` ishlatish (aniqroq, lekin kattaroq)
- [ ] `mT5` yoki `mBART` bilan o'zbek summarization
- [ ] FastAPI bilan REST API
- [ ] Gradio bilan demo interface

---

##  Requirements

>  **Muhim:** `torch` ni avval alohida o'rnating!

```bash
# 1-qadam: PyTorch (CUDA 11.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2-qadam: Qolganlar
pip install -r requirements.txt
```

```
# requirements.txt

# Transformers & Training
transformers>=4.40.0
datasets>=2.18.0
accelerate>=0.27.0
evaluate>=0.4.1

# STT - Phase 1
librosa>=0.10.0
soundfile>=0.12.1
jiwer>=3.0.3

# LLM Summarization - Phase 2
sentencepiece>=0.1.99
rouge_score>=0.1.2

# Uzbek Dataset (Gemini API)
google-genai>=1.0.0

# Umumiy
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```
