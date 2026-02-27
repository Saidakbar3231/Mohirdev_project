# =============================================================================
# create_uz_summary_dataset.py
# Gemini API orqali o'zbekcha transcript → summary dataset yaratish
# pip install google-genai pandas tqdm
# =============================================================================

import os
import re
import time
import pandas as pd
from tqdm import tqdm
from google import genai

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyDQ_giqUogzvb_Pf1UnRwOmBIot0ZoVIMs"
METADATA_CSV   = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\datas\metadata.csv"
OUTPUT_CSV     = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\datas\uz_summary_dataset.csv"
CHECKPOINT_CSV = r"C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\datas\uz_summary_checkpoint.csv"

MODEL_NAME  = "gemini-2.5-flash"
DELAY_SEC   = 1.0   # rate limit uchun
BATCH_SAVE  = 50    # har 50 ta dan checkpoint saqlash

# ─────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_summary(transcript: str) -> str:
    prompt = f"""Quyidagi o'zbek tilidagi suhbat yoki nutq parchasi berilgan.
Uni 1-2 jumlada qisqacha xulosa qil. Faqat o'zbek tilida yoz.
Qo'shimcha izoh yoki tushuntirish berma, faqat xulosa yoz.

Nutq:
{transcript}

Xulosa:"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text.strip()

# ─────────────────────────────────────────────
# METADATA O'QISH
# ─────────────────────────────────────────────
def read_metadata_csv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"^file_name,transcription\r?\n", "", content)
    pattern = re.compile(r'([^\n,]+?\.wav),"(.*?)"(?=\r?\n[^\n,]+?\.wav|\r?\n?$)', re.DOTALL)
    matches = pattern.findall(content)
    rows = []
    for fname, trans in matches:
        fname = fname.strip().strip('"')
        trans = trans.replace("\n", " ").replace("\r", " ").replace('""', '"').strip()
        if fname and trans:
            rows.append({"file_name": fname, "transcription": trans})
    return pd.DataFrame(rows)

print("Dataset yuklanmoqda...")
df = read_metadata_csv(METADATA_CSV)
df = df.dropna(subset=["transcription"])
df = df[df["transcription"].str.strip() != ""].reset_index(drop=True)
print(f"Jami: {len(df)} ta transcript")

# ─────────────────────────────────────────────
# CHECKPOINT — qolgan joydan davom etish
# ─────────────────────────────────────────────
already_done = set()
results = []

if os.path.exists(CHECKPOINT_CSV):
    checkpoint_df = pd.read_csv(CHECKPOINT_CSV)
    already_done = set(checkpoint_df["file_name"].tolist())
    results = checkpoint_df.to_dict("records")
    print(f"Checkpoint: {len(already_done)} ta allaqachon tayyor, qolganidan davom etilmoqda...")

# ─────────────────────────────────────────────
# SUMMARY YARATISH
# ─────────────────────────────────────────────
todo = len(df) - len(already_done)
print(f"\n Gemini ({MODEL_NAME}) bilan summary yaratilmoqda...")
print(f"   Qolgan: {todo} ta\n")

errors = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Summary"):
    fname      = row["file_name"]
    transcript = row["transcription"]

    if fname in already_done:
        continue

    # Juda qisqa bo'lsa o'zini summary qilib saqlash
    if len(transcript.split()) < 5:
        results.append({
            "file_name":     fname,
            "transcription": transcript,
            "summary":       transcript
        })
        already_done.add(fname)
        continue

    try:
        summary = generate_summary(transcript)
        results.append({
            "file_name":     fname,
            "transcription": transcript,
            "summary":       summary
        })
        already_done.add(fname)

    except Exception as e:
        print(f"\n Xato ({fname}): {e}")
        errors += 1
        results.append({
            "file_name":     fname,
            "transcription": transcript,
            "summary":       transcript[:150]   # xato bo'lsa transcript boshi
        })
        already_done.add(fname)

    time.sleep(DELAY_SEC)

    # Checkpoint saqlash
    if len(results) % BATCH_SAVE == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False, encoding="utf-8")
        tqdm.write(f" Checkpoint: {len(results)} ta saqlandi")

# ─────────────────────────────────────────────
# YAKUNIY SAQLASH
# ─────────────────────────────────────────────
final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"\n{'='*60}")
print(f" Dataset tayyor!")
print(f"   Jami      : {len(final_df)} ta juftlik")
print(f"   Xatolar   : {errors} ta")
print(f"   Saqlandi  : {OUTPUT_CSV}")
print(f"{'='*60}")
print("\nNamuna (3 ta):")
for _, r in final_df.head(3).iterrows():
    print(f"\n  Transcript : {r['transcription'][:80]}...")
    print(f"  Summary    : {r['summary']}")