import re, csv

with open(r'C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\datas\metadata.csv', 'r', encoding='utf-8') as f:
    content = f.read()

# Header ni o'tkazib yuborish
content = re.sub(r'^file_name,transcription\n', '', content)

# Pattern: fayl.wav,"matn..." — ko'p qatorli bo'lsa ham
pattern = re.compile(r'(\S[^\n,]+?\.wav),"(.*?)"(?=\n\S|\Z)', re.DOTALL)
matches = pattern.findall(content)

print(f'Topildi: {len(matches)} ta qator')

with open(r'C:\Users\sobir\OneDrive\Documents\Mohirdev_STT_LLM\datas\metadata_clean.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'transcription'])
    for fname, trans in matches:
        trans_clean = trans.replace('\n', ' ').replace('""', '"').strip()
        writer.writerow([fname.strip(), trans_clean])

print('Saqlandi: metadata_clean.csv')