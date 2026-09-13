import json
import os
import random

RAW_FILE = "./data/processed/train_raw_full.jsonl"
OUT_DIR = "./data/processed"

HALLUCINATIONS = {
    "esperando": "منتظرم",
    "Gesicht": "رسیدش",
    "مویisel": "مو",
    "멘د": "معلق",
    "번دلی": "بسته‌ای",
    "ocupado": "بوق اشغال",
}

ENTITY_KEY_MAP = {
    "timeframe": "time_range",
    "time_ref": "time_range",
    "date": "time_range",
    "duration": "time_range",
    "deadline": "time_range",
    "tracking_code": "order_id",
    "sku": "product_id",
}


def clean_sample(record):
    user_msg = record["messages"][1]["content"]
    for bad, good in HALLUCINATIONS.items():
        user_msg = user_msg.replace(bad, good)
    record["messages"][1]["content"] = user_msg

    assistant_data = json.loads(record["messages"][2]["content"])

    # یکسان‌سازی کلیدها
    assistant_data["entities"] = {
        ENTITY_KEY_MAP.get(k, k): v
        for k, v in assistant_data.get("entities", {}).items()
    }

    # حل تداخل شکایت با تاخیر
    if assistant_data["intent"] == "general_complaint":
        delay_words = [
            "نرسیده",
            "ارسال نشده",
            "ارسال نمیشه",
            "در حال پردازشه",
            "دیر کرده",
            "تاخیر",
        ]
        if any(w in user_msg for w in delay_words):
            assistant_data["intent"] = "order_delay"
            assistant_data["requires_tool"] = True
            assistant_data["tool"] = "escalate_shipping"
            assistant_data["requires_confirmation"] = False

    record["messages"][2]["content"] = json.dumps(
        assistant_data, ensure_ascii=False
    )
    return record


# خواندن و تمیز کردن داده‌ها
all_records = []
seen_texts = set()

with open(RAW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = clean_sample(json.loads(line))
        txt = rec["messages"][1]["content"]
        # حذف نمونه‌های کاملاً تکراری (Deduplication)
        if txt not in seen_texts:
            seen_texts.add(txt)
            all_records.append(rec)

# بر زدن داده‌ها
random.seed(42)
random.shuffle(all_records)

total = len(all_records)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

splits = {
    "train.jsonl": all_records[:train_end],
    "val.jsonl": all_records[train_end:val_end],
    "test.jsonl": all_records[val_end:],
}

for fname, data in splits.items():
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{fname}: {len(data)} نمونه ذخیره شد.")