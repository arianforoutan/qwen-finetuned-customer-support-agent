import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY", ),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)
MODEL_NAME = ("gpt-4o-mini")

OUTPUT_DIR = "./data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RAW_FILE = os.path.join(OUTPUT_DIR, "train_raw_full.jsonl")

TAXONOMY = {
    "order_tracking": {
        "tool": "get_order_status",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "پیگیری وضعیت فعلی سفارش، موقعیت مکانی بسته یا کد رهگیری پستی",
    },
    "order_delay": {
        "tool": "escalate_shipping",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "اعتراض یا پیگیری تاخیر در دریافت کالا، ماندن در وضعیت پردازش، یا دیرکرد مامور پست",
    },
    "order_cancellation": {
        "tool": "cancel_order",
        "requires_tool": True,
        "requires_confirmation": True,
        "desc": "درخواست لغو، انصراف یا حذف سفارش قبل از رسیدن مرسوله",
    },
    "order_modification": {
        "tool": "modify_order",
        "requires_tool": True,
        "requires_confirmation": True,
        "desc": "تغییر آدرس پستی، ویرایش شماره موبایل، جابجایی سایز/رنگ یا تغییر اقلام قبل از ارسال",
    },
    "payment_failed": {
        "tool": "check_payment_status",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "کسر پول از کارت بانکی و عدم دریافت کد ثبت سفارش یا خطای درگاه پرداخت",
    },
    "payment_methods": {
        "tool": "rag_policy_search",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "سوالات درباره نحوه خرید اقساطی، پرداخت در محل، خرید با چک صیادی یا سقف اعتبار",
    },
    "refund_request": {
        "tool": "create_refund_ticket",
        "requires_tool": True,
        "requires_confirmation": True,
        "desc": "درخواست عودت کالا پس از تحویل و بازگرداندن مبلغ به شماره شبا یا کارت",
    },
    "refund_status": {
        "tool": "get_refund_status",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "پیگیری زمان واریز پول کالای مرجوع شده یا تاییدیه انبار",
    },
    "refund_policy": {
        "tool": "rag_policy_search",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "قوانین مدت زمان مهلت تست (۷ روزه)، شرایط پذیرش کالای باز شده یا قوانین پلمب",
    },
    "product_availability": {
        "tool": "check_inventory",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "استعلام موجود بودن محصول، تاریخ شارژ مجدد، تنوع رنگ‌بندی یا سایزها در انبار",
    },
    "product_technical": {
        "tool": "rag_product_search",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "مشخصات فنی، تطابق سخت‌افزاری، ولتاژ، توان مصرفی و شیوه راه‌اندازی محصول",
    },
    "warranty_inquiry": {
        "tool": "rag_policy_search",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "مدت اعتبار گارانتی، نوع شرکت ارائه‌دهنده، پوشش شکستگی یا خدمات پس از فروش",
    },
    "shipping_cost": {
        "tool": "calculate_shipping",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "هزینه حمل‌ونقل بر اساس شهر یا وزن، روش‌های ارسال (تیپاکس/پست پیشتاز/پیک)",
    },
    "damaged_product": {
        "tool": "register_complaint",
        "requires_tool": True,
        "requires_confirmation": True,
        "desc": "گزارش آسیب فیزیکی محصول، شکستگی بسته، خط و خش یا ارسال کالای اشتباهی",
    },
    "general_complaint": {
        "tool": "register_complaint",
        "requires_tool": True,
        "requires_confirmation": False,
        "desc": "نارضایتی از نحوه مکالمه یا پاسخگویی کارشناسان، قطعی مکرر سایت یا بدقولی کلی سیستم",
    },
}

SYSTEM_PROMPT = "شما موتور تحلیل پیام مشتریان customer-support-ai هستید. پیام کاربر را تحلیل کنید و خروجی را دقیقاً در قالب JSON مشخص‌شده برگردانید."


def generate_batch(intent_name, metadata, batch_size=25):
    prompt = f"""شما مسئول تولید دیتاست هوش مصنوعی پشتیبانی فروشگاه اینترنتی ایرانی هستید.
تعداد {batch_size} پیام منحصربه‌فرد، واقع‌گرایانه و کاربردی به زبان فارسی برای Intent زیر بسازید:

Intent: {intent_name}
توضیح هدف: {metadata['desc']}

دستورالعمل‌های الزامی:
1. فقط و فقط زبان فارسی (استفاده از کلمات غیرفارسی مثل کره ای، اسپانیایی، آلمانی یا اسامی بی معنا مطلقا ممنوع است).
2. تنوع لحن: رسمی، عامیانه، کلافه، تایپ سریع همراه با غلط های تایپی رایج کیبورد (فاصله اشتباه، جابجایی حروف، ک/ي).
3. برخی رکوردها بدون موجودیت (Entity) باشند و برخی حاوی شناسه سفارش (عددی فارسی یا انگلیسی)، نام کالا یا بازه زمانی.
4. خروجی فقط یک آرایه معتبر JSON شامل آبجکت‌های {{"user_text": "...", "entities": {{...}}}} بدون هیچ متن یا توضیح اضافه.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.88,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error on {intent_name}: {e}")
        return []


def main():
    # 5 بچ ۲۵ تایی = ۱۲۵ نمونه برای هر اینتنت (در مجموع حدود ۱,۸۷۵ نمونه)
    batches = 5
    batch_size = 25
    count = 0

    with open(RAW_FILE, "w", encoding="utf-8") as f:
        for intent, meta in tqdm(TAXONOMY.items(), desc="Generating Full Data"):
            for _ in range(batches):
                items = generate_batch(intent, meta, batch_size=batch_size)
                for item in items:
                    txt = item.get("user_text", "").strip()
                    ent = item.get("entities", {})
                    if txt:
                        rec = {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": txt},
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {
                                            "intent": intent,
                                            "entities": ent,
                                            "requires_tool": meta[
                                                "requires_tool"
                                            ],
                                            "tool": meta["tool"],
                                            "requires_confirmation": meta[
                                                "requires_confirmation"
                                            ],
                                            "needs_clarification": len(ent) == 0
                                            and intent
                                            in [
                                                "order_tracking",
                                                "order_cancellation",
                                            ],
                                        },
                                        ensure_ascii=False,
                                    ),
                                },
                            ]
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        count += 1
                time.sleep(0.3)

    print(f"\nتولید کامل شد! مجموع نمونه‌ها: {count}")


if __name__ == "__main__":
    main()