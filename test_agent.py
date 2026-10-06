from src.agent.core import CustomerSupportAgent

agent = CustomerSupportAgent(chat_model_name="qwen-support")

tests = [
    "سلام وقتتون بخیر، سفارش من با شماره ۱۰۲۵ کجاست؟",
    "می‌خوام سفارشم رو کنسل کنم، شمارش ۱۰۲۵ هست",
    "قوانین مرجوعی کالا در سایت شما به چه صورت هست؟",
    "سلام، خسته نباشید.",
    "بسته شماره 12345 وضعیتش چیه؟",
    "آیا گوشی سامسونگ s24 ultra موجود دارین؟"
]

for msg in tests:
    print(f"\n[کاربر]: {msg}")
    result = agent.handle_message(msg)
    print(f"[تصمیم سریع Laya]: {result.get('intent')}")
    print(f"[پاسخ نهایی Qwen]:\n{result['response']}")
    print("=" * 60)