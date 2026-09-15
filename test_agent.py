from src.agent.core import CustomerSupportAgent


def run_tests():
  agent = CustomerSupportAgent(model_name="support-agent")

  print("=" * 60)
  print("تست ۱: استعلام وضعیت مرسوله")
  print("=" * 60)
  res1 = agent.handle_message("سلام، بسته ۹۸۲۳۴ من چی شد پس؟")
  print("پاسخ:", res1["response"])

  print("\n" + "=" * 60)
  print("تست ۲: درخواست لغو سفارش + تایید کاربر")
  print("=" * 60)
  # مرحله اول درخواست
  step1 = agent.handle_message("سفارشم با کد ۵۵۴۴۳ رو لغو کنید لطفاً")
  print("پاسخ گاردریل:", step1["response"])

  step2 = agent.handle_message("بله حتما لغوش کنید", session_context=step1)
  print("پاسخ نهایی پس از تایید:", step2["response"])

  print("\n" + "=" * 60)
  print("تست ۳: سوال از سیاست‌های بازگشت کالا (RAG)")
  print("=" * 60)
  res3 = agent.handle_message("تا چند روز فرصت دارم جنسی که خریدم رو پس بدم؟")
  print("پاسخ دانشی:", res3["response"])


if __name__ == "__main__":
  run_tests()