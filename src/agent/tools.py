"""Operational Tools for Customer Support Agent."""

from typing import Any, Dict
from src.agent.mock_data import MOCK_DATABASE, MOCK_KNOWLEDGE_BASE


class SupportTools:

  @staticmethod
  def get_order_status(order_id: str) -> Dict[str, Any]:
    """استعلام وضعیت سفارش از پایگاه داده."""
    order = MOCK_DATABASE["orders"].get(str(order_id).strip())
    if not order:
      return {
          "success": False,
          "message": (
              f"سفارشی با شناسه «{order_id}» در سامانه یافت نشد. لطفاً شماره"
              " سفارش را بررسی کنید."
          ),
      }

    items_str = "، ".join(order["items"])
    msg = (
        f"سفارش {order_id} در وضعیت «{order['status']}» قرار دارد. اقلام:"
        f" {items_str} (مقصد: {order['city']})."
    )
    return {"success": True, "message": msg, "data": order}

  @staticmethod
  def escalate_shipping(order_id: str) -> Dict[str, Any]:
    """ثبت تیکت تعجیل و اولویت‌بخشی برای مرسولات دارای تاخیر."""
    order = MOCK_DATABASE["orders"].get(str(order_id).strip())
    if not order:
      return {
          "success": False,
          "message": (
              f"سفارش {order_id} پیدا نشد، اما گزارش تاخیر ثبت و به واحد لجستیک"
              " ارجاع شد."
          ),
      }

    msg = f"درخواست تعجیل برای سفارش {order_id} ({order['status']}) ثبت و به واحد توزیع {order['city']} ابلاغ شد."
    return {"success": True, "message": msg}

  @staticmethod
  def cancel_order(order_id: str) -> Dict[str, Any]:
    """لغو سفارش پس از دریافت تاییدیه صریح از مشتری."""
    order = MOCK_DATABASE["orders"].get(str(order_id).strip())
    if not order:
      return {
          "success": False,
          "message": f"سفارش با شناسه {order_id} برای لغو یافت نشد.",
      }

    if not order.get("can_cancel", False):
      return {
          "success": False,
          "message": (
              f"سفارش {order_id} در وضعیت «{order['status']}» است و امکان لغو"
              " خودکار آن وجود ندارد. لطفاً با پشتیبانی تلفنی تماس بگیرید."
          ),
      }

    order["status"] = "لغو شده"
    order["can_cancel"] = False
    return {
        "success": True,
        "message": (
            f"سفارش {order_id} با موفقیت لغو شد. مبلغ"
            f" {order['total_price']:,} تومان تا ۲۴ ساعت آینده به حساب شما"
            " بازگردانده خواهد شد."
        ),
    }

  @staticmethod
  def check_inventory(product_name: str) -> Dict[str, Any]:
    """استعلام موجودی و قیمت کالا در انبار."""
    norm_query = product_name.strip().lower()
    for prod_name, info in MOCK_DATABASE["inventory"].items():
      # تطابق کلمات کلیدی
      query_tokens = [w for w in norm_query.split() if len(w) > 2]
      if any(token in prod_name.lower() for token in query_tokens):
        if info["available"]:
          msg = (
              f"محصول «{prod_name}» در حال حاضر موجود است ({info['stock']} عدد"
              f" در انبار، قیمت: {info['price']:,} تومان)."
          )
        else:
          msg = (
              f"محصول «{prod_name}» متأسفانه در حال حاضر ناموجود است و به زودی"
              " شارژ خواهد شد."
          )
        return {"success": True, "message": msg, "data": info}

    return {
        "success": False,
        "message": (
            f"کالایی مطابق با عنوان «{product_name}» در فهرست موجودی انبار پیدا"
            " نشد."
        ),
    }

  def rag_policy_search(self, query: str) -> dict:
    """جستجوی دقیق و معنایی در پایگاه دانش قوانین فروشگاه."""
    query_lower = query.lower()

    # ۱. قوانین مرجوعی و پس دادن کالا
    if any(
        k in query_lower
        for k in ["refund", "مرجوع", "پس", "عودت", "استرداد", "چند روز"]
    ):
      return {
          "status": "success",
          "message": MOCK_KNOWLEDGE_BASE.get("refund_policy"),
      }

    # ۲. قوانین ارسال و پست
    if any(
        k in query_lower
        for k in ["ship", "ارسال", "پست", "تیپاکس", "هزینه ارسال", "پیک"]
    ):
      return {
          "status": "success",
          "message": MOCK_KNOWLEDGE_BASE.get("shipping_methods"),
      }

    # ۳. شرایط گارانتی و ضمانت
    if any(k in query_lower for k in ["warrant", "گارانتی", "ضمانت", "خراب"]):
      return {
          "status": "success",
          "message": MOCK_KNOWLEDGE_BASE.get("warranty_policy"),
      }

    # ۴. جستجوی متنی کلمات در کل اسناد
    for doc in MOCK_KNOWLEDGE_BASE.values():
      if any(token in doc for token in query_lower.split() if len(token) > 2):
        return {"status": "success", "message": doc}

    return {
        "status": "not_found",
        "message": (
            "اطلاعات دقیقی در بخش قوانین متناسب با درخواست شما یافت نشد؛ جهت"
            " راهنمایی بیشتر به کارشناس متصل می‌شوید."
        ),
    }

    
  @staticmethod
  def register_complaint(details: str) -> Dict[str, Any]:
    """ثبت رسمی شکایت مشتری."""
    ticket_id = "CMP-9402"
    return {
        "success": True,
        "message": (
            f"شکایت شما با کد پیگیری {ticket_id} در سیستم ثبت شد و حداکثر ظرف ۲"
            " ساعت کاری توسط مدیر پشتیبانی پیگیری خواهد شد."
        ),
    }