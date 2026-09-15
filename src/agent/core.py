"""Core Engine for Autonomous Customer Support Agent."""

import json
import re
from typing import Any, Dict, Optional
import ollama
from src.agent.tools import SupportTools


class CustomerSupportAgent:

  def __init__(self, model_name: str = "support-agent"):
    self.model_name = model_name
    self.tools = SupportTools()

  def _extract_json(self, raw_text: str) -> Dict[str, Any]:
    """پاکسازی پاسخ مدل و استخراج آبجکت معتبر JSON."""
    cleaned = raw_text.strip()
    if "```" in cleaned:
      matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
      if matches:
        cleaned = matches[0].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
      cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)

  def route_message(self, user_text: str) -> Dict[str, Any]:
    """ارسال پیام به اولاما و گرفتن تحلیل ساخت‌یافته."""
    response = ollama.chat(
        model=self.model_name,
        messages=[{"role": "user", "content": user_text}],
        options={"temperature": 0.1},
    )
    raw_content = response["message"]["content"]
    return self._extract_json(raw_content)

  def handle_message(
      self, user_text: str, session_context: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """پردازش اند-تو-اند پیام مشتری و اجرای بیزینس‌لاجیک."""
    cleaned_input = user_text.strip().lower()

    if (
        session_context
        and session_context.get("status") == "waiting_for_confirmation"
    ):
      positive_tokens = [
          "بله",
          "اره",
          "آره",
          "تایید",
          "حتما",
          "کنسل کن",
          "لغوش کن",
          "yes",
      ]
      negative_tokens = ["خیر", "نه", "دست نگه دار", "نمیخوام", "کنسل نکن", "no"]

      raw_order_id = session_context.get("decision", {}).get(
          "entities", {}
      ).get("order_id") or session_context.get("pending_order_id")
      target_id = str(raw_order_id) if raw_order_id else None

      if any(tok in cleaned_input for tok in positive_tokens):
        action_res = self.tools.cancel_order(target_id)
        return {
            "status": "completed",
            "intent": "order_cancellation_confirmed",
            "response": action_res.get("message"),
            "decision": session_context.get("decision"),
        }
      elif any(tok in cleaned_input for tok in negative_tokens):
        return {
            "status": "cancelled_by_user",
            "intent": "order_cancellation_aborted",
            "response": (
                "فرآیند لغو سفارش لغو شد. وضعیت سفارش شما دست‌نخورده باقی ماند."
            ),
            "decision": session_context.get("decision"),
        }
      else:
        session_context = None

    try:
      decision = self.route_message(user_text)
    except Exception as e:
      return {
          "status": "fallback",
          "intent": "unknown",
          "response": (
              "متأسفانه در پردازش پیام شما خطایی رخ داد؛ شما را به کارشناس"
              " پشتیبانی متصل می‌کنیم."
          ),
          "error": str(e),
      }

    intent = decision.get("intent", "general_inquiry")
    entities = decision.get("entities", {})
    requires_tool = decision.get("requires_tool", False)
    tool_name = decision.get("tool")
    requires_confirmation = decision.get("requires_confirmation", False)
    needs_clarification = decision.get("needs_clarification", False)

    order_id = entities.get("order_id")
    order_id_str = str(order_id) if order_id is not None else None

    knowledge_keywords = [
            "چند روز",
            "پس بدم",
            "مرجوع",
            "گارانتی",
            "قوانین",
            "شرایط بازگشت",
            "استرداد",
        ]
    if (
        any(kw in cleaned_input for kw in knowledge_keywords)
        and not order_id_str
       ):
        res = self.tools.rag_policy_search(user_text)
        return {
              "status": "completed",
              "intent": "policy_inquiry",
              "response": res.get("message"),
              "decision": decision,
          }
    
    if (
        needs_clarification
        or (requires_tool and tool_name in ["get_order_status", "cancel_order"] and not order_id_str)
        or (intent in ["cancel_order", "order_tracking"] and not order_id_str)
    ):
      return {
          "status": "needs_clarification",
          "intent": intent,
          "response": "برای پیگیری یا ثبت درخواست، لطفاً شماره سفارش عددی خود را ارسال فرمایید.",
          "decision": decision,
      }

    if requires_confirmation or intent == "cancel_order":
      return {
          "status": "waiting_for_confirmation",
          "intent": intent,
          "pending_order_id": order_id_str,
          "response": f"آیا اطمینان قطعی دارید که می‌خواهید سفارش {order_id_str} را لغو کنید؟ لطفاً با «بله» یا «خیر» پاسخ دهید.",
          "decision": decision,
      }

    tool_output = ""
    if requires_tool:
      if tool_name == "get_order_status":
        res = self.tools.get_order_status(order_id_str)
        tool_output = res.get("message")
      elif tool_name == "escalate_shipping":
        res = self.tools.escalate_shipping(order_id_str)
        tool_output = res.get("message")
      elif tool_name == "check_inventory":
        res = self.tools.check_inventory(entities.get("product_name", ""))
        tool_output = res.get("message")
      elif tool_name == "rag_policy_search":
        query_text = (
            intent if intent != "general_inquiry" else user_text
        )
        res = self.tools.rag_policy_search(query_text)
        tool_output = res.get("message")
      elif tool_name == "register_complaint":
        res = self.tools.register_complaint(user_text)
        tool_output = res.get("message")
      else:
        tool_output = "درخواست شما دریافت شد و در حال پیگیری است."
    else:
      tool_output = "پیام شما دریافت شد. در صورت نیاز به راهنمایی در خدمتیم."

    return {
        "status": "completed",
        "intent": intent,
        "response": tool_output,
        "decision": decision,
    }