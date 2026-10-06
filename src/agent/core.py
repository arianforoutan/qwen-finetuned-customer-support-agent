
import re
from typing import Any, Dict, Optional
import ollama
from src.agent.tools import SupportTools
from src.agent.fast_router import FastRouting


class CustomerSupportAgent:

    def __init__(self, chat_model_name: str = "qwen-support"):
        self.chat_model_name = chat_model_name
        self.tools = SupportTools()
        self.router = FastRouting()

    def _extract_order_id(self, text: str):
        match = re.search(r"\b\d{4,8}\b", text)
        return match.group(0) if match else None

    def _generate_natural_response(self, user_text: str, intent: str, system_data: str) -> str:
        prompt = f"""نقش: شما اپراتور پشتیبانی فروشگاه هستید (نه خریدار).
      پیام مشتری: "{user_text}"
      نیت مشتری: {intent}
      اطلاعات پایگاه داده سیستم: "{system_data}"

       دستورالعمل: با رعایت ادب و صمیمیت، پاسخی کوتاه از زبان پشتیبان برای مشتری بنویسید. در صورتی که نیاز به تایید لغو بود، از مشتری سوال بپرسید."""

        try:
            res = ollama.chat(
                model=self.chat_model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            return res["message"]["content"].strip()
        except Exception:
            return system_data

    def handle_message(
        self, user_text: str, session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        cleaned_input = user_text.strip().lower()

        if session_context and session_context.get("status") == "waiting_for_confirmation":
            positive_tokens = ["بله", "اره", "آره", "تایید", "حتما", "کنسل کن", "لغوش کن", "yes"]
            negative_tokens = ["خیر", "نه", "دست نگه دار", "نمیخوام", "کنسل نکن", "no"]

            target_id = session_context.get("pending_order_id")

            if any(tok in cleaned_input for tok in positive_tokens):
                action_res = self.tools.cancel_order(target_id)
                final_msg = self._generate_natural_response(
                    user_text, "order_cancellation_confirmed", action_res.get("message", "")
                )
                return {
                    "status": "completed",
                    "intent": "order_cancellation_confirmed",
                    "response": final_msg
                }
            elif any(tok in cleaned_input for tok in negative_tokens):
                return {
                    "status": "cancelled_by_user",
                    "intent": "order_cancellation_aborted",
                    "response": "فرآیند لغو سفارش متوقف شد و سفارش شما همچنان فعال است."
                }
            else:
                session_context = None

        decision = self.router.route(user_text)
        intent = decision.get("intent", "general_inquiry")
        requires_tool = decision.get("requires_tool", False)
        tool_name = decision.get("tool")

        order_id_str = self._extract_order_id(user_text)

        knowledge_keywords = ["چند روز", "پس بدم", "مرجوع", "گارانتی", "قوانین", "شرایط بازگشت", "استرداد"]
        if any(kw in cleaned_input for kw in knowledge_keywords) and not order_id_str:
            res = self.tools.rag_policy_search(user_text)
            final_msg = self._generate_natural_response(user_text, "policy_inquiry", res.get("message", ""))
            return {
                "status": "completed",
                "intent": "policy_inquiry",
                "response": final_msg,
                "decision": decision
            }

        if (requires_tool and tool_name in ["get_order_status", "cancel_order"] and not order_id_str) or \
           (intent in ["cancel_order", "order_tracking"] and not order_id_str):
            clarification_raw = "برای پیگیری یا ثبت درخواست، لطفاً شماره سفارش عددی خود را ارسال فرمایید."
            final_msg = self._generate_natural_response(user_text, "needs_clarification", clarification_raw)
            return {
                "status": "needs_clarification",
                "intent": intent,
                "response": final_msg,
                "decision": decision
            }

        if intent == "cancel_order":
            confirm_raw = f"آیا مطمئن هستید که می‌خواهید سفارش {order_id_str} را لغو کنید؟ لطفاً با «بله» یا «خیر» اعلام کنید."
            final_msg = self._generate_natural_response(user_text, "waiting_for_confirmation", confirm_raw)
            return {
                "status": "waiting_for_confirmation",
                "intent": intent,
                "pending_order_id": order_id_str,
                "response": final_msg,
                "decision": decision
            }

        tool_output = ""
        
        if intent == "general_inquiry":
            tool_output = "کاربر پیامی عمومی و احوال‌پرسی فرستاده است؛ او را خوشامد بگویید و بپرسید چه کمکی از دست شما برمی‌آید."
            
        elif requires_tool:
            if tool_name == "get_order_status":
                res = self.tools.get_order_status(order_id_str)
                tool_output = res.get("message", "")
            elif tool_name == "escalate_shipping":
                res = self.tools.escalate_shipping(order_id_str)
                tool_output = res.get("message", "")
            elif tool_name == "check_inventory":
                res = self.tools.check_inventory(user_text)
                tool_output = res.get("message", "")
            elif tool_name == "rag_policy_search":
                res = self.tools.rag_policy_search(user_text)
                tool_output = res.get("message", "")
            elif tool_name == "register_complaint":
                res = self.tools.register_complaint(user_text)
                tool_output = res.get("message", "")
            else:
                tool_output = "درخواست شما دریافت شد و در حال پیگیری است."
        else:
            tool_output = "درخواست کاربر بدون نیاز به فراخوانی دیتابیس است. راهنمایی لازم را ارائه دهید."

        final_answer = self._generate_natural_response(user_text, intent, tool_output)

        return {
            "status": "completed",
            "intent": intent,
            "response": final_answer,
            "decision": decision
        }