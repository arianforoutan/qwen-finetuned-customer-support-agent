from laya import Router


class FastRouting:
    def __init__(self):
        self.router = Router(preload=True)

        self.questions = {
            "intent": {
                "type": "choice",
                "instructions": "موضوع یا هدف اصلی پیام کاربر چیست؟",
                "criteria": {
                    "check_inventory": "استعلام موجود بودن یا نبودن یک کالا، خرید محصول جدید، قیمت کالا یا گوشی و لپ‌تاپ",
                    "order_tracking": "پیگیری سفارش ثبت شده قبلی، کد رهگیری، وضعیت مرسوله یا بسته پستی",
                    "cancel_order": "لغو یا انصراف از خرید، کنسل کردن سفارش ثبت‌شده، پس دادن پول",
                    "policy_inquiry": "قوانین سایت، شرایط مرجوعی، گارانتی، هزینه پست یا مهلت تست",
                    "general_inquiry": "سلام، احوال‌پرسی، خسته نباشید یا تشکر بدون درخواست کالا یا سفارش"
                }
            },
            "requires_tool": {
                "type": "choice",
                "instructions": "آیا برای پاسخ نیاز به بررسی دیتابیس (سفارش یا انبار) هست؟",
                "criteria": {
                    "true": "بررسی سفارش، لغو سفارش، یا بررسی موجودی و قیمت کالا در انبار",
                    "false": "احوال‌‌پرسی، چت متفرقه یا سوال درباره قوانین عمومی"
                }
            }
        }

    def route(self, user_message: str):
            state = {"message": user_message}
            decision = self.router.predict(state , self.questions)

            intent = decision["answers"]["intent"]["choice"]
            requires_tool = decision["answers"]["requires_tool"]["choice"] == "true"


            tool_mapping = {
            "order_tracking": "get_order_status",
            "cancel_order": "cancel_order",
            "return_request": "request_return",
            "change_address": "update_shipping_address"
            }

            selected_tool = tool_mapping.get(intent, None) if requires_tool else None
        
            return {
            "intent": intent,
            "requires_tool": requires_tool,
            "tool": selected_tool,
            "requires_confirmation": intent == "cancel_order",
            "needs_clarification": False
        }



