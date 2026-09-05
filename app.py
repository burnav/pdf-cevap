import os
import logging
import asyncio
import requests
from flask import Flask, request
from pypdf import PdfReader
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Logging Ayarları
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Çevre Değişkenleri
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://burnav-go2-patent-asistani4.hf.space/gradio_api/call/predict")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")  # Render otomatik sunar (örn: https://app.onrender.com)

# SSS.pdf Metin Okuma
def get_pdf_text(pdf_path="sss.pdf"):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        logger.error(f"PDF okunurken hata oluştu: {e}")
    return text

FAQ_CONTEXT = get_pdf_text()

def query_hf_space(user_question: str) -> str:
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    prompt = (
        f"Aşağıda SSS belgesinden alınan bilgiler yer almaktadır:\n"
        f"--- SSS BAŞLANGICI ---\n{FAQ_CONTEXT}\n--- SSS BİTİŞİ ---\n\n"
        f"Müşteri Sorusu: {user_question}\n\n"
        f"Lütfen yukarıdaki SSS bilgilerine dayanarak müşterinin sorusuna açık ve net bir cevap ver."
    )

    payload = {"data": [prompt]}

    try:
        response = requests.post(HF_SPACE_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        event_id = response.json().get("event_id")

        if not event_id:
            return "Modelden yanıt kimliği (event_id) alınamadı."

        result_url = f"{HF_SPACE_URL}/{event_id}"
        result_response = requests.get(result_url, headers=headers, timeout=60)
        result_response.raise_for_status()

        lines = result_response.text.strip().split("\n")
        for line in lines:
            if line.startswith("data:"):
                import json
                data_str = line.replace("data:", "").strip()
                data_json = json.loads(data_str)
                if isinstance(data_json, list) and len(data_json) > 0:
                    return str(data_json[0])

        return "Yanıt işlenirken bir sorun oluştu."

    except Exception as e:
        logger.error(f"HF Space isteğinde hata: {e}")
        return "Üzgünüm, şu an yanıt üretilemiyor. Lütfen daha sonra tekrar deneyiniz."

# Telegram Bot Uygulamasını Oluştur
telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Merhaba! SSS rehberimiz üzerinden sorularınızı yanıtlamaya hazırım. Sorunuzu iletebilirsiniz.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    status_message = await update.message.reply_text("Yanıt hazırlanıyor, lütfen bekleyiniz...")
    bot_response = query_hf_space(user_text)
    await status_message.edit_text(bot_response)

# Handler'ları Ekle
telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Flask Web Sunucusu
web_app = Flask(__name__)

@web_app.route('/', methods=['GET'])
def health_check():
    return "Bot aktif ve Webhook modunda çalışıyor!", 200

@web_app.route(f'/{TELEGRAM_BOT_TOKEN}', methods=['POST'])
def webhook():
    """Telegram'dan gelen bildirimleri işler."""
    if request.method == "POST":
        update_data = request.get_json(force=True)
        update = Update.de_json(update_data, telegram_app.bot)
        
        # Async telegram_app çağrısını senkron Flask rotasında çalıştır
        asyncio.run(telegram_app.initialize())
        asyncio.run(telegram_app.process_update(update))
        return "OK", 200
    return "Invalid", 400

def set_telegram_webhook():
    """Telegram'a Webhook adresini bildirir."""
    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{TELEGRAM_BOT_TOKEN}"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={webhook_url}"
        try:
            res = requests.get(url, timeout=10)
            logger.info(f"Webhook kurulum sonucu: {res.json()}")
        except Exception as e:
            logger.error(f"Webhook ayarlanırken hata: {e}")

if __name__ == "__main__":
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN bulunamadı!")
        
    # Telegram Webhook Kaydını Yap
    set_telegram_webhook()
    
    # Flask Sunucusunu Başlat
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)
