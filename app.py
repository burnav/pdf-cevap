import os
import logging
import threading
import requests
from flask import Flask
from pypdf import PdfReader
import telebot

# Logging Ayarları
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Çevre Değişkenleri
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://burnav-go2-patent-asistani4.hf.space/gradio_api/call/predict")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN çevre değişkeni bulunamadı!")

# Bot Nesnesini Başlat
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Flask Web Sunucusu (Render Health Check İçin)
web_app = Flask(__name__)

@web_app.route('/')
def health_check():
    return "Bot aktif ve çalışıyor!", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    # Werkzeug log gürültüsünü engelle
    import logging as flask_log
    flask_log.getLogger('werkzeug').setLevel(flask_log.ERROR)
    web_app.run(host="0.0.0.0", port=port)

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
    """Hugging Face Gradio SSE API'sine istek gönderir."""
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
        # 1. Olay çağrısını başlat
        response = requests.post(HF_SPACE_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        event_id = response.json().get("event_id")

        if not event_id:
            return "Modelden yanıt kimliği (event_id) alınamadı."

        # 2. Sonucu SSE endpoint'inden al
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

# Telegram Komut & Mesaj Dinleyicileri
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Merhaba! SSS rehberimiz üzerinden sorularınızı yanıtlamaya hazırım. Sorunuzu iletebilirsiniz.")

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    status_msg = bot.reply_to(message, "Yanıt hazırlanıyor, lütfen bekleyiniz...")
    bot_response = query_hf_space(message.text)
    bot.edit_message_text(chat_id=status_msg.chat.id, message_id=status_msg.message_id, text=bot_response)

if __name__ == "__main__":
    # 1. Flask Web Sunucusunu Arka Plan Thread'inde Başlat
    threading.Thread(target=run_flask, daemon=True).start()
    logger.info("Flask Web Sunucu başlatıldı.")

    # 2. Telegram Bot Polling'i Ana Thread'de Çalıştır
    logger.info("Telegram Botu dinlemeye geçiyor...")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
