import os
import logging
import requests
from pypdf import PdfReader
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Logging ayarları
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Çevre Değişkenleri (Render'dan çekilecek)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://burnav-go2-patent-asistani4.hf.space/gradio_api/call/predict")

# sss.pdf dosyasından metin çıkarma
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

# PDF içeriğini belleğe al
FAQ_CONTEXT = get_pdf_text()

def query_hf_space(user_question: str) -> str:
    """Hugging Face Gradio SSE API'sine istek gönderir."""
    headers = {
        "Content-Type": "application/json"
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    # Hugging Face Modelinize giden istem (Prompt) yapısı
    prompt = (
        f"Aşağıda SSS belgesinden alınan bilgiler yer almaktadır:\n"
        f"--- SSS BAŞLANGICI ---\n{FAQ_CONTEXT}\n--- SSS BİTİŞİ ---\n\n"
        f"Müşteri Sorusu: {user_question}\n\n"
        f"Lütfen yukarıdaki SSS bilgilerine dayanarak müşterinin sorusuna açık ve net bir cevap ver."
    )

    payload = {
        "data": [prompt]
    }

    try:
        # 1. Aşama: Olay çağrısını başlat ve event_id al
        response = requests.post(HF_SPACE_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        event_id = response.json().get("event_id")

        if not event_id:
            return "Modelden yanıt kimliği (event_id) alınamadı."

        # 2. Aşama: Olay sonucunu SSE (Server-Sent Events) endpoint'inden çek
        result_url = f"{HF_SPACE_URL}/{event_id}"
        result_response = requests.get(result_url, headers=headers, timeout=60)
        result_response.raise_for_status()

        # SSE çıktısını parse et
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Merhaba! SSS rehberimiz üzerinden sorularınızı yanıtlamaya hazırım. Sorunuzu iletebilirsiniz.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Müşteriye bekleniyor bilgisi ver
    status_message = await update.message.reply_text("Yanıt hazırlanıyor, lütfen bekleyiniz...")
    
    bot_response = query_hf_space(user_text)
    
    await status_message.edit_text(bot_response)

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN çevre değişkeni bulunamadı!")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot başlatılıyor...")
    app.run_polling()

if __name__ == "__main__":
    main()