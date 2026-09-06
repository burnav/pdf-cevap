import os
import re
import logging
import threading
import requests
from flask import Flask
from pypdf import PdfReader
import telebot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Logging Ayarları
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Çevre Değişkenleri
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL", 
    "https://burnav-go2-patent-asistani4.hf.space/gradio_api/call/predict"
).strip("[]()'\" ")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN çevre değişkeni bulunamadı!")

# Bot Nesnesini Başlat
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, threaded=True, num_threads=10)

# Flask Web Sunucusu (Render Health Check İçin)
web_app = Flask(__name__)

@web_app.route('/')
def health_check():
    return "Bot RAG desteği ile aktif ve çalışıyor!", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    import logging as flask_log
    flask_log.getLogger('werkzeug').setLevel(flask_log.ERROR)
    web_app.run(host="0.0.0.0", port=port)

# --- RAG BÖLÜMÜ: Dinamik PDF Parçalama (Chunking) ---

def load_and_chunk_pdf(pdf_path="sss.pdf", chunk_size=800, overlap=100):
    """
    PDF'i okur ve metni sabit karakter boyutlarında (chunk_size) ve 
    birbirini örten (overlap) küçük parçalara böler.
    """
    chunks = []
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        # Gereksiz birden fazla boşluğu ve satır sonlarını temizle
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        if not full_text:
            logger.warning("PDF boş veya metin okunamadı.")
            return ["SSS bilgisi yüklenemedi."]

        # Sabit boyutlu parçalara böl (Karakter bazlı sliding window)
        start = 0
        text_length = len(full_text)

        while start < text_length:
            end = start + chunk_size
            chunk = full_text[start:end]
            
            # Kelimenin ortasından bölünmesini engellemek için son boşluktan kes
            if end < text_length:
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    chunk = chunk[:last_space]
                    end = start + last_space

            cleaned_chunk = chunk.strip()
            if len(cleaned_chunk) > 30:
                chunks.append(cleaned_chunk)
            
            # Bir sonraki parçaya overlap (örtüşme) kadar geriden başla
            start = end - overlap if (end - overlap) > start else end

        logger.info(f"PDF başarıyla okundu. Toplam {len(chunks)} küçük parçaya ayrıldı.")
    except Exception as e:
        logger.error(f"PDF işlenirken RAG hatası: {e}")
        chunks = ["SSS bilgisi yüklenemedi."]
    
    return chunks

# PDF Parçalarını Belleğe Al
FAQ_CHUNKS = load_and_chunk_pdf(chunk_size=800, overlap=100)

def retrieve_relevant_context(user_question: str, top_k=3) -> str:
    """
    Kullanıcının sorusu ile PDF parçaları arasındaki TF-IDF benzerliğini hesaplar
    ve en alakalı parçaları döndürür.
    """
    if not FAQ_CHUNKS:
        return ""
        
    try:
        documents = FAQ_CHUNKS + [user_question]
        vectorizer = TfidfVectorizer().fit_transform(documents)
        vectors = vectorizer.toarray()

        question_vector = vectors[-1]
        chunk_vectors = vectors[:-1]

        similarities = cosine_similarity([question_vector], chunk_vectors)[0]
        related_indices = similarities.argsort()[-top_k:][::-1]

        retrieved_texts = []
        for idx in related_indices:
            if similarities[idx] > 0.02:
                retrieved_texts.append(FAQ_CHUNKS[idx])

        if not retrieved_texts:
            retrieved_texts = [FAQ_CHUNKS[0]]

        logger.info(f"RAG: {len(FAQ_CHUNKS)} parçadan en alakalı {len(retrieved_texts)} parça seçildi.")
        return "\n---\n".join(retrieved_texts)

    except Exception as e:
        logger.error(f"RAG arama sırasında hata: {e}")
        return FAQ_CHUNKS[0] if FAQ_CHUNKS else ""

# --- HUGGING FACE İSTEĞİ BÖLÜMÜ ---

def query_hf_space(user_question: str) -> str:
    """Hugging Face Gradio SSE API'sine geliştirilmiş prompt ile istek atar."""
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    relevant_context = retrieve_relevant_context(user_question, top_k=3)

    # Bağlantıları ve URL'leri açıkça vermesi için yönlendirilmiş Prompt
    prompt = (
        f"Sen yardımsever bir müşteri temsilcisisin. Aşağıda sağlanan bilgiye sadık kalarak kullanıcının sorusunu yanıtla.\n"
        f"ÖNEMLİ KURALLAR:\n"
        f"1. Sağlanan metinde geçen web adresi, URL, e-posta veya iletişim bilgilerini tam ve açık halleriyle yaz. Asla 'LINK' veya yer tutucu ibareler kullanma.\n"
        f"2. Sadece sağlanan bilgideki verilere dayanarak cevap ver.\n\n"
        f"--- SAĞLANAN BİLGİ ---\n{relevant_context}\n--- BİLGİ BİTİŞİ ---\n\n"
        f"Müşteri Sorusu: {user_question}\n\n"
        f"Yanıt:"
    )

    payload = {"data": [prompt]}

    try:
        response = requests.post(HF_SPACE_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        event_id = response.json().get("event_id")

        if not event_id:
            return "Modelden yanıt kimliği (event_id) alınamadı."

        result_url = f"{HF_SPACE_URL}/{event_id}"
        result_response = requests.get(result_url, headers=headers, timeout=30)
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

    except requests.exceptions.Timeout:
        logger.error("Hugging Face isteği zaman aşımına uğradı.")
        return "Yanıt süresi aşıldı. Lütfen tekrar deneyiniz."
    except Exception as e:
        logger.error(f"HF Space isteğinde hata: {e}")
        return "Üzgünüm, şu an yanıt üretilemiyor. Lütfen daha sonra tekrar deneyiniz."

# --- TELEGRAM MESAJ İŞLEME (THREADED) ---

def process_message_async(message):
    try:
        status_msg = bot.reply_to(message, "Yanıt hazırlanıyor, lütfen bekleyiniz...")
        bot_response = query_hf_space(message.text)
        bot.edit_message_text(
            chat_id=status_msg.chat.id, 
            message_id=status_msg.message_id, 
            text=bot_response
        )
    except Exception as e:
        logger.error(f"Mesaj işlenirken hata oluştu: {e}")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Merhaba! SSS rehberimiz üzerinden sorularınızı yanıtlamaya hazırım. Sorunuzu iletebilirsiniz.")

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    threading.Thread(target=process_message_async, args=(message,), daemon=True).start()

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    logger.info("Flask Web Sunucu başlatıldı.")

    logger.info("Telegram Botu dinlemeye geçiyor...")
    try:
        bot.remove_webhook()
    except Exception as e:
        logger.warning(f"Webhook kaldırılırken uyarı: {e}")

    bot.infinity_polling(timeout=10, long_polling_timeout=5, skip_pending=True)
