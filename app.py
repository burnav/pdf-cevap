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
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

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

# --- RAG BÖLÜMÜ: PDF Okuma ve Parçalama (Chunking) ---

def load_and_chunk_pdf(pdf_path="sss.pdf", chunk_size=300):
    """
    PDF'i okur, paragraf/soru-cevap bloklarına göre parçalara (chunks) ayırır.
    """
    chunks = []
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        # Boş satırlardan veya soru başlıklarından mantıksal bölme yapalım
        raw_chunks = re.split(r'\n\s*\n', full_text)
        
        for raw in raw_chunks:
            cleaned = raw.strip()
            if len(cleaned) > 20:  # Çok kısa anlamsız satırları ele
                chunks.append(cleaned)
                
        logger.info(f"PDF başarıyla okundu. Toplam {len(chunks)} bilgi parçasına ayrıldı.")
    except Exception as e:
        logger.error(f"PDF işlenirken RAG hatası: {e}")
        chunks = ["SSS bilgisi yüklenemedi."]
    
    return chunks

# PDF Parçalarını Belleğe Al
FAQ_CHUNKS = load_and_chunk_pdf()

def retrieve_relevant_context(user_question: str, top_k=2) -> str:
    """
    Kullanıcının sorusu ile PDF parçaları arasındaki TF-IDF benzerliğini hesaplar
    ve en alakalı top_k adet parçayı birleştirip döndürür.
    """
    if not FAQ_CHUNKS:
        return ""
        
    try:
        # Soruyu ve tüm parçaları vektörleştir
        documents = FAQ_CHUNKS + [user_question]
        vectorizer = TfidfVectorizer().fit_transform(documents)
        vectors = vectorizer.toarray()

        # Son eleman kullanıcı sorusu, öncekiler doküman parçaları
        question_vector = vectors[-1]
        chunk_vectors = vectors[:-1]

        # Benzerlik skorlarını hesapla
        similarities = cosine_similarity([question_vector], chunk_vectors)[0]

        # En yüksek skora sahip top_k parçanın indekslerini al
        related_indices = similarities.argsort()[-top_k:][::-1]

        retrieved_texts = []
        for idx in related_indices:
            # Sadece belirli bir eşik değerinin üzerindeki anlamlı eşleşmeleri al (örneğin > 0.05)
            if similarities[idx] > 0.05:
                retrieved_texts.append(FAQ_CHUNKS[idx])

        # Eğer hiç eşleşme bulunamadıysa ilk 2 parçayı veya genel bağlamı ver
        if not retrieved_texts:
            retrieved_texts = FAQ_CHUNKS[:top_k]

        selected_context = "\n---\n".join(retrieved_texts)
        logger.info(f"RAG: {len(retrieved_texts)} adet alakalı parça seçildi.")
        return selected_context

    except Exception as e:
        logger.error(f"RAG arama sırasında hata: {e}")
        # Hata durumunda ilk parçayı dön
        return "\n---\n".join(FAQ_CHUNKS[:top_k])

# --- HUGGING FACE ISTEDI BÖLÜMÜ ---

def query_hf_space(user_question: str) -> str:
    """Hugging Face Gradio SSE API'sine RAG ile küçültülmüş bağlamı gönderir."""
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    # 1. RAG ile Müşteri Sorusuyla En Alakalı Parçayı Çek
    relevant_context = retrieve_relevant_context(user_question, top_k=2)

    # 2. Küçültülmüş ve Odaklanmış Prompt Oluştur
    prompt = (
        f"Aşağıdaki SSS bilgisine dayanarak soruyu yanıtla:\n"
        f"--- SSS İLGİLİ BÖLÜM ---\n{relevant_context}\n--- BİTİŞ ---\n\n"
        f"Müşteri Sorusu: {user_question}\n\n"
        f"Cevap:"
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
