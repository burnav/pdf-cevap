import os
import re
import time
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
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")  # Render tarafindan otomatik tanimlanir

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN çevre değişkeni bulunamadı!")

# Bot Nesnesini Başlat
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, threaded=True, num_threads=10)

# Flask Web Sunucusu (Render Health Check & Keep-Alive İçin)
web_app = Flask(__name__)

@web_app.route('/')
def health_check():
    return "Bot RAG desteği ile aktif ve çalışıyor!", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    import logging as flask_log
    flask_log.getLogger('werkzeug').setLevel(flask_log.ERROR)
    web_app.run(host="0.0.0.0", port=port)

# --- RENDER KİLİTLENME ENGELLEYİCİ (SELF-PING) ---

def keep_alive():
    """
    Render Free Tier'in 15 dakika hareketsizlik sonrasi sunucuyu 
    uyku moduna (spin down) almasini engellemek için her 10 dakikada 
    bir kendi kendine HTTP istegi atar.
    """
    url = RENDER_EXTERNAL_URL or "http://127.0.0.1:10000/"
    logger.info(f"Keep-alive servisi başlatıldı. Hedef URL: {url}")
    while True:
        time.sleep(600)  # 10 dakikada bir (600 saniye)
        try:
            requests.get(url, timeout=10)
            logger.info("Keep-alive ping başarılı (Sunucu uyanık tutuldu).")
        except Exception as e:
            logger.warning(f"Keep-alive ping hatası: {e}")

# --- PROFESYONEL RAG BÖLÜMÜ: SSS Yapısına Özel Akıllı Chunking ---

def load_and_chunk_pdf(pdf_path="sss.pdf"):
    """
    PDF'i okur ve metni rastgele karakterler yerine SSS yapisina uygun 
    sekilde '---' veya [SORU] ayracina gore anlamli bloklara boler.
    """
    chunks = []
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        if not full_text.strip():
            logger.warning("PDF boş veya metin okunamadı.")
            return ["SSS bilgisi yüklenemedi."]

        # 1. Metni '---' ayracina gore bloklara bol (En temiz SSS chunking yontemi)
        raw_blocks = full_text.split("---")
        
        for block in raw_blocks:
            cleaned_block = block.strip()
            # Cok kisa veya anlamsiz bloklari ele
            if len(cleaned_block) > 20:
                chunks.append(cleaned_block)

        # Eger '---' ile bolunemediysa alternatif bolumleme yap
        if len(chunks) <= 1:
            chunks = [c.strip() for c in re.split(r'\n(?=\[SORU\])', full_text) if len(c.strip()) > 20]

        logger.info(f"PDF başarıyla okundu. Toplam {len(chunks)} anlamli SSS bloğuna ayrıldı.")
    except Exception as e:
        logger.error(f"PDF işlenirken RAG hatası: {e}")
        chunks = ["SSS bilgisi yüklenemedi."]
    
    return chunks

# PDF Parçalarını Belleğe Al
FAQ_CHUNKS = load_and_chunk_pdf()

def retrieve_relevant_context(user_question: str, top_k=2) -> str:
    """
    Kullanıcının sorusu ile SSS blokları arasındaki TF-IDF benzerliğini hesaplar
    ve en alakalı soru-cevap bloklarını döndürür.
    """
    if not FAQ_CHUNKS:
        return ""
        
    try:
        # Metinleri arama öncesi normalize et
        documents = FAQ_CHUNKS + [user_question]
        vectorizer = TfidfVectorizer(
    analyzer='char_wb',      # Kelime sınırları içinde karakter tabanlı arama yapar
    ngram_range=(3, 5)       # Kelimeleri 3, 4 ve 5'erli harf öbeklerine böler
).fit_transform(documents)
        vectors = vectorizer.toarray()

        question_vector = vectors[-1]
        chunk_vectors = vectors[:-1]

        similarities = cosine_similarity([question_vector], chunk_vectors)[0]
        related_indices = similarities.argsort()[-top_k:][::-1]

        retrieved_texts = []
        for idx in related_indices:
            # Benzerlik esigi
            if similarities[idx] > 0.01:
                retrieved_texts.append(FAQ_CHUNKS[idx])

        if not retrieved_texts:
            retrieved_texts = [FAQ_CHUNKS[0]]

        logger.info(f"RAG: {len(FAQ_CHUNKS)} bloktan en alakalı {len(retrieved_texts)} blok seçildi.")
        return "\n\n=== İLGİLİ SSS BİLGİSİ ===\n\n".join(retrieved_texts)

    except Exception as e:
        logger.error(f"RAG arama sırasında hata: {e}")
        return FAQ_CHUNKS[0] if FAQ_CHUNKS else ""

# --- HUGGING FACE İSTEĞİ BÖLÜMÜ ---

def query_hf_space(user_question: str) -> str:
    """Hugging Face Gradio SSE API'sine sıkı prompt ile istek atar."""
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    relevant_context = retrieve_relevant_context(user_question, top_k=2)

    prompt = (
        f"Sen Go2Patents firmasının resmi müşteri temsilcisisin.\n"
        f"GÖREVİN: Müşterinin sorusunu SADECE VE SADECE aşağıdaki BİLGİ metnine dayanarak yanıtlamaktır.\n\n"
        f"KATI KURALLAR:\n"
        f"1. Kendi genel kültür bilgini, dış bilgileri veya tahminlerini KESİNLİKLE KULLANMA.\n"
        f"2. Müşterinin sorduğu sorunun cevabı BİLGİ metninde AÇIKÇA geçmiyorsa, doğrudan ve aynen şu kelimelerle yanıt ver:\n"
        f"   \"Bilgi için https://www.go2patents.com/ iletişim formumuz üzerinden bize ulaşabilirsiniz.\"\n"
        f"3. Yanıtında asla kendi cümleni ekleme, uyarlama yapma veya spora/farklı konulara atıfta bulunma.\n"
        f"4. Sadece metinde geçen telefon, e-posta veya web adresi gibi bilgileri tam olarak aktar.\n\n"
        f"--- BİLGİ BAŞLANGICI ---\n{relevant_context}\n--- BİLGİ BİTİŞİ ---\n\n"
        f"Müşteri Sorusu: {user_question}\n\n"
        f"Yanıt:"
    )

    payload = {"data": [prompt]}

    try:
        response = requests.post(HF_SPACE_URL, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        event_id = response.json().get("event_id")

        if not event_id:
            return "Modelden yanıt kimliği (event_id) alınamadı."

        result_url = f"{HF_SPACE_URL}/{event_id}"
        result_response = requests.get(result_url, headers=headers, timeout=40)
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
        return "Yanıt süresi aşıldı. Lütfen sorunuzu tekrar iletiniz."
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
    bot.reply_to(message, "Merhaba! Go2Patents müşteri asistanıyım. Sorularınızı iletebilirsiniz.")

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    threading.Thread(target=process_message_async, args=(message,), daemon=True).start()

if __name__ == "__main__":
    # 1. Flask Web Sunucusunu Başlat
    threading.Thread(target=run_flask, daemon=True).start()
    logger.info("Flask Web Sunucu başlatıldı.")

    # 2. Render Uykuyu Önleme Servisini Başlat (Self-Ping)
    threading.Thread(target=keep_alive, daemon=True).start()

    # 3. Telegram Polling
    logger.info("Telegram Botu dinlemeye geçiyor...")
    try:
        bot.remove_webhook()
    except Exception as e:
        logger.warning(f"Webhook kaldırılırken uyarı: {e}")

    bot.infinity_polling(timeout=10, long_polling_timeout=5, skip_pending=True)
