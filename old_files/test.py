import time
from langchain_ollama import ChatOllama

# --- AYARLAR ---
MODEL_NAME = "qwen2.5-coder:7b"
# 8GB VRAM için güvenli context limiti. 
# Bunu ileride ihtiyaca göre artıracağız ama başlangıç için 4096 ideal.
CONTEXT_WINDOW = 4096 

def test_connection():
    print(f"🔄 Model yükleniyor: {MODEL_NAME}...")
    print(f"💾 Hedeflenen Context Penceresi: {CONTEXT_WINDOW} token")
    
    try:
        # LLM Tanımlama
        llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0.1, # Daha tutarlı kod/mantık için düşük sıcaklık
            num_ctx=CONTEXT_WINDOW
        )

        start_time = time.time()
        
        # Basit bir mantık sorusu soralım
        query = "Bir Python listesindeki tekrar eden elemanları silmenin en performanslı yolu nedir? Tek cümleyle açıkla."
        print(f"\n❓ Soru: {query}")
        
        response = llm.invoke(query)
        
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ CEVAP:\n{response.content}")
        print(f"\n⏱️ Geçen Süre: {duration:.2f} saniye")
        print("🎉 Kurulum Başarılı! Sistem 8GB VRAM üzerinde çalışmaya hazır.")

    except Exception as e:
        print(f"\n❌ HATA: {e}")
        print("Lütfen 'ollama serve' komutunun çalıştığından veya model isminin doğru olduğundan emin ol.")

if __name__ == "__main__":
    test_connection()