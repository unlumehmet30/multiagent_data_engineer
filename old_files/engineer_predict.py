import os
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# --- AYARLAR ---
MODEL_NAME = "qwen2.5-coder:7b"
llm = ChatOllama(model=MODEL_NAME, temperature=0.1, num_ctx=8192)

def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def extract_code(text):
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def run_prediction_engineer():
    print("\n🔮 PREDICTION ENGINEER ÇALIŞIYOR...\n" + "="*40)
    
    # 1. Kaynak Kodu Oku (Referans)
    train_code = read_file("train.py")
    
    if not train_code:
        print("❌ 'train.py' bulunamadı! Ajanın referans alacağı kod yok.")
        return

    # 2. Prompt (Talimatlar)
    system_prompt = """
    You are a Senior ML Engineer.
    You previously wrote 'train.py' to train a LightGBM model.
    
    YOUR TASK:
    Write a separate script named 'predict.py' that loads the trained model and makes predictions on new dummy data.
    
    CRITICAL RULES:
    1. ANALYZE 'train.py' carefully. Look at which columns were dropped and which were kept (X).
    2. The dummy data in 'predict.py' MUST have the EXACT SAME columns/features as 'X' in 'train.py'.
    3. DATA TYPES: If 'train.py' used categorical features, ensure 'predict.py' casts them to 'category' dtype (df['col'] = df['col'].astype('category')).
    4. LOAD SAFELY: Use 'joblib.load'.
    5. OUTPUT format: Print the product name and the predicted sales amount.
    
    OUTPUT:
    Provide ONLY the Python code inside ```python ... ``` blocks.
    """
    
    user_msg = f"""
    Here is the 'train.py' code you wrote:
    ---
    {train_code}
    ---
    
    Now generate the corresponding 'predict.py'.
    Create logical dummy data for the test (e.g. 2-3 rows).
    """
    
    print("⏳ 'train.py' analiz ediliyor ve 'predict.py' yazılıyor...")
    
    # 3. Modeli Çağır
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ])
    
    # 4. Kodu Ayıkla ve Kaydet
    code = extract_code(response.content)
    
    if code:
        with open("predict.py", "w", encoding="utf-8") as f:
            f.write(code)
        print("\n✅ BAŞARILI: 'predict.py' oluşturuldu.")
        print("Şimdi 'python predict.py' komutunu çalıştırabilirsin.")
    else:
        print("\n❌ HATA: Kod üretilemedi.")

if __name__ == "__main__":
    run_prediction_engineer()