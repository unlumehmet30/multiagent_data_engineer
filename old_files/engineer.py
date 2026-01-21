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
        return f"MISSING FILE: {filename}"

def extract_code(text, language="python"):
    """
    LLM cevabından kod bloklarını ayıklar.
    """
    pattern = rf"```{language}(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text # Kod bloğu bulamazsa metni döndürsün (riskli ama deneriz)

def run_engineer():
    print("\n👷 ML ENGINEER (MÜHENDİS) ÇALIŞIYOR...\n" + "="*40)
    
    # 1. Girdileri Topla
    design_doc = read_file("final_design_doc.md")
    analyst_report = read_file("analyst_report.txt")
    
    if "MISSING FILE" in design_doc or "MISSING FILE" in analyst_report:
        print("❌ Eksik dosyalar var! Lütfen önce analyst.py ve design_team.py çalıştırın.")
        return

    # 2. System Prompt (Çok Sıkı Kurallar)
    system_prompt = """
    You are a Senior Machine Learning Engineer.
    Your goal is to write the PRODUCTION CODE based on the Design Document.
    
    INPUTS:
    1. Analyst Report (Data schema, column names).
    2. Design Doc (Model choice, preprocessing steps).
    
    TASK:
    Write a complete Python script named 'train.py'.
    
    CODING STANDARDS:
    - Use `scikit-learn` Pipelines and ColumnTransformer.
    - Create a class named `ModelTrainer`.
    - Include `train()` and `save_model()` methods.
    - Handle missing values (SimpleImputer) automatically.
    - Use `joblib` to save the model.
    - The code MUST BE COMPLETE. No placeholders like '# code goes here'.
    - Assume the data file is 'satis_verisi.csv'.
    
    OUTPUT FORMAT:
    Provide ONLY the Python code inside ```python ... ``` blocks.
    """
    
    user_msg = f"""
    Here is the Data Info:
    {analyst_report}
    
    Here is the Design Doc:
    {design_doc}
    
    Generate 'train.py' now.
    """
    
    print("⏳ Kod yazılıyor (Bu işlem model hızına göre 30-60sn sürebilir)...")
    
    # 3. Modeli Çağır
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ])
    
    # 4. Kodu Ayıkla ve Kaydet
    full_response = response.content
    code = extract_code(full_response, "python")
    
    if code:
        with open("train.py", "w", encoding="utf-8") as f:
            f.write(code)
        print("\n✅ BAŞARILI: 'train.py' dosyası oluşturuldu.")
        print(f"📦 Kod Boyutu: {len(code)} karakter.")
    else:
        print("\n❌ HATA: Model kod bloğu üretmedi. Cevabı kontrol et:")
        print(full_response)

    # 5. Requirements Dosyasını İste (Bonus)
    print("\n📦 Requirements.txt oluşturuluyor...")
    req_response = llm.invoke("Based on the 'train.py' you just wrote, list the library names for requirements.txt. Only the list.")
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(req_response.content)
    print("✅ 'requirements.txt' oluşturuldu.")

if __name__ == "__main__":
    run_engineer()