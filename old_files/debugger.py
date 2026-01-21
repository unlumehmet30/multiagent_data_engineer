import sys
import re
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# --- AYARLAR ---
MODEL_NAME = "qwen2.5-coder:7b"
llm = ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=8192)

def read_file(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def extract_code(text):
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def run_debugger():
    print("\n🚑 DEBUGGER (TEMİZLİKÇİ) ÇALIŞIYOR...\n" + "="*40)
    
    broken_code = read_file("train.py")
    if not broken_code:
        print("❌ 'train.py' bulunamadı!")
        return

    # ÖZEL HATA TANIMI: Input y contains NaN
    error_log = """
    ValueError: Input y contains NaN.
    
    Diagnosis:
    The target variable (y) has missing values (NaN). 
    Scikit-learn cannot train if the Target variable has missing rows.
    """

    system_prompt = """
    You are a Senior Python Expert. 
    The 'train.py' script is failing because the TARGET variable has missing values (NaN).
    
    YOUR TASK:
    1. Analyze the code to find where the DataFrame is loaded (pd.read_csv).
    2. IMMEDIATELY after loading, ADD a line to drop rows where the target is NaN.
    3. The target column is likely 'Satis_Adedi' (or check what is assigned to 'y').
    4. Example fix: 
       df = pd.read_csv(...)
       df = df.dropna(subset=['Satis_Adedi']) # <--- ADD THIS
    
    5. Keep the rest of the code exactly the same.
    6. Output ONLY the corrected Python code.
    """
    
    user_msg = f"""
    --- CODE WITH DATA ISSUE ---
    {broken_code}
    
    --- ERROR ---
    {error_log}
    
    Fix the code by dropping NaN values in the target column.
    """
    
    print("⏳ Kod temizleniyor (NaN satırları silme kodu ekleniyor)...")
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ])
    
    fixed_code = extract_code(response.content)
    
    if fixed_code:
        with open("train.py", "w", encoding="utf-8") as f:
            f.write(fixed_code)
        print("\n✅ DÜZELTİLDİ: 'train.py' güncellendi.")
        print("Hedef değişkendeki boş satırlar artık silinecek.")
    else:
        print("❌ HATA: Kod düzeltilemedi.")

if __name__ == "__main__":
    run_debugger()