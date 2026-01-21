import subprocess
import sys
import os
import re
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

def run_script(script_name):
    """Scripti çalıştırır ve (başarı, çıktı/hata) döner."""
    print(f"\n⚙️  ÇALIŞTIRILIYOR: {script_name}...")
    result = subprocess.run(
        [sys.executable, script_name], 
        capture_output=True, 
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    
    if result.returncode == 0:
        return True, result.stdout
    else:
        return False, result.stderr

def fix_code(script_name, error_log):
    print(f"\n🚑 TAMİR MODU: {script_name} için çözüm aranıyor...")
    
    # Bozuk kodu ve referans olabilecek train dosyasını oku
    broken_code = read_file(script_name)
    train_code = read_file("train.py") # Context için çok önemli!
    
    system_prompt = """
    You are an Expert Python Debugger.
    The user is trying to run a script, but it failed.
    
    TASK:
    1. Analyze the Error Log and the Broken Code.
    2. Check 'train.py' (if provided) to understand how the model was trained (features, preprocessing).
    3. FIX the broken code.
    
    COMMON ISSUES IN ML:
    - Shape Mismatch: Predict data has different columns than Train data.
    - Type Errors: Sending strings (Object) to a model expecting Int/Float.
    - Date Parsing: Datetime objects causing Float errors.
    
    OUTPUT:
    Return ONLY the full corrected Python code inside ```python ... ``` blocks.
    """
    
    user_msg = f"""
    --- BROKEN SCRIPT ({script_name}) ---
    {broken_code}
    
    --- REFERENCE (train.py) ---
    {train_code}
    
    --- ERROR LOG ---
    {error_log}
    
    Fix the code to match the training logic and resolve the error.
    """
    
    print("⏳ LLM Çözüm Üretiyor...")
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ])
    
    fixed_code = extract_code(response.content)
    
    if fixed_code and len(fixed_code) > 50:
        # Kodu kaydet
        with open(script_name, "w", encoding="utf-8") as f:
            f.write(fixed_code)
        print("✅ Kod güncellendi.")
        return True
    else:
        print("❌ LLM geçerli bir kod üretemedi.")
        return False

def auto_heal(target_script, max_retries=3):
    print(f"🤖 OTONOM DEBUG BAŞLATILDI: {target_script}")
    print("="*40)
    
    for attempt in range(max_retries):
        print(f"\n🔄 DENEME {attempt + 1}/{max_retries}")
        
        # 1. Çalıştır
        success, output = run_script(target_script)
        
        # 2. Başarılıysa bitir
        if success:
            print("\n🎉 BAŞARILI! Script hatasız çalıştı.")
            print("-" * 20)
            print(output) # Çıktıyı göster
            return
        
        # 3. Hataysa tamir et
        print(f"\n💥 HATA ALGILANDI:\n{output.strip()[-500:]}") # Son 500 karakteri göster
        
        fix_success = fix_code(target_script, output)
        
        if not fix_success:
            print("🚫 Tamir edilemedi, işlem durduruluyor.")
            break

if __name__ == "__main__":
    # Hangi dosyayı tamir edeceğini argüman olarak al veya varsayılan predict.py olsun
    target = "predict.py" 
    auto_heal(target)