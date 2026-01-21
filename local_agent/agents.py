import subprocess
import os
import re
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_experimental.utilities import PythonREPL
from state import AgentState

# Script'in bulunduğu dizin (dosya yolları için)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- AYARLAR ---
# Hibrit Model Yapısı (VRAM Dostu)
# 1. Düşünme ve Planlama için (DeepSeek-R1)
llm_thinker = ChatOllama(model="deepseek-r1:8b", temperature=0.6, num_ctx=8192, keep_alive=0)

# 2. Kodlama için (Qwen-Coder)
llm_coder = ChatOllama(model="qwen2.5-coder:7b", temperature=0, num_ctx=8192, keep_alive=0)
repl = PythonREPL()

# --- 1. DATA ANALYST NODE ---
def analyst_node(state: AgentState):
    print("\n🕵️ ANALYST: Veriyi inceliyorum...")
    data_path = state.get("data_path", "satis_verisi.csv")
    
    # Prompt: Python tool kullanarak veriyi analiz et
    prompt = f"""
    You are a Data Analyst. The file is '{data_path}'.
    Write python code to load the data, check df.head(), df.info() and df.describe().
    
    IMPORTANT: In your summary, you MUST list:
    1. EXACT column names (copy-paste from df.columns)
    2. Data types for each column
    3. Missing values count
    4. Sample values from first few rows
    
    OUTPUT FORMAT:
    First line: "CODE_START"
    Then the code.
    Last line: "CODE_END"
    Then your detailed summary with EXACT column names.
    """
    
    response = llm_thinker.invoke([SystemMessage(content=prompt)])
    content = response.content
    
    # Basit bir parser: Kodu ayıkla ve çalıştır
    report = content
    if "CODE_START" in content and "CODE_END" in content:
        code = content.split("CODE_START")[1].split("CODE_END")[0].strip()
        print(f"   Running Code:\n{code[:50]}...") # Log
        try:
            # PythonREPL ile çalıştır
            # Not: Veriyi yüklemek için 'pd' import edilmeli
            exec_result = repl.run("import pandas as pd\n" + code)
            report = f"ANALYSIS RESULT:\n{exec_result}\n\nSUMMARY:\n{content.split('CODE_END')[1]}"
        except Exception as e:
            report = f"Error analyzing data: {e}"

    return {
        "messages": [AIMessage(content=f"Analyst Report:\n{report}")],
        "analyst_report": report
    }

# --- 2. ARCHITECT NODE ---
def architect_node(state: AgentState):
    print("\n🏗️ ARCHITECT: Mimarisi tasarlıyorum...")
    report = state.get("analyst_report", "No report")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)
    
    # Eğer eleştiri varsa prompt'u değiştir
    if critique and critique != "APPROVE":
        print(f"   ⚠️ Eleştiriler dikkate alınıyor (Revizyon {revision_count + 1})...")
        instruction = f"""
        PREVIOUS DESIGN was rejected with this CRITIQUE:
        "{critique}"
        
        PLEASE REVISE the design to address these issues.
        """
    else:
        instruction = "Create the initial design."

    prompt = f"""
    You are a ML Architect. Based on this analysis:
    {report}
    
    TIMETABLE: {instruction}
    
    Design a machine learning pipeline using Python libraries suitable for the data.
    Propose the BEST model architecture (e.g., LightGBM, XGBoost, RandomForest, etc.) based on the data characteristics.
    
    Specify: 
    1. Features to use
    2. Target Variable (Must be numeric for regression)
    3. Preprocessing steps (Imputation, Encoding, Scaling)
    4. Model Selection (Explain WHY you chose this model)
    5. Evaluation Metric
    
    Keep it technical and concise.
    """
    
    response = llm_thinker.invoke([SystemMessage(content=prompt)])
    
    # Revizyon sayısını artır
    return {
        "messages": [AIMessage(content=f"Design Doc:\n{response.content}")],
        "design_doc": response.content,
        "revision_count": revision_count + 1
    }

# --- 2.5 CRITIC NODE ---
def critic_node(state: AgentState):
    print("\n🧐 CRITIC: Tasarımı inceliyorum...")
    design = state.get("design_doc", "")
    analyst_report = state.get("analyst_report", "")
    
    prompt = f"""
    You are a Senior ML Critic. Review this architecture design:
    
    DESIGN:
    {design}
    
    DATA REPORT:
    {analyst_report}
    
    Your goal is to find CRITICAL FLAWS, especially:
    1. Data Leakage (e.g., using Target variable as a Feature).
    2. Invalid Validation Strategy (e.g., random split for time-series data).
    3. Missing Preprocessing (e.g., not encoding categorical variables).
    4. Logical Errors.
    
    If the design is GOOD and SAFE, output ONLY the word: "APPROVE".
    If there are issues, list them clearly as bullet points for the Architect to fix.
    """
    
    response = llm_thinker.invoke([SystemMessage(content=prompt)])
    critique = response.content.strip()
    
    if "APPROVE" in critique.upper() and len(critique) < 50: # Bazen APPROVE yanına açıklama yazabilir, kısa ise onay say
        print("   ✅ Tasarım ONAYLANDI.")
        critique = "APPROVE"
    else:
        print("   ❌ Tasarım REDDEDİLDİ. Eleştiriler gönderiliyor...")
        print(f"   Eleştiri Özet: {critique[:100]}...")
        
    return {
        "messages": [AIMessage(content=f"Critique: {critique}")],
        "critique": critique
    }

# --- 3. ENGINEER NODE ---
MAX_RETRIES = 10  # Maksimum hata düzeltme denemesi artırıldı

def engineer_node(state: AgentState):
    design = state.get("design_doc", "")
    analyst_report = state.get("analyst_report", "")  # Veri raporunu al
    test_error = state.get("test_error", "")
    retry_count = state.get("retry_count", 0)
    data_path = state.get("data_path", "satis_verisi.csv")
    
    # Self-Healing: Eğer önceki kodda hata varsa, düzeltme moduna gir
    if test_error:
        retry_count += 1
        print(f"\n👷 ENGINEER: Hata tespit edildi, kodu düzeltiyorum... (Deneme {retry_count}/{MAX_RETRIES})")
        prompt = f"""
        You are a Python ML Engineer.
        The previous code you wrote had this error:
        
        ERROR:
        {test_error}
        
        CURRENT CODE:
        {state.get("final_code", "")}
        
        DESIGN REQUIREMENTS:
        {design}
        
        DATA ANALYSIS REPORT (Use this to understand column types):
        {analyst_report}
        
        IMPORTANT: The data file is named '{data_path}'. Use this EXACT filename.
        
        RULES:
        - Fix the error in the code.
        - IMPORTANT: The TARGET variable (y) MUST be excluded from the features (X). 
        - DO NOT USE the target column inside the training features! This causes Data Leakage.
        - Load data from '{data_path}' (this is the EXACT filename).
        - Use any Python ML libraries recommended in the design (sklearn, lightgbm, xgboost, etc.).
        - If 'ValueError: could not convert string to float' happens, handle categorical columns (OneHotEncoder).
        - Handle missing values.
        - Save model as 'model.joblib'.
        - OUTPUT ONLY THE FIXED CODE inside ```python ... ``` blocks.
        """
    else:
        retry_count = 0  # Yeni kod yazılıyorsa sayacı sıfırla
        print("\n👷 ENGINEER: Kodu yazıyorum...")
        prompt = f"""
        You are a Python ML Engineer.
        Write a complete 'train.py' script based on this design:
        {design}
        
        DATA ANALYSIS REPORT (Use this to understand column types):
        {analyst_report}
        
        IMPORTANT: The data file is named '{data_path}'. Use this EXACT filename.
        
        RULES:
        - IMPORTANT: The TARGET variable (y) MUST be excluded from the features (X). 
        - DO NOT USE the target column inside the training features! This causes Data Leakage.
        - Load data from '{data_path}' (this is the EXACT filename).
        - Use any Python ML libraries recommended in the design (sklearn, lightgbm, xgboost, etc.).
        - Handle missing values robustly (e.g. `df.ffill()` or `SimpleImputer`).
        - Save model as 'model.joblib'.
        - OUTPUT ONLY THE CODE inside ```python ... ``` blocks.
        """
    
    response = llm_coder.invoke([SystemMessage(content=prompt)])
    
    # Kodu ayıkla
    code = response.content
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    
    # Dosyaya yaz (Script dizinine)
    file_path = os.path.join(SCRIPT_DIR, "train_auto.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    print(f"   📝 Kod '{file_path}' dosyasına yazıldı.")
        
    return {
        "messages": [AIMessage(content="Code written to 'train_auto.py'.")],
        "final_code": code,
        "test_error": "",  # Yeni kod yazıldığında önceki hatayı temizle
        "retry_count": retry_count
    }

# --- 4. TESTER NODE ---
# --- 4. TESTER NODE (Otonom Paket Yüklemeli) ---
def tester_node(state: AgentState):
    print("\n🧪 TESTER: Kodu test ediyorum...")
    
    retry_count = state.get("retry_count", 0)
    
    # Retry limit kontrolü
    if retry_count >= MAX_RETRIES:
        print(f"   ⚠️ MAKSIMUM DENEME SAYISINA ULAŞILDI ({MAX_RETRIES})!")
        print("   🛑 Döngü durduruluyor. Kod hatalı olabilir.")
        return {
            "messages": [AIMessage(content=f"Max retries ({MAX_RETRIES}) reached. Stopping self-healing loop.")],
            "test_error": ""
        }
    
    file_path = os.path.join(SCRIPT_DIR, "train_auto.py")
    
    # Dosya var mı kontrol et
    if not os.path.exists(file_path):
        error_msg = f"Dosya bulunamadı: {file_path}"
        print(f"   ❌ {error_msg}")
        return {
            "messages": [AIMessage(content=f"Test Failed: {error_msg}")],
            "test_error": error_msg
        }
    
    try:
        # 1. Kodu Çalıştır
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=SCRIPT_DIR
        )
        
        # 2. Sonucu Analiz Et
        if result.returncode == 0:
            print("   ✅ TEST BAŞARILI!")
            if result.stdout:
                print(f"   Output: {result.stdout[:200]}...")
            return {
                "messages": [AIMessage(content="Test Passed! Code runs successfully.")],
                "test_error": ""
            }
        else:
            # Hata var, çıktıyı al
            error_output = result.stderr or result.stdout or "Unknown error"
            
            # 3. Otonom Paket Yükleme Kontrolü (ModuleNotFoundError)
            if "ModuleNotFoundError" in error_output:
                # Regex ile modül ismini bul: No module named 'xyz'
                match = re.search(r"No module named '(\w+)'", error_output)
                if match:
                    missing_module = match.group(1)
                    print(f"   📦 Eksik kütüphane tespit edildi: '{missing_module}'")
                    print(f"   ⬇️ Yükleniyor...")
                    
                    # Pip install çalıştır
                    install_result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", missing_module],
                        capture_output=True,
                        text=True
                    )
                    
                    if install_result.returncode == 0:
                        print(f"   ✅ '{missing_module}' başarıyla yüklendi. Test yeniden başlatılıyor (RETRY)...")
                        
                        # HEMEN RETRY YAP (İkinci şans)
                        retry_result = subprocess.run(
                            [sys.executable, file_path],
                            capture_output=True,
                            text=True,
                            timeout=120,
                            cwd=SCRIPT_DIR
                        )
                        
                        if retry_result.returncode == 0:
                            print("   ✅ RETRY TEST BAŞARILI!")
                            return {
                                "messages": [AIMessage(content=f"Test Passed after installing '{missing_module}'.")],
                                "test_error": ""
                            }
                        else:
                            # Yine hata verdiyse hatayı güncelle
                            error_output = retry_result.stderr or retry_result.stdout
                            print("   ❌ RETRY TEST BAŞARISIZ!")
                    else:
                        print(f"   ❌ Paket yükleme başarısız oldu: {install_result.stderr}")

            # Eğer buraya geldiyse ya modül hatası değildir ya da yükleme işe yaramamıştır
            print(f"   ❌ TEST BAŞARISIZ!")
            print(f"   Hata: {error_output[:500]}...")
            return {
                "messages": [AIMessage(content=f"Test Failed: {error_output[:2000]}")],
                "test_error": error_output[:2000]
            }
            
    except subprocess.TimeoutExpired:
        error_msg = "Kod çalıştırma zaman aşımına uğradı (120s)"
        print(f"   ⏰ {error_msg}")
        return {
            "messages": [AIMessage(content=f"Test Failed: {error_msg}")],
            "test_error": error_msg
        }
    except Exception as e:
        error_msg = f"Test sırasında beklenmeyen hata: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {
            "messages": [AIMessage(content=f"Test Failed: {error_msg}")],
            "test_error": error_msg
        }