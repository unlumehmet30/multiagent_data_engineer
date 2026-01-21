import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent_tools import python_analysis_tool 

# --- AYARLAR ---
MODEL_NAME = "qwen2.5-coder:7b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0, 
    num_ctx=8192, 
)

def extract_python_code(text):
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    return None

def run_robust_analyst():
    # --- KRİTİK DEĞİŞİKLİK: Prompt Stratejisi ---
    # Ajana "Adım Adım" gitmesini emrediyoruz.
    # İlk adımda hesaplama yapmak YASAK. Sadece kolonları görmeli.
    
    system_prompt = """
    You are a cautious Data Analyst. You must analyze 'satis_verisi.csv'.
    
    CRITICAL RULE: NEVER assume column names (like 'Product', 'Sales').
    The data might be in Turkish (e.g., 'Urun', 'Satis_Adedi').
    
    STRATEGY:
    1. FIRST STEP: Write code ONLY to load the dataframe and print `df.columns` and `df.head()`.
    2. STOP there. Do not try to calculate anything in the first step.
    3. Wait for the output to see the REAL column names.
    4. SECOND STEP: Once you know the column names, write new code to find missing values and the best selling item using the CORRECT names.
    5. Final Answer: Summarize findings in TURKISH.
    
    FORMAT:
    Write python code inside ```python ... ``` blocks.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Veri setini analiz et. Önce sütun isimlerine bak, sonra en çok satan ürünü bul.")
    ]
    
    print(f"🤖 ANALİST BAŞLATILDI (AKILLI MOD)...\n{'='*40}")
    
    # Döngü limitini 6 yaptık
    for step in range(6):
        print(f"\n🔄 Adım {step+1} çalışıyor...")
        
        response = llm.invoke(messages)
        content = response.content
        print(f"\n🧠 AJAN:\n{content}")
        
        messages.append(AIMessage(content=content))
        
        code = extract_python_code(content)
        
        if code:
            print(f"\n⚡ KOD ALGILANDI VE ÇALIŞTIRILIYOR...")
            execution_result = python_analysis_tool.invoke(code)
            print(f"\n📊 ÇIKTI:\n{execution_result}")
            
            # Sonucu LLM'e geri besle
            result_message = f"CODE OUTPUT:\n{execution_result}\n\n(Now use the actual column names you see above to proceed.)"
            messages.append(HumanMessage(content=result_message))
        else:
            print("\n✅ KOD BLOĞU YOK - İŞLEM TAMAMLANDI.")
            
            final_report = content # LLM'in son cevabı rapordur
            with open("analyst_report.txt", "w", encoding="utf-8") as f:
                f.write(final_report)
            print("💾 Rapor 'analyst_report.txt' dosyasına kaydedildi.")
            break

if __name__ == "__main__":
    run_robust_analyst()