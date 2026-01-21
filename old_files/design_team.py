from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# --- AYARLAR ---
MODEL_NAME = "qwen2.5-coder:7b"
llm = ChatOllama(model=MODEL_NAME, temperature=0.2, num_ctx=8192)

def read_analyst_report():
    try:
        with open("analyst_report.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Analiz raporu bulunamadı. Lütfen önce analyst.py çalıştırın."

def run_architect(report_content):
    print("\n🏗️ ARCHITECT (MİMAR) ÇALIŞIYOR...\n" + "="*40)
    
    system_prompt = """
    You are a Senior ML Architect.
    Your goal is to design a Production-Ready Machine Learning pipeline based on the Analyst's report.
    
    INPUT: Data Analysis Report.
    OUTPUT: A technical design document (Markdown).
    
    REQUIREMENTS:
    1. Select the best model (LightGBM, XGBoost, RandomForest, etc.) and explain WHY.
    2. Define Feature Engineering strategy (Encoding, Scaling, Lag features if time-series).
    3. Define Validation Strategy (K-Fold, TimeSeriesSplit).
    4. Metrics (RMSE, Accuracy, F1-Score).
    
    Write the response in TURKISH.
    """
    
    user_msg = f"Here is the Analyst Report:\n---\n{report_content}\n---\nCreate the ML Design Doc."
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ])
    
    print(f"\n📄 TASARIM TASLAĞI:\n{response.content}")
    return response.content

def run_critic(design_doc):
    print("\n🧐 CRITIC (ELEŞTİRMEN) ÇALIŞIYOR...\n" + "="*40)
    
    system_prompt = """
    You are a Harsh ML Critic.
    Your goal is to review the ML Design Document.
    
    LOOK FOR RISKS:
    1. Data Leakage (e.g., using future data in time series).
    2. Overfitting risks.
    3. Is the metric appropriate for the problem?
    
    OUTPUT:
    - If good: Write "APPROVED" at the beginning, then summarize why.
    - If bad: Write "REJECTED" and explain what to fix.
    
    Write the response in TURKISH.
    """
    
    user_msg = f"Here is the proposed design:\n---\n{design_doc}\n---\nReview it."
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ])
    
    print(f"\n📝 ELEŞTİRİ RAPORU:\n{response.content}")
    return response.content

if __name__ == "__main__":
    # 1. Raporu Oku
    report = read_analyst_report()
    
    if "Analiz raporu bulunamadı" in report:
        print(report)
    else:
        # 2. Mimar Tasarlasın
        design = run_architect(report)
        
        # 3. Eleştirmen Yorumlasın
        critique = run_critic(design)
        
        # 4. Sonucu Kaydet
        with open("final_design_doc.md", "w", encoding="utf-8") as f:
            f.write(f"# ML SYSTEM DESIGN\n\n## PROPOSAL\n{design}\n\n## CRITIQUE\n{critique}")
        print("\n✅ Tasarım ve Eleştiri 'final_design_doc.md' dosyasına kaydedildi.")