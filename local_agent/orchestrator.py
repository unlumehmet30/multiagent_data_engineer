from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from state import AgentState
from agents import analyst_node, architect_node, engineer_node, tester_node, critic_node

# --- SUPERVISOR (YÖNETİCİ) ---
# Görevi: State'e bakarak sıradaki işçiyi seçmek.

def supervisor_node(state: AgentState):
    messages = state['messages']
    analyst_report = state.get('analyst_report', '')
    design_doc = state.get('design_doc', '')
    final_code = state.get('final_code', '')
    test_error = state.get('test_error', '')
    
    # State'e bakarak kararı deterministik olarak belirle
    if not analyst_report:
        decision = "Analyst"
    elif not design_doc:
        decision = "Architect"
    elif not final_code:
        decision = "Engineer"
    elif test_error:
        # Test hatası varsa ama kod var -> Engineer tekrar düzeltmeli
        # NOT: Bu durum Tester'dan döndüğünde olur
        decision = "Engineer"
    else:
        # Her şey tamam ve test geçti
        decision = "FINISH"
    
    print(f"\n👑 SUPERVISOR KARARI: {decision}")
    return {"next_step": decision}

# --- TESTER ROUTING FUNCTION ---
def route_after_test(state: AgentState) -> str:
    """Tester'dan sonra nereye gidileceğine karar ver."""
    test_error = state.get('test_error', '')
    if test_error:
        # Hata var -> Engineer'a geri dön (Self-Healing Loop)
        print("   🔄 Hata bulundu, Engineer'a geri dönülüyor...")
        return "Engineer"
    else:
        # Hata yok -> Supervisor'a git
        print("   ✅ Test geçti, Supervisor'a gidiliyor...")
        print("   ✅ Test geçti, Supervisor'a gidiliyor...")
        return "Supervisor"

# --- CRITIC ROUTING FUNCTION ---
def route_after_critic(state: AgentState) -> str:
    """Critic'ten sonra nereye gidileceğine karar ver."""
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)
    
    if critique == "APPROVE" or revision_count >= 3:
        if revision_count >= 3:
            print("   ⚠️ Maksimum revizyon sayısına ulaşıldı, zorla ilerleniyor.")
        else:
            print("   ✅ Tasarım Onaylandı.")
        return "Engineer"
    else:
        print("   🔄 Tasarım reddedildi, Architect'e geri dönülüyor...")
        return "Architect"


# --- GRAPH KURULUMU ---
workflow = StateGraph(AgentState)

# 1. Düğümleri Ekle
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Architect", architect_node)
workflow.add_node("Engineer", engineer_node)  # Sadece bir tane olmalı
workflow.add_node("Tester", tester_node)
workflow.add_node("Critic", critic_node)  # YENİ: Critic düğümü

# 2. Başlangıç Noktası
workflow.set_entry_point("Supervisor")

# 3. Yönlendirme (Conditional Edges)
# Supervisor'ın kararına göre nereye gideceğimizi haritalıyoruz.
workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next_step"],
    {
        "Analyst": "Analyst",
        "Architect": "Architect",
        "Engineer": "Engineer",
        "Critic": "Critic",
        "FINISH": END
    }
)

# 4. İşçilerden Sonraki Akış
workflow.add_edge("Analyst", "Supervisor")
workflow.add_edge("Analyst", "Supervisor")
# Architect artık Critic'e gidiyor, Supervisor'a değil
workflow.add_edge("Architect", "Critic")

# Critic -> Conditional Edge (Adversarial Loop)
workflow.add_conditional_edges(
    "Critic",
    route_after_critic,
    {
        "Engineer": "Engineer",
        "Architect": "Architect"
    }
)

# YENİ: Engineer -> Tester (kod yazıldıktan sonra test et)
workflow.add_edge("Engineer", "Tester")

# YENİ: Tester -> Conditional Edge (Self-Healing Loop)
workflow.add_conditional_edges(
    "Tester",
    route_after_test,
    {
        "Engineer": "Engineer",  # Hata varsa Engineer'a dön
        "Supervisor": "Supervisor"  # Hata yoksa Supervisor'a git
    }
)

# 5. Derle
app = workflow.compile()

# --- ÇALIŞTIR ---
if __name__ == "__main__":
    print("🚀 OTONOM ML SİSTEMİ BAŞLATILIYOR (Self-Healing Mode)...")
    print("=" * 60)
    
    # Başlangıç durumu
    initial_state = {
        "messages": [HumanMessage(content="Lütfen 'satis_verisi.csv' dosyasını kullanarak bir satış tahmin modeli kur.")],
        "data_path": "satis_verisi.csv",
        "analyst_report": "",
        "design_doc": "",
        "final_code": "",
        "next_step": "",
        "test_error": "",
        "test_error": "",
        "retry_count": 0,
        "critique": "",
        "revision_count": 0,
    }
    
    # Akışı başlat (Recursion limit artırıldı)
    for output in app.stream(initial_state, {"recursion_limit": 50}):
        pass  # Print işlemleri node'ların içinde yapılıyor zaten
    
    print("\n" + "=" * 60)
    print("🏁 İŞLEM TAMAMLANDI!")
    print("   📄 'train_auto.py' dosyasını kontrol et.")
    print("   🤖 'model.joblib' modeli oluşturulmuş olmalı.")
