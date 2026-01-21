from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage

# Ajanlar arasındaki ortak hafıza yapısı
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str
    data_path: str
    analyst_report: str
    design_doc: str
    final_code: str
    test_error: str
    retry_count: int  # Kaç kez hata düzeltme denemesi yapıldı
    # Adversarial Review Loop için
    critique: str
    revision_count: int  # Tasarımın kaç kez revize edildiği