import pandas as pd
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

# Python REPL (Read-Eval-Print Loop) motorunu başlatıyoruz
python_repl = PythonREPL()

# Global değişkenleri (context) saklamak için bir sözlük
# Bu sayede ajan:
# Adım 1: df = pd.read_csv(...)
# Adım 2: print(df.head())
# diyebilir ve df kaybolmaz.
repl_globals = {"pd": pd} 

def run_python_code(code: str) -> str:
    """
    Ajanın ürettiği Python kodunu çalıştırır ve çıktısını (print) döner.
    Hata alırsan hatayı döner, böylece ajan kodunu düzeltebilir.
    """
    try:
        # Kodu çalıştır, globals sözlüğünü kullan
        result = python_repl.run(code) # Not: Basit REPL globals'i tam desteklemeyebilir, 
                                       # bu yüzden aşağıda exec() tabanlı daha sağlam bir yapı kuracağız.
        return result
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# --- DAHA SAĞLAM BİR YAKLAŞIM ---
# LangChain'in standart REPL'i bazen state tutmakta zorlanır.
# Kendi 'Stateful' çalıştırıcımızı yazalım.

class StatefulPythonInterpreter:
    def __init__(self):
        self.globals = {"pd": pd} # Pandas otomatik yüklü gelsin
    
    def run(self, code: str) -> str:
        import io
        import sys
        
        # Standart çıktıyı (stdout) yakalamak için
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        
        try:
            # Kodu global context içinde çalıştır
            exec(code, self.globals)
            output = redirected_output.getvalue()
            return output if output else "(Kod çalıştı ancak print() ile bir çıktı üretmedi.)"
        except Exception as e:
            return f"PYTHON HATASI:\n{e}"
        finally:
            sys.stdout = old_stdout

# Tool'u başlatalım
interpreter = StatefulPythonInterpreter()

def python_tool_func(code: str):
    print(f"\n🐍 [AJAN KOD YAZIYOR]...\n{'-'*30}\n{code}\n{'-'*30}")
    return interpreter.run(code)

# LangChain Tool objesi olarak paketleyelim
python_analysis_tool = Tool(
    name="python_interpreter",
    func=python_tool_func,
    description="Python kodu çalıştırır. Veri analizi, grafik çizimi ve hesaplamalar için bunu kullan. Sadece kodu gönder."
)