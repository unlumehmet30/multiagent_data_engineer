from agent_tools import python_analysis_tool

print("🧪 Tool Testi Başlıyor...")

# Test 1: Basit Matematik
code1 = """
a = 5
b = 10
print(f"Toplam: {a + b}")
"""
print("1. Test (Matematik):", python_analysis_tool.invoke(code1))

# Test 2: Pandas DataFrame (Hafıza Testi)
# Önce df oluşturalım
code2 = """
data = {'Urun': ['Elma', 'Armut'], 'Fiyat': [100, 200]}
df = pd.DataFrame(data)
print("DataFrame oluşturuldu.")
"""
print("2. Test (Veri Yükleme):", python_analysis_tool.invoke(code2))

# Sonra df'i kullanalım (State korunuyor mu?)
code3 = """
print(df.describe())
"""
print("3. Test (Hafıza Erişimi):", python_analysis_tool.invoke(code3))