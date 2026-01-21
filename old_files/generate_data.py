import pandas as pd
import numpy as np

# Rastgele ama mantıklı bir satış verisi oluşturalım
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
products = ["Laptop", "Mouse", "Keyboard", "Monitor"]

data = {
    "Tarih": dates,
    "Urun": np.random.choice(products, 100),
    "Satis_Adedi": np.random.randint(1, 20, 100),
    "Birim_Fiyat": np.random.uniform(100, 2000, 100), # Biraz gürültülü veri
    "Bolge": np.random.choice(["Istanbul", "Ankara", "Izmir"], 100)
}

df = pd.DataFrame(data)

# Biraz "pislik" (Data Quality Issues) ekleyelim ki ajan yakalasın
df.loc[5, "Satis_Adedi"] = None  # Eksik veri
df.loc[10, "Birim_Fiyat"] = -500 # Hatalı veri (Negatif fiyat)

df.to_csv("satis_verisi.csv", index=False)
print("✅ 'satis_verisi.csv' oluşturuldu.")