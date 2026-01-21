import pandas as pd
import numpy as np

# Dosyayı oku
print("Dosya okunuyor...")
try:
    df = pd.read_csv('satis_verisi.csv')
except Exception as e:
    print(f"Hata: {e}")
    exit(1)

print(f"Orijinal Veri Boyutu: {df.shape}")

# Yeni DataFrame oluştur
new_df = pd.DataFrame()

# Kolon eşleştirmeleri
new_df['Tarih'] = df['Tarih_Converted']

# Ürün Adı boşsa Stok Açıklamasını kullan
new_df['Urun'] = df['Ürün Adı'].fillna(df['Stok-Hizmet Açıklaması'])

new_df['Satis_Adedi'] = df['Miktar']
new_df['Birim_Fiyat'] = df['Birim Fiyat']

# Bolge yerine Satış Elemanı kullan (Kategorik değişken olarak)
new_df['Bolge'] = df['Satış Elemanı'].fillna('Bilinmiyor')

# Veri temizliği
# Sayısal değerlerdeki olası hataları temizle
new_df['Satis_Adedi'] = pd.to_numeric(new_df['Satis_Adedi'], errors='coerce').fillna(0)
new_df['Birim_Fiyat'] = pd.to_numeric(new_df['Birim_Fiyat'], errors='coerce').fillna(0)

# Tarih formatını kontrol et
new_df['Tarih'] = pd.to_datetime(new_df['Tarih'], errors='coerce')
new_df = new_df.dropna(subset=['Tarih'])  # Tarihi olmayanları at

# Kaydet
print("Dönüştürülen veri kaydediliyor...")
new_df.to_csv('satis_verisi.csv', index=False)
print(f"Yeni Veri Boyutu: {new_df.shape}")
print("İşlem Başarılı! Dosya 'Tarih,Urun,Satis_Adedi,Birim_Fiyat,Bolge' formatına getirildi.")
