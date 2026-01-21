# ML SYSTEM DESIGN

## PROPOSAL
# Makine Öğrenimi Süreçü Dökümü

## 1. Model Seçimi ve Nedeni
En iyi model seçimi için birkaç faktöri göz önünde bulundurmak gerekir:
- **Veri Tipi**: Verimizde tarih, ürün, satış adedi ve bölge gibi kategorik ve sürekli değişkenler bulunmaktadır.
- **Model Performansı**: RMSE (Root Mean Squared Error) ve F1-Score metrikleri kullanılacaktır.

**Seçilen Model: LightGBM**
- **Nedeni**: LightGBM, hızlı eğitim süresine sahip olup, yüksek verimliliği sunar. Özellikle büyük veri setlerinde performansı oldukça iyi olur ve memory usage'ını optimize eder. Ayrıca, LightGBM, gradient boosting algoritması tabanlıdır ve hızlı bir şekilde model eğitimi sağlar.

## 2. Özellik Mühendisliği
Özellik mühendisliği stratejisi şu şekildedir:
- **Kategorik Değişkenler**: One-Hot Encoding kullanarak kategorik değişkenleri dönüştürülür.
  - Örneğin, "Bölge" değişkeni için farklı bölgeler için bir dizi 0 ve 1 değerine dönüştürülecektir.
- **Sayısal Değişkenler**: Standard Scaler kullanarak sürekli değişkenler (Satış Adedi ve Birim Fiyat) ölçeklendirilir.

## 3. Doğrulama Stratejisi
Doğrulama stratejisi şu şekildedir:
- **K-Fold Cross Validation**: Modelin genelleştirme performansını değerlendirmek için K-Fold Cross Validation kullanılır. Bu yöntem, veri setini belirlenen k sayısı kadar parçalara böler ve her bir parçası test kümesi olarak kullanılacak şekilde diğer parçaların eğitim kümesi olarak kullanılacaktır.

## 4. Metrikler
Metrikler şu şekildedir:
- **RMSE (Root Mean Squared Error)**: Modelin tahminlerinin gerçek değerlerden ne kadar uzak olduğunu ölçer.
- **F1-Score**: Özellikle sınıflandırma problemleri için kullanılan metrik, pozitif sınıfı doğru olarak tanımlayan ve aynı zamanda falso pozitif olmayan örneklerin doğru olarak tanımlandığı oranını gösterir.

## 5. Özet
Bu teknik döküm, veri setimiz üzerinde bir makine öğrenimi modeli oluşturmak için gerekli olan tüm adımları içeriyor. LightGBM modeli, özellik mühendisliği stratejisi ve K-Fold Cross Validation kullanarak modelin performansını optimize edecektir. RMSE ve F1-Score metrikleri, modelin genelleştirme performansını değerlendirmek için kullanılacaktır.

---

Bu döküm, veri setimiz üzerinde bir makine öğrenimi modeli oluşturmak için gerekli olan tüm adımları içeriyor.

## CRITIQUE
**REJECTED**

1. **Data Leakage**: RMSE ve F1-Score metriklerinin kullanılması, sınıflandırma problemleri için uygun değil. Bu metrikler, sınıflandırma problemi için kullanılırken, regresyon problemleri için uygun değildir. RMSE (Root Mean Squared Error), modelin tahminlerinin gerçek değerlerden ne kadar uzak olduğunu ölçer ve bu, regression problemleri için uygun bir metriktir.

2. **Overfitting Risks**: K-Fold Cross Validation kullanılmış, bu genellikle overfitting riskini azaltır. Ancak, modelin performansını optimize etmek için diğer teknikler de kullanılabilir, örneğin feature selection veya regularization yöntemleri.

3. **Metric Appropriateness**: RMSE metrikinin kullanılması uygun ancak F1-Score metrikinin kullanılması uygun değil. Bu nedenle, F1-Score'yi kaldırıp sadece RMSE kullanmak daha uygun olacaktır.

**Öneriler:**
- F1-Score'yi kaldırın ve sadece RMSE kullanın.
- Modelin performansını optimize etmek için diğer teknikler de kullanılabilir. Örneğin, feature selection veya regularization yöntemleri.

Bu düzeltmeler yapıldığında, modelin daha uygun bir şekilde değerlendirilmesi sağlanacaktır.