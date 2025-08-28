
# 🎯 Student Exam Performance Prediction

Bu proje, öğrenciye ait sosyo-demografik özellikler ve önceki notlarına dayanarak **G3 (final notu)** değerini tahmin eden uçtan uca bir makine öğrenmesi uygulamasıdır.

## 🔍 Proje Amacı

Öğrenci başarı verileri kullanılarak regresyon modelleriyle son dönem (G3) notunun tahmin edilmesi hedeflenmiştir. Modelleme süreci kapsamında veri ön işleme, özellik mühendisliği, modelleme, değerlendirme ve Flask ile web arayüzü geliştirilmiştir.

## 📊 Kullanılan Veriler

Veri seti öğrencilere ait şu bilgileri içermektedir:

- Demografik Bilgiler: cinsiyet, yaş, adres, aile durumu vb.
- Akademik Bilgiler: önceki notlar (G1, G2), ders dışı aktiviteler, öğrenim süresi vb.
- Hedef Değişken: **G3 (final notu)**

## ⚙️ Kullanılan Modeller

- Linear Regression
- Ridge, Lasso
- Decision Tree, Random Forest, XGBoost, CatBoost, AdaBoost
- KNN Regressor

## ✅ En Başarılı Model

```
XGBoostRegressor: Train R2: 0.9791, Test R2: 0.8007
```

## 🌐 Web Uygulaması (Flask)

Form üzerinden öğrenci bilgileri girildiğinde **matematik notu (G3)** tahmin edilmektedir.

### Kullanılan Teknolojiler

- Python
- Flask
- Scikit-learn, XGBoost, CatBoost
- HTML/CSS (Bootstrap)
- Logging, Exception Handling
- Pipeline ve Model Persistency (joblib/pickle)

## 📷 Uygulama Görseli

![GIF Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGgybGhhbHFmbXZ2dzRreXAzMWxjOXE2aGRoM3poZTZzOHJ2dDVqdiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Yl5aO3gdVfsQ0/giphy.gif)

## 🧠 Nasıl Çalışır?

1. Veriyi kullanıcıdan HTML formu ile alır.
2. Preprocessing pipeline ile veriyi işler.
3. Eğitilmiş modeli kullanarak tahmin yapar.
4. Tahmini kullanıcıya sunar.

## 🚀 Kurulum

```bash
git clone https://github.com/Kubilayalici/end_to_end_mlproject.git
cd end_to_end_mlproject
pip install -r requirements.txt
python app.py
```

## Calistirma (Run)

- Gelistirme:  `python app.py` 
- Uretim (Gunicorn):  `gunicorn -b 0.0.0.0:8000 app:app` 
  - Alternatif WSGI hedefi:  `gunicorn -b 0.0.0.0:8000 app:application` 

## 👨‍💻 Geliştirici

**Kubilay ALICI**  
[GitHub](https://github.com/Kubilayalici) | [LinkedIn](https://www.linkedin.com/in/kubilay-alici-8822a21b9/)

---

📌 Bu proje, veri bilimi yeteneklerinizi sergilemek ve model geliştirme süreçlerini uçtan uca anlamak için güçlü bir örnektir.
