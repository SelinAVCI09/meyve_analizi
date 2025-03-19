import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Modeli yükle
model = tf.keras.models.load_model('meyve_model.h5')

# Sınıf isimlerini yükle
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Streamlit arayüz
st.title("Meyve ve Sebze Tazelik Analizi")
uploaded_file = st.file_uploader("Resim yükle", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    # RGBA yerine RGB formatına dönüştürme
    image = image.convert("RGB")
    st.image(image, caption='Yüklenen Resim', use_column_width=True)

    # Görüntüyü modele uygun boyutta yeniden boyutlandırma
    img = image.resize((150, 150))
    # Görüntüyü modelin beklediği formata getirme (batch boyutu ekleme ve normalizasyon)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Tahmin yapma
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Sonuçları gösterme
    st.write(f"Tahmin: **{predicted_class}**")
    st.write(f"Güven Oranı: **{confidence:.2f}%**")
