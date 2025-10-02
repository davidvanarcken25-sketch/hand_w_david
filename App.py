import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# --- Función de predicción ---
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    confidence = round(100 * np.max(pred[0]), 2)
    return result, confidence

# --- Configuración de la página ---
st.set_page_config(page_title="🖊️ Reconocimiento de Dígitos", page_icon="✍️", layout="centered")

# --- Título principal ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🖊️ Reconocimiento de Dígitos escritos a mano</h1>", unsafe_allow_html=True)
st.write("Dibuja un número en el panel y presiona **Predecir** para ver el resultado.")

# --- Canvas para dibujar ---
st.subheader("✏️ Panel de dibujo")
stroke_width = st.slider('Ancho de línea', 5, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# --- Botón de predicción ---
if st.button('🔍 Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        res, conf = predictDigit(input_image)

        st.success(f"✅ El dígito es: **{res}**")
        st.progress(int(conf))
        st.info(f"Confianza del modelo: **{conf}%**")
    else:
        st.warning('⚠️ Por favor dibuja un número antes de predecir.')

# --- Sidebar ---
st.sidebar.title("ℹ️ Acerca de")
st.sidebar.write("Esta aplicación evalúa la capacidad de una **Red Neuronal Artificial (RNA)** para reconocer dígitos escritos a mano.")
st.sidebar.write("📌 Basado en el desarrollo de *Vinay Uniyal*")
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Desarrollado por: **Tu Nombre**")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>🖥️ Proyecto académico - 2025</p>", unsafe_allow_html=True)
