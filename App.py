import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# =========================
#  FUNCIÓN: Predecir número
# =========================
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# ==============================
#  FUNCIÓN: Detectar forma básica
# ==============================
def detectar_forma(img):
    img_gray = ImageOps.grayscale(img).resize((200,200))
    arr = np.array(img_gray)
    _, thresh = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return "Nada detectado"

    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.04*cv2.arcLength(c, True), True)

    # Clasificación
    lados = len(approx)
    if lados == 3:
        return "🔺 Triángulo"
    elif lados == 4:
        return "◼️ Cuadrado"
    elif lados > 5:
        return "⭕ Círculo"
    else:
        return "Figura desconocida"

# =========================
# CONFIGURACIÓN DE LA APP
# =========================
st.set_page_config(page_title='🖌️ Detección de Números y Figuras', layout='wide')

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🖌️ Detección de Números y Figuras Geométricas</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Dibuja un número (0–9) o una figura geométrica (círculo, cuadrado, triángulo)</h4>", unsafe_allow_html=True)

# =========================
# CANVAS DE DIBUJO
# =========================
st.sidebar.title("🎨 Opciones de Dibujo")
stroke_width = st.sidebar.slider('✏️ Ancho de línea', 1, 30, 15)
stroke_color = st.sidebar.color_picker("🎨 Color del lápiz", "#FFFFFF")
bg_color = st.sidebar.color_picker("🌌 Color de fondo", "#000000")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=250,
    width=250,
    key="canvas",
)

# =========================
# BOTONES DE ACCIÓN
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button('🔢 Predecir Número'):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            img = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
            res = predictDigit(img)
            st.success('El número detectado es: ' + str(res))
        else:
            st.warning('Por favor dibuja un número en el canvas.')

with col2:
    if st.button("📐 Detectar Forma"):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            img = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
            resultado = detectar_forma(img)
            st.success(f"La figura detectada es: {resultado}")
        else:
            st.warning("Por favor dibuja una figura en el canvas.")

# =========================
# SIDEBAR - INFO EXTRA
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Acerca de la app")
st.sidebar.info(
    "Esta aplicación permite reconocer **números escritos a mano (0–9)** "
    "usando un modelo de IA entrenado con MNIST, "
    "y detectar **figuras geométricas básicas** con visión por computadora."
)



