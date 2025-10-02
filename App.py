import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import random

# =========================
# Función de predicción
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

model = load_model()

def predict_digit(image):
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')/255.0
    img = img.reshape((1,28,28,1))
    pred = model.predict(img, verbose=0)
    result = np.argmax(pred[0])
    confidence = np.max(pred[0])
    return result, confidence

# =========================
# Configuración de la app
# =========================
st.set_page_config(page_title='Juego de Reconocimiento de Dígitos', layout='wide')

st.title("🎮 Juego de Reconocimiento de Dígitos")
st.write("Dibuja el número que se te pide en cada nivel. ¡Completa los 3 niveles para ganar!")

# =========================
# Variables de sesión
# =========================
if "nivel" not in st.session_state:
    st.session_state.nivel = 1
if "puntaje" not in st.session_state:
    st.session_state.puntaje = 0
if "objetivo" not in st.session_state:
    st.session_state.objetivo = random.randint(0,9)

# =========================
# Mostrar información de nivel
# =========================
st.subheader(f"Nivel {st.session_state.nivel} de 3")
st.info(f"👉 Dibuja el número: **{st.session_state.objetivo}**")

# =========================
# Canvas del nivel
# =========================
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15, key="slider")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)", 
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=200,
    width=200,
    key=f"canvas_{st.session_state.nivel}"
)

# =========================
# Botón de detección
# =========================
if st.button("✅ Verificar Dibujo"):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        res, conf = predict_digit(input_image)

        st.write(f"🔢 El modelo detectó: **{res}** (confianza {conf:.2f})")

        if res == st.session_state.objetivo:
            st.success("🎉 ¡Correcto!")
            st.session_state.puntaje += 1
        else:
            st.error("❌ Incorrecto")

        # Pasar al siguiente nivel
        if st.session_state.nivel < 3:
            st.session_state.nivel += 1
            st.session_state.objetivo = random.randint(0,9)
        else:
            st.balloons()
            st.success(f"Juego terminado 🎮 Puntaje final: {st.session_state.puntaje}/3")
    else:
        st.warning("Por favor dibuja algo en el lienzo.")

# =========================
# Sidebar con puntaje
# =========================
st.sidebar.title("📊 Progreso")
st.sidebar.write(f"Nivel actual: {st.session_state.nivel}/3")
st.sidebar.write(f"Puntaje: {st.session_state.puntaje}")

if st.sidebar.button("🔄 Reiniciar Juego"):
    st.session_state.nivel = 1
    st.session_state.puntaje = 0
    st.session_state.objetivo = random.randint(0,9)
    st.experimental_rerun()

