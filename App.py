import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ------------------------------
# Función para predecir dígitos
# ------------------------------
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

# ------------------------------
# Función para detectar expresiones con reglas simples
# ------------------------------
def detectar_expresion(img):
    img_gray = ImageOps.grayscale(img).resize((50,50))
    arr = np.array(img_gray)

    # Dividir en parte superior e inferior
    arriba = arr[:25, :]
    abajo = arr[25:, :]

    intensidad_arriba = np.sum(arriba < 128)  # pixeles oscuros
    intensidad_abajo = np.sum(abajo < 128)

    # Reglas básicas
    if intensidad_abajo > intensidad_arriba * 1.2:
        return "😊 Feliz"
    elif intensidad_arriba > intensidad_abajo * 1.2:
        return "😢 Triste"
    else:
        return "😐 Serio"

# ------------------------------
# Configuración de la app
# ------------------------------
st.set_page_config(page_title='Reconocimiento de Dígitos y Expresiones', layout='wide')
st.title('🖌️ Reconocimiento de Dígitos escritos a mano y Expresiones')
st.subheader("Dibuja un dígito o una carita (feliz, triste, seria) en el panel y presiona un botón")

# Parámetros del canvas
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# ------------------------------
# Botón para predecir dígito
# ------------------------------
if st.button('Predecir Dígito'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        img = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        res = predictDigit(img)
        st.header('El dígito es: ' + str(res))
    else:
        st.warning('Por favor dibuja en el canvas el dígito.')

# ------------------------------
# Botón para detectar expresión
# ------------------------------
if st.button('Detectar Expresión'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        img = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        expresion = detectar_expresion(img)
        st.header(f"La expresión parece: {expresion}")
    else:
        st.warning('Por favor dibuja una carita en el canvas.')

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("Acerca de:")
st.sidebar.text("Esta app reconoce:")
st.sidebar.text(" - Dígitos escritos a mano (0-9)")
st.sidebar.text(" - Caritas simples (feliz, triste, seria)")
st.sidebar.text("Demostración con reglas + IA básica.")

