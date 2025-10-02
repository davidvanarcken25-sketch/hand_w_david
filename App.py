import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Función para predecir números
# -----------------------------
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

# -----------------------------
# Función para detectar formas
# -----------------------------
def detectar_forma_simple(img):
    img_gray = ImageOps.grayscale(img).resize((200,200))
    arr = np.array(img_gray)

    # Binarizar (0 o 1)
    arr = (arr < 128).astype(np.uint8)

    # Contar pixeles negros
    total_black = np.sum(arr)

    if total_black < 100:  # nada dibujado
        return "❌ No se detecta forma"

    # Heurísticas simples
    rows_black = np.sum(arr, axis=1)
    cols_black = np.sum(arr, axis=0)

    if abs(np.argmax(rows_black) - 100) < 20 and abs(np.argmax(cols_black) - 100) < 20:
        return "⭕ Círculo"
    elif np.max(rows_black) > 150 and np.max(cols_black) > 150:
        return "◼️ Cuadrado"
    elif np.argmax(rows_black) < 80:
        return "🔺 Triángulo"
    else:
        return "📐 Figura desconocida"

# -----------------------------
# Interfaz en Streamlit
# -----------------------------
st.set_page_config(page_title='Reconocimiento de Números y Figuras', layout='wide')

st.markdown("<h1 style='text-align: center; color: cyan;'>✍️ Reconocimiento de Números y Figuras Geométricas</h1>", unsafe_allow_html=True)
st.write("Dibuja un número o una forma geométrica en el panel y presiona **Detectar**")

# Panel de dibujo
stroke_width = st.slider('✏️ Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'  # blanco
bg_color = '#000000'      # negro

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=250,
    width=250,
    key="canvas",
)

# Botón de predicción
if st.button('🔍 Detectar'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image = input_image.convert("RGB")

        # Detectar número
        try:
            num = predictDigit(input_image)
            st.success(f"🔢 Número detectado: **{num}**")
        except:
            st.warning("⚠️ No se pudo detectar un número (quizás dibujaste una figura).")

        # Detectar figura
        forma = detectar_forma_simple(input_image)
        st.info(f"🟦 Forma detectada: **{forma}**")

    else:
        st.error("Por favor dibuja algo en el canvas.")

# Sidebar
st.sidebar.title("ℹ️ Acerca de:")
st.sidebar.markdown("""
Esta aplicación permite reconocer:

- 🔢 Dígitos escritos a mano (0-9)  
- ⭕ Formas geométricas básicas (círculo, cuadrado, triángulo)  

Desarrollado con **Streamlit + TensorFlow + Numpy** ✨
""")




