import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import tensorflow as tf
import math

# ================================
# Cargar modelo entrenado MNIST
# ================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("handwritten.h5")

model = load_model()

# ================================
# Funciones auxiliares
# ================================
def point_line_distance(pt, a, b):
    (x0, y0), (x1, y1), (x2, y2) = pt, a, b
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x0 - x1, y0 - y1)
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)
    if t < 0:
        return math.hypot(x0 - x1, y0 - y1)
    elif t > 1:
        return math.hypot(x0 - x2, y0 - y2)
    px = x1 + t * dx
    py = y1 + t * dy
    return math.hypot(x0 - px, y0 - py)

def rdp(points, epsilon):
    if len(points) < 3:
        return points[:]
    s = 0
    index = 0
    a = points[0]
    b = points[-1]
    for i in range(1, len(points)-1):
        d = point_line_distance(points[i], a, b)
        if d > s:
            index = i
            s = d
    if s > epsilon:
        left = rdp(points[:index+1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [a, b]

def detectar_forma(img):
    img_gray = ImageOps.grayscale(img).resize((200, 200))
    img_gray = img_gray.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(img_gray)

    bin_map = (arr > 150).astype(np.uint8)  # fondo negro, trazo blanco
    h, w = bin_map.shape
    visited = np.zeros_like(bin_map, dtype=np.uint8)
    largest_component = []

    # detectar componente conectado más grande
    for y in range(h):
        for x in range(w):
            if bin_map[y, x] == 1 and visited[y, x] == 0:
                stack = [(x, y)]
                comp = []
                visited[y, x] = 1
                while stack:
                    cx, cy = stack.pop()
                    comp.append((cx, cy))
                    for nx, ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                        if 0 <= nx < w and 0 <= ny < h and visited[ny, nx] == 0 and bin_map[ny, nx] == 1:
                            visited[ny, nx] = 1
                            stack.append((nx, ny))
                if len(comp) > len(largest_component):
                    largest_component = comp

    if len(largest_component) < 50:
        return "❌ No se detecta forma", None, None

    comp_mask = np.zeros_like(bin_map, dtype=np.uint8)
    for (x, y) in largest_component:
        comp_mask[y, x] = 1

    border_pts = []
    for (x, y) in largest_component:
        if (x == 0 or x == w-1 or y == 0 or y == h-1 or
            comp_mask[y-1, x] == 0 or comp_mask[y+1, x] == 0 or comp_mask[y, x-1] == 0 or comp_mask[y, x+1] == 0):
            border_pts.append((x, y))

    xs = [p[0] for p in border_pts]
    ys = [p[1] for p in border_pts]
    cx = sum(xs)/len(xs)
    cy = sum(ys)/len(ys)
    angles = [math.atan2(y - cy, x - cx) for (x,y) in border_pts]
    pts_with_angles = sorted(zip(angles, border_pts))
    contour = [p for a,p in pts_with_angles]

    perim = 0.0
    for i in range(len(contour)):
        x1,y1 = contour[i]
        x2,y2 = contour[(i+1) % len(contour)]
        perim += math.hypot(x2-x1, y2-y1)

    eps = max(2.0, 0.02 * perim)
    approx = rdp(contour, eps)
    vert_count = len(approx)

    area = len(largest_component)
    circularity = (4 * math.pi * area) / (perim*perim + 1e-9)

    if vert_count <= 2:
        shape = "⭕ Círculo"
    elif vert_count == 3:
        shape = "🔺 Triángulo"
    elif vert_count == 4:
        shape = "◼️ Cuadrado"
    elif vert_count <= 6:
        shape = "📐 Polígono"
    else:
        shape = "⭕ Círculo" if circularity > 0.45 else "📐 Figura desconocida"

    bin_img = Image.fromarray((comp_mask*255).astype('uint8')).convert("RGB")
    overlay = img.convert("RGB").resize((200,200))
    draw = ImageDraw.Draw(overlay)
    if len(contour) > 1:
        draw.line(contour + [contour[0]], fill=(255,255,0), width=2)
    for (x,y) in approx:
        draw.ellipse((x-4, y-4, x+4, y+4), fill=(255,0,0))

    return shape, bin_img, overlay

def preprocesar_numero(img):
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# ================================
# Interfaz de usuario
# ================================
st.set_page_config(page_title="✍️ Detección de Números y Formas", page_icon="✍️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF5733;'>✍️ Dibuja un número o una forma geométrica</h1>", unsafe_allow_html=True)
st.write("👉 Usa el canvas de abajo. Luego presiona el botón para detectar si es un **número** o una **figura geométrica**.")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("🔢 Detectar Número"):
        if canvas_result.image_data is not None:
            arr = np.array(canvas_result.image_data)
            img = Image.fromarray(arr.astype('uint8'), 'RGBA')
            img = img.convert("RGB")
            processed = preprocesar_numero(img)
            pred = model.predict(processed)
            st.success(f"El número detectado es: **{np.argmax(pred)}**")
        else:
            st.warning("Por favor dibuja un número.")

with col2:
    if st.button("📐 Detectar Forma"):
        if canvas_result.image_data is not None:
            arr = np.array(canvas_result.image_data)
            img = Image.fromarray(arr.astype('uint8'), 'RGBA')
            img = img.convert("RGB")
            forma, bin_img, overlay = detectar_forma(img)
            st.success(f"La figura detectada es: **{forma}**")
            if bin_img is not None:
                st.image(bin_img, caption="Binarizada", width=200)
            if overlay is not None:
                st.image(overlay, caption="Contorno y vértices", width=250)
        else:
            st.warning("Por favor dibuja una figura.")

