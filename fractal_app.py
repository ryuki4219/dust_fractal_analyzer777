# app.py
import streamlit as st
import cv2, numpy as np, plotly.express as px, concurrent.futures
from functools import lru_cache

st.set_page_config(page_title="Dust Analyzer", layout="wide")
st.title("フラクタル＆清浄度アナライザー")

# ------------------ ユーティリティ ------------------ #
def resize_keep(img, max_side=1024):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

@st.cache_data(hash_funcs={bytes: hash}, show_spinner=False)
def load_and_preprocess(file_bytes, blur_sigma, canny1, canny2):
    arr = np.frombuffer(file_bytes, np.uint8)
    color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    color_s = resize_keep(color)              # 表示用
    gray_s  = cv2.cvtColor(color_s, cv2.COLOR_BGR2GRAY)

    # 2値化（大津 or 固定閾値どちらでも）
    _, binary = cv2.threshold(gray_s, 0, 255, cv2.THRESH_OTSU)

    # エッジ検出 for 粒径
    blur = cv2.GaussianBlur(gray_s, (0, 0), blur_sigma)
    edges = cv2.Canny(blur, canny1, canny2)

    return color_s, gray_s, binary, edges

# ---------------------------------------------------- #
def histogram_uniformity(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()
    return float((hist**2).sum())

def mean_particle_size(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    areas = np.array([cv2.contourArea(c) for c in contours])
    diameters = 2 * np.sqrt(areas / np.pi)
    return float(diameters.mean())

def box_count(binary, k):
    S = np.add.reduceat(np.add.reduceat(binary, np.arange(0, binary.shape[0], k), 0),
                        np.arange(0, binary.shape[1], k), 1)
    return np.count_nonzero(S)

def fractal_dimension(binary):
    binary = (binary == 0).astype(np.uint8)   # 対象を黒画素に
    min_size = 2;  max_size = min(binary.shape)//2
    sizes = np.unique(np.logspace(1, np.log2(max_size), num=8, base=2, dtype=int))
    counts = [box_count(binary, s) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def occupancy(binary):
    return float((binary == 0).mean()) * 100   # 黒率

# ------------------ サイドバー ------------------ #
with st.sidebar:
    uploaded = st.file_uploader("画像をアップロード", ["png", "jpg", "jpeg"])
    st.markdown("--- **Canny パラメータ** ---")
    blur_sigma = st.slider("Gaussian σ", 0.5, 3.0, 1.0, 0.1)
    canny1 = st.slider("Canny Th1", 20, 200, 50, 5)
    canny2 = st.slider("Canny Th2", 50, 300, 150, 5)

if uploaded:
    color, gray, binary, edges = load_and_preprocess(
        uploaded.read(), blur_sigma, canny1, canny2)

    # 指標を並列計算
    with concurrent.futures.ThreadPoolExecutor() as ex:
        fut = {
            "uniform": ex.submit(histogram_uniformity, gray),
            "particle": ex.submit(mean_particle_size, edges),
            "frac": ex.submit(fractal_dimension, binary),
            "occ": ex.submit(occupancy, binary)
        }
        results = {k: f.result() for k, f in fut.items()}

    # ----------- 画面レイアウト ------------- #
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), caption="カラー", use_column_width=True)
    with col2:
        st.image(binary, caption="2値化", clamp=True, use_column_width=True)

    # メトリクス 2×2
    def to_stars(val, bins):
        idx = np.digitize(val, bins, right=True)
        return "★"*(4-idx) + "☆"*idx   # bins は昇順しきい値2,3,4要素

    m1, m2 = st.columns(2)
    m1.metric("ヒストグラム均一度", f"{results['uniform']:.2f}",
              to_stars(results['uniform'], [0.25,0.5,0.75]))
    m2.metric("平均粒径(px)", f"{results['particle']:.2f}",
              to_stars(results['particle'], [3,6,9]))
    m3, m4 = st.columns(2)
    m3.metric("フラクタル次元", f"{results['frac']:.3f}",
              to_stars(results['frac'], [1.2,1.4,1.6]))
    m4.metric("空間占有率(%)", f"{results['occ']:.2f}",
              to_stars(results['occ'], [1,5,10]))

    # レーダーチャート
    import plotly.express as px, pandas as pd
    df = pd.DataFrame({
        "指標": ["均一度", "粒径", "フラクタル", "占有率"],
        "値" :  [results['uniform'], results['particle'],
                 results['frac'], results['occ']]
    })
    fig = px.line_polar(df, r="値", theta="指標", line_close=True,
                        range_r=[0, max(df['値'])*1.2])
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ヒストグラム＆粒径分布"):
        hist_fig = px.histogram(np.ravel(gray), nbins=50, title="Gray Histogram")
        st.plotly_chart(hist_fig, use_container_width=True)
