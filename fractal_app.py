# -*- coding: utf-8 -*-
"""
fractal_app.py – 2値化精度とサイドバー UI 改良版
  ✔️ 2値化方式を選択（大津 / 適応 / 手動しきい値）
  ✔️ サイドバーに全オプションを再配置
  ✔️ 既存の 4 指標 + レーダーチャート & フラクタル次元グラフは維持
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ────────────────────────────── Streamlit ページ設定 ──────────────────────────────
st.set_page_config(page_title="Fractal‑Analyzer β2", layout="centered")

st.title("フラクタル解析 Web アプリ")

# ────────────────────────────── Sidebar オプション ──────────────────────────────
with st.sidebar:
    st.header("⚙ 解析オプション")
    bin_method = st.radio(
        "2値化方式",
        ["大津(自動)", "適応的(ADAPTIVE)", "手動スライダー"],
        index=0
    )
    manual_th = None
    if bin_method == "手動スライダー":
        manual_th = st.slider("手動しきい値 (0‑255)", 0, 255, 128, 1)
    invert_bin = st.checkbox("白黒を反転 (埃を白で解析)", value=True)
    max_side   = st.slider("リサイズ上限(px)", 256, 1024, 800, 64)

# ────────────────────────────── 便利関数 ──────────────────────────────
@st.cache_data(show_spinner=False)
def resize_keep(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def box_count(bin_img, size):
    S = np.add.reduceat(
        np.add.reduceat(bin_img, np.arange(0, bin_img.shape[0], size), axis=0),
        np.arange(0, bin_img.shape[1], size), axis=1)
    return np.count_nonzero(S)

def eval_cleanliness(score):
    if score < 25:   return "とても汚い"
    if score < 50:   return "汚い"
    if score < 75:   return "綺麗"
    return "とても綺麗"

# ────────────────────────────── メイン解析 ──────────────────────────────
@st.cache_data(show_spinner=True)
def analyze(img_bytes, max_side, bin_method, manual_th, invert):
    # ---------- 読み込み & 前処理 ----------
    file_bytes = np.frombuffer(img_bytes, np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_color = resize_keep(img_color, max_side)
    img_gray  = resize_keep(img_gray , max_side)

    # ---------- 2値化 ----------
    if bin_method == "大津(自動)":
        _, binary = cv2.threshold(img_gray, 0, 255,
                                  cv2.THRESH_OTSU |
                                  (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY))
    elif bin_method == "適応的(ADAPTIVE)":
        adaptive = cv2.adaptiveThreshold(
            img_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or MEAN_C
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            21, 2)
        binary = adaptive
    else:
        _, binary = cv2.threshold(
            img_gray, manual_th, 255,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)

    # ---------- 指標計算 ----------
    occupancy = np.count_nonzero(binary) / binary.size * 100

    hist      = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()
    hist_uni  = 1 / (hist_norm.std() + 1e-9)     # 大 → 均一

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0])
    mean_particle = np.sqrt(areas.mean()) if areas.size else 0

    max_size = max(2, min(binary.shape)//2)
    sizes = np.unique(np.logspace(1, np.log2(max_size), num=10, base=2, dtype=int))
    counts = [box_count(binary, s) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = -coeffs[0]

    # ---------- スコア正規化 (0‑100) ----------
    occ_score  = max(0, 100 - occupancy)
    uni_score  = np.clip(hist_uni*10, 0, 100)
    mps_score  = np.clip(mean_particle, 0, 100)
    frac_score = np.clip((2.0 - fractal_dim)*100, 0, 100)

    overall = np.mean([occ_score, uni_score, mps_score, frac_score])
    grade   = eval_cleanliness(overall)

    metrics = {
        "Occupancy%":  (occupancy, occ_score),
        "HistUniform": (hist_uni,  uni_score),
        "MeanParticle":(mean_particle, mps_score),
        "FractalDim":  (fractal_dim, frac_score)
    }
    return img_color, binary, sizes, counts, metrics, grade

# ────────────────────────────── 画像入力 ──────────────────────────────
uploaded = st.file_uploader("画像を選択 (png/jpg)", type=["png", "jpg", "jpeg", "bmp"])
if uploaded:
    img_c, bin_img, sizes, counts, metrics, grade = analyze(
        uploaded.read(), max_side, bin_method, manual_th, invert_bin)

    # ---------- 表示 ----------
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB), caption="元画像", use_column_width=True)
    with col2:
        st.image(bin_img, caption=f"2値化画像 – {bin_method}", clamp=True, use_column_width=True)

    # レーダーチャート
    radar_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Score":  [v[1] for v in metrics.values()]
    })
    fig_r = px.line_polar(radar_df, r="Score", theta="Metric",
                          line_close=True, range_r=[0, 100],
                          title="Cleanliness Radar (0‑100)")
    fig_r.update_traces(fill='toself')
    st.plotly_chart(fig_r, use_container_width=True)

    # フラクタル次元グラフ
    fig_fd, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.log(sizes), np.log(counts), "o-", color="tab:olive")
    ax.set_xlabel("log(Box Size)")
    ax.set_ylabel("log(Count)")
    ax.set_title(f"Fractal Dimension: {metrics['FractalDim'][0]:.4f}")
    st.pyplot(fig_fd)

    # 数値 & 総合判定
    st.markdown(f"### 総合判定 : **{grade}**")
    st.markdown("\n".join([
        f"- **空間占有率**: {metrics['Occupancy%'][0]:.2f} %",
        f"- **ヒストグラム均一度**: {metrics['HistUniform'][0]:.4f}",
        f"- **平均粒径**: {metrics['MeanParticle'][0]:.2f} px",
        f"- **フラクタル次元**: {metrics['FractalDim'][0]:.4f}"
    ]))
