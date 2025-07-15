# -*- coding: utf-8 -*-
"""
fractal_app.py – UI/UX 改良版
  ✅ 4 指標（ヒストグラム均一度・平均粒径・フラクタル次元・空間占有率）
  ✅ 段階評価：「とても汚い / 汚い / 綺麗 / とても綺麗」
  ✅ レーダーチャートで一望
  ✅ フラクタル次元グラフは従来通り
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ────────────────────────────────────────── UI ──────────────────────────────────────────
st.set_page_config(page_title="Fractal‑Analyzer β", layout="centered")
st.title("フラクタル解析 Web アプリ (UI/UX 改良版)")

# ────────────────────────────────── 便利関数 / 評価関数 ─────────────────────────────────
@st.cache_data(show_spinner=False)
def resize_keep(img, max_side=800):
    """⻑辺が max_side px になるようリサイズ（縦横比保持）"""
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
    """0‑100 の“綺麗度”に換算して 4 段階で返す"""
    # score は「高いほど綺麗」に正規化済みで渡す
    if score < 25:   return "とても汚い"
    if score < 50:   return "汚い"
    if score < 75:   return "綺麗"
    return "とても綺麗"

# ────────────────────────── メイン解析関数（キャッシュ可） ───────────────────────────
@st.cache_data(show_spinner=True)
def analyze_image(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # ── ① 前処理 ──────────────────────────────────────────────
    img_color = resize_keep(img_color, 800)
    img_gray  = resize_keep(img_gray , 800)

    # 2値化（大津の二値化で精度 ↑）
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # ── ② 指標計算 ────────────────────────────────────────────
    # 2‑1. 空間占有率（白画素割合 %）
    occupancy = np.count_nonzero(binary) / binary.size * 100

    # 2‑2. ヒストグラム均一度（大きいほど均一＝綺麗とみなす）
    hist     = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()
    uniformity = hist_norm.std()  # σ が小さい→均一，ここでは 1/σ を取る
    hist_uni_score = 1 / (uniformity + 1e-9)

    # 2‑3. 平均粒径（白連結成分の平均面積の平方根 ≒ 直径）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0])
    mean_particle = np.sqrt(areas.mean()) if areas.size else 0

    # 2‑4. フラクタル次元（Box‑counting）
    max_size = max(2, min(binary.shape) // 2)
    sizes = np.unique(np.logspace(1, np.log2(max_size), num=10, base=2, dtype=int))
    counts = [box_count(binary, s) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = -coeffs[0]

    # ── ③ 正規化してスコア (0‑100) ──────────────────────────────
    # ※実運用で統計を取ってチューニング推奨
    occ_score  = max(0, 100 - occupancy)          # 占有率: 少ないほど綺麗
    uni_score  = np.clip(hist_uni_score * 10, 0, 100)
    mps_score  = np.clip(mean_particle, 0, 100)   # 大粒だとスコア低め
    frac_score = np.clip((2.0 - fractal_dim) * 100, 0, 100)  # 次元 2→最悪 0→最高

    # 総合評価は単純平均
    overall = np.mean([occ_score, uni_score, mps_score, frac_score])
    grade = eval_cleanliness(overall)

    metrics = {
        "Occupancy%": (occupancy, occ_score),
        "HistUniform": (hist_uni_score, uni_score),
        "MeanParticle": (mean_particle, mps_score),
        "FractalDim": (fractal_dim, frac_score)
    }
    return img_color, binary, sizes, counts, metrics, grade

# ───────────────────────────────── UI 入力 ──────────────────────────────────
uploaded = st.file_uploader("画像を選択 (png/jpg)", type=["png", "jpg", "jpeg", "bmp"])
if uploaded:
    (
        img_color,
        binary,
        sizes, counts,
        metrics,
        grade
    ) = analyze_image(uploaded.read())

    # ── 画像表示 ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB), caption="元画像", use_column_width=True)
    with col2:
        st.image(binary, caption="2値化画像 (大津法)", clamp=True, use_column_width=True)

    # ── レーダーチャート (4 指標) ─────────────────────────────
    radar_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Score": [m[1] for m in metrics.values()]   # 0‑100 スケール
    })
    fig_radar = px.line_polar(
        radar_df,
        r="Score", theta="Metric",
        line_close=True,
        range_r=[0, 100],
        title="Cleanliness Radar Chart (0‑100)",
    )
    fig_radar.update_traces(fill='toself')
    fig_radar.update_layout(showlegend=False, height=400, margin=dict(t=40, l=20, r=20, b=20))
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── フラクタル次元グラフ ────────────────────────────────
    fig_fd, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.log(sizes), np.log(counts), 'o-', color="tab:olive")
    ax.set_xlabel("log(Box Size)")
    ax.set_ylabel("log(Count)")
    ax.set_title(f"Fractal Dimension: {metrics['FractalDim'][0]:.4f}")
    st.pyplot(fig_fd)

    # ── 数値表示 & 評価 ──────────────────────────────────────
    st.markdown(f"### 総合判定 : **{grade}**")
    st.markdown(
        "\n".join([
            f"- **空間占有率**: {metrics['Occupancy%'][0]:.2f} %",
            f"- **ヒストグラム均一度**: {metrics['HistUniform'][0]:.4f}",
            f"- **平均粒径**: {metrics['MeanParticle'][0]:.2f} px",
            f"- **フラクタル次元**: {metrics['FractalDim'][0]:.4f}"
        ])
    )
