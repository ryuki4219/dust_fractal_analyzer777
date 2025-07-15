# -*- coding: utf-8 -*-
"""
fractal_app_v2.py – 2値化極性統一 + グラフ横並びレイアウト
"""
import streamlit as st
import cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Fractal‑Analyzer β2", layout="centered")
st.title("フラクタル解析 Web アプリ")

# ───────────── Sidebar
with st.sidebar:
    st.header("⚙ 解析オプション")
    bin_method = st.radio(
        "2値化方式",
        ["大津(自動)", "適応的(ADAPTIVE)", "手動スライダー"]
    )
    manual_th = st.slider("手動しきい値 (0‑255)", 0, 255, 128, 1) \
        if bin_method == "手動スライダー" else None
    max_side = st.slider("リサイズ上限(px)", 256, 1024, 800, 64)

# ───────────── Utility
@st.cache_data(show_spinner=False)
def resize_keep(img, max_side):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*max_side/max(h, w)), int(h*max_side/max(h, w)))) \
        if max(h, w) > max_side else img

def box_count(bin_img, size):
    S = np.add.reduceat(np.add.reduceat(bin_img, np.arange(0, bin_img.shape[0], size), axis=0),
                        np.arange(0, bin_img.shape[1], size), axis=1)
    return np.count_nonzero(S)

def eval_grade(score):
    return ("とても汚い" if score < 25 else
            "汚い"        if score < 50 else
            "綺麗"        if score < 75 else
            "とても綺麗")

# ───────────── Main
@st.cache_data(show_spinner=True)
def analyze(img_bytes, max_side, bin_method, manual_th):
    b = np.frombuffer(img_bytes, np.uint8)
    color = resize_keep(cv2.imdecode(b, cv2.IMREAD_COLOR), max_side)
    gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    if bin_method == "大津(自動)":
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    elif bin_method == "適応的(ADAPTIVE)":
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 2)
    else:
        _, bin_img = cv2.threshold(gray, manual_th, 255, cv2.THRESH_BINARY)

    # 粒子を白に統一（背景黒）
    white_pixels = np.count_nonzero(bin_img == 255)
    if white_pixels < bin_img.size / 2:
        bin_img = cv2.bitwise_not(bin_img)  # 反転して粒子を白に

    occupancy = white_pixels / bin_img.size * 100

    hist_norm = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    hist_norm /= hist_norm.sum()
    hist_uni = 1/(hist_norm.std()+1e-9)

    contours,_ = cv2.findContours(bin_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours if cv2.contourArea(c)>0])
    mean_particle = np.sqrt(areas.mean()) if areas.size else 0

    max_size = max(2, min(bin_img.shape)//2)
    sizes = np.unique(np.logspace(1, np.log2(max_size), num=10, base=2, dtype=int))
    counts= [box_count(bin_img,s) for s in sizes]
    dim   = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]

    occ_sc  = max(0, 100 - occupancy)
    uni_sc  = np.clip(hist_uni*10, 0, 100)
    mps_sc  = np.clip(mean_particle, 0, 100)
    dim_sc  = np.clip((2.0-dim)*100, 0, 100)
    overall = np.mean([occ_sc, uni_sc, mps_sc, dim_sc])
    grade   = eval_grade(overall)

    metrics = {
        "Occupancy%":  (occupancy,  occ_sc),
        "HistUniform": (hist_uni,   uni_sc),
        "MeanParticle":(mean_particle, mps_sc),
        "FractalDim":  (dim,        dim_sc)
    }
    return color, bin_img, sizes, counts, metrics, grade

u = st.file_uploader("画像を選択 (png/jpg)", type=["png","jpg","jpeg","bmp"])
if u:
    col, bin_img, sizes, counts, M, grade = analyze(u.read(), max_side, bin_method, manual_th)

    c1, c2 = st.columns(2)
    with c1: st.image(cv2.cvtColor(col, cv2.COLOR_BGR2RGB), caption="元画像", use_column_width=True)
    with c2: st.image(bin_img, caption="2値化画像", clamp=True, use_column_width=True)

    # --- レーダー & フラクタルグラフを横並び ---
    rc1, rc2 = st.columns(2)

    radar_df = pd.DataFrame({"Metric": list(M), "Score": [v[1] for v in M.values()]})
    fig_r = px.line_polar(radar_df, r="Score", theta="Metric", line_close=True,
                          range_r=[0,100], height=330)
    fig_r.update_traces(fill='toself')
    with rc1: st.plotly_chart(fig_r, use_container_width=True)

    fig_fd, ax = plt.subplots(figsize=(3,3))
    ax.plot(np.log(sizes), np.log(counts), "o-", color="tab:olive")
    ax.set_xlabel("log(Box Size)"); ax.set_ylabel("log(Count)")
    ax.set_title(f"FD: {M['FractalDim'][0]:.4f}", fontsize=10)
    with rc2: st.pyplot(fig_fd, use_container_width=True)

    st.markdown(f"### 総合判定 : **{grade}**")
