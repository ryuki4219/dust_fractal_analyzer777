# -*- coding: utf-8 -*-
"""
fractal_app_v3_fix_polarity3.py
  - 自動2値化は大津法ではなく適応的2値化（ガウシアン）に変更
  - 粒子＝白、背景＝黒を厳密に保証
  - サイドバーの日本語も自然に修正済み
"""
import streamlit as st, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt, plotly.express as px

st.set_page_config(page_title="Fractal‑Analyzer β3", layout="centered")
st.title("フラクタル解析 Web アプリ")

# ───────────── Sidebar
with st.sidebar:
    st.header("⚙ 解析オプション")
    bin_method = st.radio(
        "2値化手法を選択",
        ["自動（適応的ガウシアン）", "適応的（ガウシアン）", "手動しきい値"],
        index=0
    )
    manual_th = st.slider("手動しきい値 (0‑255)", 0, 255, 128, 1) \
        if bin_method == "手動しきい値" else None
    max_side = st.slider("リサイズ上限（px）", 256, 1024, 800, 64)

# ───────────── Utils
@st.cache_data(show_spinner=False)
def resize_keep(img, max_px):
    h, w = img.shape[:2]
    scale = max_px / max(h, w)
    return cv2.resize(img, (int(w*scale), int(h*scale))) if scale < 1 else img

def box_count(bin_img, size):
    S = np.add.reduceat(np.add.reduceat(bin_img, np.arange(0, bin_img.shape[0], size), axis=0),
                        np.arange(0, bin_img.shape[1], size), axis=1)
    return np.count_nonzero(S)

def eval_grade(avg_score):
    return ("とても汚い" if avg_score < 25 else
            "汚い"        if avg_score < 50 else
            "綺麗"        if avg_score < 75 else
            "とても綺麗")

def enforce_white_particles(bw_img):
    """粒子(対象)を白、背景を黒 に強制"""
    white_pix = np.count_nonzero(bw_img == 255)
    return bw_img if white_pix <= bw_img.size / 2 else cv2.bitwise_not(bw_img)

# ───────────── Main Analyzer
@st.cache_data(show_spinner=True)
def analyze(file_bytes, max_side, bin_method, manual_th):
    col = resize_keep(cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR), max_side)
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)

    # ----- 二値化 -----
    if bin_method == "自動（適応的ガウシアン）":
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 2)
    elif bin_method == "適応的（ガウシアン）":
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 2)
    else:  # 手動
        bin_img = cv2.threshold(gray, manual_th, 255, cv2.THRESH_BINARY)[1]

    bin_img = enforce_white_particles(bin_img)  # ここで必ず粒子=白、背景=黒

    # ----- 指標計算 -----
    occupancy = np.count_nonzero(bin_img == 255) / bin_img.size * 100
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()
    hist_uniform = 1 / (hist.std() + 1e-9)

    cnt, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in cnt if cv2.contourArea(c) > 0])
    mean_particle = np.sqrt(areas.mean()) if areas.size else 0

    max_sz = max(2, min(bin_img.shape) // 2)
    sizes = np.unique(np.logspace(1, np.log2(max_sz), num=10, base=2, dtype=int))
    counts = [box_count(bin_img, s) for s in sizes]
    fd = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]

    scores = {
        "Occupancy%":   100 - occupancy,                        # 少ないほど高得点
        "HistUniform":  np.clip(hist_uniform * 10, 0, 100),
        "MeanParticle": np.clip(mean_particle, 0, 100),
        "FractalDim":   np.clip((2.0 - fd) * 100, 0, 100)
    }
    grade = eval_grade(np.mean(list(scores.values())))

    raw_vals = {
        "Occupancy%(粒子率)": f"{occupancy:.2f}",
        "HistUniform":        f"{hist_uniform:.3f}",
        "MeanParticle(px)":   f"{mean_particle:.1f}",
        "FractalDim":         f"{fd:.4f}"
    }
    desc = {
        "Occupancy%(粒子率)": "画像面積のうち粒子が占める割合",
        "HistUniform":        "ヒストグラム均一度（高いほどムラ少）",
        "MeanParticle(px)":   "粒子の平均径（px）",
        "FractalDim":         "フラクタル次元（表面粗さ）"
    }
    better = {
        "Occupancy%(粒子率)": "↓ 少ないほど良",
        "HistUniform":        "↑ 大きいほど良",
        "MeanParticle(px)":   "↑ 大きいほど良",
        "FractalDim":         "↓ 小さいほど良"
    }
    return col, bin_img, sizes, counts, scores, raw_vals, desc, better, grade

# ───────────── UI
u = st.file_uploader("画像を選択 (png/jpg)", ["png", "jpg", "jpeg", "bmp"])
if u:
    col, bin_img, sizes, counts, scores, raw, desc, better, grade = \
        analyze(u.read(), max_side, bin_method, manual_th)

    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(col, cv2.COLOR_BGR2RGB), caption="元画像", use_column_width=True)
    with c2:
        st.image(bin_img, caption="2値化画像 (粒子=白)", clamp=True, use_column_width=True)

    g1, g2 = st.columns(2)
    radar_df = pd.DataFrame({"Metric": list(scores), "Score": list(scores.values())})
    fig_r = px.line_polar(radar_df, r="Score", theta="Metric", line_close=True,
                          range_r=[0, 100], height=320)
    fig_r.update_traces(fill='toself')
    with g1:
        st.plotly_chart(fig_r, use_container_width=True)

    fig_fd, ax = plt.subplots(figsize=(3, 3))
    ax.plot(np.log(sizes), np.log(counts), "o-", color="tab:olive")
    ax.set_xlabel("log(Box Size)"); ax.set_ylabel("log(Count)")
    ax.set_title(f"Fractal Dimension: {raw['FractalDim']}", fontsize=9)
    with g2:
        st.pyplot(fig_fd, use_container_width=True)

    st.subheader("指標の実測値 & 説明")
    tbl = pd.DataFrame({
        "Metric": list(raw.keys()),
        "Value": [v for v in raw.values()],
        "説明":   [desc[k] for k in raw],
        "良い方向": [better[k] for k in raw]
    })
    st.table(tbl)

    st.markdown(f"### 総合判定 : **{grade}**")
