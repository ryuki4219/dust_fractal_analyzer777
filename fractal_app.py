# -*- coding: utf-8 -*-
"""
fractal_app_v3.py – 2値化極性を保持しつつ UI/UX 改良（指標表追加）
"""
import streamlit as st, cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt, plotly.express as px

st.set_page_config(page_title="Fractal‑Analyzer β3", layout="centered")
st.title("フラクタル解析 Web アプリ")

# ───────────── Sidebar
with st.sidebar:
    st.header("⚙ 解析オプション")
    bin_method = st.radio(
        "2値化方式",
        ["大津(自動)", "適応的(ADAPTIVE)", "手動スライダー"],
        index=0
    )
    manual_th = st.slider("手動しきい値 (0‑255)", 0, 255, 128, 1) \
        if bin_method == "手動スライダー" else None
    max_side = st.slider("リサイズ上限(px)", 256, 1024, 800, 64)

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

def force_white(binary):
    """白画素が少なければ反転して粒子を白に。ただし返却前に copy() で別物に"""
    white = np.count_nonzero(binary == 255)
    return binary if white >= binary.size/2 else cv2.bitwise_not(binary)

# ───────────── Main Analyzer
@st.cache_data(show_spinner=True)
def analyze(file_bytes, max_side, bin_method, manual_th):
    b   = np.frombuffer(file_bytes, np.uint8)
    col = resize_keep(cv2.imdecode(b, cv2.IMREAD_COLOR), max_side)
    gray= cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)

    if bin_method == "大津(自動)":
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    elif bin_method == "適応的(ADAPTIVE)":
        bin_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,21,2)
    else:
        _, bin_img = cv2.threshold(gray, manual_th, 255, cv2.THRESH_BINARY)

    bin_img = force_white(bin_img)     # 極性を統一（表示用にもそのまま使用）

    occupancy = np.count_nonzero(bin_img==255)/bin_img.size*100

    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    hist = hist/hist.sum()
    hist_uniform = 1/(hist.std()+1e-9)

    cnt,_ = cv2.findContours(bin_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in cnt if cv2.contourArea(c)>0])
    mean_particle = np.sqrt(areas.mean()) if areas.size else 0

    max_sz = max(2, min(bin_img.shape)//2)
    sizes  = np.unique(np.logspace(1, np.log2(max_sz), num=10, base=2, dtype=int))
    counts = [box_count(bin_img,s) for s in sizes]
    fd     = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]

    # 正規化スコア(0‑100) – 適当に線形スケール
    scores = {
        "Occupancy%":      100 - occupancy,
        "HistUniform":     np.clip(hist_uniform*10,0,100),
        "MeanParticle":    np.clip(mean_particle,0,100),
        "FractalDim":      np.clip((2.0-fd)*100,0,100)
    }
    grade = eval_grade(np.mean(list(scores.values())))

    raw_vals = {
        "Occupancy%(粒子率)": f"{occupancy:.2f}",
        "HistUniform":        f"{hist_uniform:.3f}",
        "MeanParticle(px)":   f"{mean_particle:.1f}",
        "FractalDim":         f"{fd:.4f}"
    }
    return col, bin_img, sizes, counts, scores, raw_vals, grade

# ───────────── UI
u = st.file_uploader("画像を選択 (png/jpg)", ["png","jpg","jpeg","bmp"])
if u:
    col, bin_img, sizes, counts, scores, raw, grade = analyze(u.read(), max_side, bin_method, manual_th)

    c1, c2 = st.columns(2)
    with c1: st.image(cv2.cvtColor(col, cv2.COLOR_BGR2RGB), caption="元画像", use_column_width=True)
    with c2: st.image(bin_img, caption="2値化画像 (粒子=白)", clamp=True, use_column_width=True)

    # グラフ 2 枚横並び
    g1, g2 = st.columns(2)

    radar_df = pd.DataFrame({"Metric": list(scores), "Score": list(scores.values())})
    fig_r = px.line_polar(radar_df, r="Score", theta="Metric", line_close=True,
                          range_r=[0,100], height=320)
    fig_r.update_traces(fill='toself')
    with g1: st.plotly_chart(fig_r, use_container_width=True)

    fig_fd, ax = plt.subplots(figsize=(3,3))
    ax.plot(np.log(sizes), np.log(counts), "o-", color="tab:olive")
    ax.set_xlabel("log(Box Size)"); ax.set_ylabel("log(Count)")
    ax.set_title(f"Fractal Dimension: {raw['FractalDim']}", fontsize=9)
    with g2: st.pyplot(fig_fd, use_container_width=True)

    # 指標値テーブル
    st.subheader("指標の実測値")
    tbl = pd.DataFrame({"Metric": list(raw.keys()), "Value": list(raw.values())})
    st.table(tbl)

    st.markdown(f"### 総合判定 : **{grade}**")
