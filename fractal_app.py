import streamlit as st
import cv2, numpy as np, matplotlib.pyplot as plt

st.set_page_config(page_title="フラクタル解析Webアプリ", layout="wide")
st.title("フラクタル解析Webアプリ")

# ----------  ユーティリティ ----------
def resize_keep(img, max_size=800):
    h, w = img.shape[:2]
    s = max_size / max(h, w) if max(h, w) > max_size else 1
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def box_count(img, size):
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
        np.arange(0, img.shape[1], size), axis=1)
    return np.count_nonzero(S)

def fractal_dimension(binary):
    max_size = min(binary.shape)//2
    if max_size < 2: return None, None, None
    sizes = np.unique(np.logspace(1, np.log2(max_size), num=10, base=2, dtype=int))
    counts = [box_count(binary, s) for s in sizes]
    dim = -np.polyfit(np.log(sizes), np.log(counts), 1)[0]
    return dim, sizes, counts

def occupancy(binary):               # %
    return np.count_nonzero(binary==255)/binary.size*100

def hist_uniform(gray):              # 0〜1
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist/hist.sum()
    return float(np.sum(p**2))

def mean_particle(edges):            # px
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    areas = np.array([cv2.contourArea(c) for c in cnts])
    diam = np.sqrt(4*areas/np.pi)
    return float(np.mean(diam))

# ----------  4段階評価 ----------
labels = ["とても汚い", "汚い", "綺麗", "とても綺麗"]
def judge(occ, uni, mps, frac):
    score = 0
    score += occ < 5
    score += uni > 0.05
    score += mps < 10
    score += (frac and frac > 1.5)
    return labels[score], score

# ----------  サイドバー ----------
with st.sidebar:
    st.header("2値化設定")
    mode = st.radio("方式", ["Otsu(自動)", "手動閾値", "Adaptive"], horizontal=True)
    t_manual = st.slider("手動閾値", 0, 255, 128) if mode=="手動閾値" else None
    blk = st.slider("blockSize", 3, 51, 11, 2)   if mode=="Adaptive" else None
    Cval = st.slider("C", -10, 10, 2)            if mode=="Adaptive" else None
    st.header("前処理")
    sigma = st.slider("Gaussian σ", 0.0, 5.0, 1.0, 0.1)
    c1 = st.slider("Canny1", 0, 200, 100); c2 = st.slider("Canny2", 0, 300, 200)

@st.cache_data(hash_funcs={bytes:hash})
def preprocess(buf, sigma, c1, c2):
    arr = np.frombuffer(buf, np.uint8)
    col = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if col is None: return None, None, None
    col = resize_keep(col)
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray,(0,0),sigma)
    edges = cv2.Canny(gray_blur,c1,c2)
    return col, gray_blur, edges

uploaded = st.file_uploader("画像をアップロード",["png","jpg","jpeg","bmp"])

if uploaded:
    col_img, gray_img, edges_img = preprocess(uploaded.read(),sigma,c1,c2)
    if col_img is None:
        st.error("画像読込失敗"); st.stop()

    # 2値化
    if mode=="Otsu(自動)":
        _, binary = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif mode=="手動閾値":
        _, binary = cv2.threshold(gray_img,t_manual,255,cv2.THRESH_BINARY)
    else:
        binary = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,blk,Cval)

    # 指標
    frac_dim, sizes, counts = fractal_dimension(binary)
    occ = occupancy(binary)
    uni = hist_uniform(gray_img)
    mps = mean_particle(edges_img)
    grade, score = judge(occ,uni,mps,frac_dim)

    # ---------- 表示 ----------
    c1,c2 = st.columns(2)
    c1.image(cv2.cvtColor(col_img,cv2.COLOR_BGR2RGB),"元画像",use_column_width=True)
    c2.image(binary,"2値化結果",clamp=True,use_column_width=True)

    # グラフ：左=まとめ棒グラフ，右=フラクタル曲線
    fig, ax = plt.subplots(1,2,figsize=(9,4))

    # ★まとめ棒グラフ
    metrics = ["Occupancy(%)","Hist Uni","Mean Size(px)"]
    vals    = [occ, uni*100, mps]          # scale match: hist*100
    colors  = ["#e74c3c","#3498db","#2ecc71"]
    ax[0].bar(metrics, vals, color=colors)
    ax[0].set_title("3 Metrics")
    ax[0].set_ylim(0, max(vals)*1.2)
    for i,v in enumerate(vals):
        ax[0].text(i, v+max(vals)*0.03, f"{v:.2f}", ha="center")

    # フラクタル曲線
    if frac_dim:
        ax[1].plot(np.log(sizes), np.log(counts), "o-")
        ax[1].set_title(f"Fractal Dimension ≈ {frac_dim:.3f}")
        ax[1].set_xlabel("log(box)"); ax[1].set_ylabel("log(count)")
    ax[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("数値結果")
    st.write(
f"""
| 指標 | 値 |
|------|----------------|
| 空間占有率 | **{occ:.2f}%** |
| ヒストグラム均一度 | **{uni:.4f}** |
| 平均粒径 | **{mps:.2f}px** |
| フラクタル次元 | **{frac_dim:.3f}** |
| **清浄度(4段階)** | **{grade}** |
""", unsafe_allow_html=True)
else:
    st.info("サイドバーでパラメータを調整し、画像をアップロードしてください。")
