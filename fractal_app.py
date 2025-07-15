import streamlit as st
import cv2, numpy as np, matplotlib.pyplot as plt

st.set_page_config(page_title="フラクタル解析Webアプリ", layout="wide")
st.title("フラクタル解析Webアプリ")

# ----------  ユーティリティ  ----------
def resize_keep(img, max_size=800):
    h, w = img.shape[:2]
    scale = max_size / max(h, w) if max(h, w) > max_size else 1
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def box_count(img, size):
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
        np.arange(0, img.shape[1], size), axis=1)
    return np.count_nonzero(S)

def fractal_dimension(binary_img):
    max_size = min(binary_img.shape) // 2
    if max_size < 2: 
        return None, None, None
    sizes = np.unique(np.logspace(1, np.log2(max_size), num=10, base=2, dtype=int))
    counts = [box_count(binary_img, s) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts

def occupancy(binary):          # %
    return np.count_nonzero(binary == 255) / binary.size * 100

def hist_uniform(gray):         # 0〜1（1に近いほど均一＝白飛び少）
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist / hist.sum()
    return float(np.sum(p**2))   # Energy

def mean_particle(edges):       # px
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    areas = np.array([cv2.contourArea(c) for c in contours])
    diam = np.sqrt(4*areas/np.pi)        # 円換算直径
    return float(np.mean(diam))

# ----------  評価（4 段階） ----------
def judge(occ, uni, mps, frac):
    score = 0
    score += occ < 5            # 空間占有率低いほど +1
    score += uni > 0.05         # 均一度高いほど +1
    score += mps < 10           # 粒径小さいほど +1
    score += (frac and frac > 1.5)
    levels = ["汚い", "やや汚い", "やや綺麗", "綺麗"]     # 0〜3
    return levels[min(score,3)], score

# ----------  サイドバー  ----------
with st.sidebar:
    st.header("2値化設定")
    mode = st.radio("方式", ["Otsu(自動)","手動閾値","Adaptive"], horizontal=True)
    t_manual = st.slider("手動閾値",0,255,128) if mode=="手動閾値" else None
    blk = st.slider("blockSize",3,51,11,2) if mode=="Adaptive" else None
    Cval = st.slider("C", -10,10,2)         if mode=="Adaptive" else None
    st.header("前処理")
    sigma = st.slider("Gaussian σ",0.0,5.0,1.0,0.1)
    c1 = st.slider("Canny1",0,200,100); c2 = st.slider("Canny2",0,300,200)

# ----------  画像読込 & 前処理（キャッシュ） ----------
@st.cache_data(hash_funcs={bytes:hash})
def preprocess(buf, sigma, c1,c2):
    arr = np.frombuffer(buf,np.uint8)
    col = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if col is None: return None,None,None
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

    # --- 2値化 ---
    if mode=="Otsu(自動)":
        _, binary = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif mode=="手動閾値":
        _, binary = cv2.threshold(gray_img,t_manual,255,cv2.THRESH_BINARY)
    else:
        binary = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,blk,Cval)

    # --- 指標計算 ---
    frac_dim, sizes, counts = fractal_dimension(binary)
    occ = occupancy(binary)
    uni = hist_uniform(gray_img)
    mps = mean_particle(edges_img)
    grade, score = judge(occ,uni,mps,frac_dim)

    # ----------  表示  ----------
    col1,col2 = st.columns(2)
    with col1: st.image(cv2.cvtColor(col_img,cv2.COLOR_BGR2RGB),caption="元画像",use_column_width=True)
    with col2: st.image(binary,caption="2値化結果",clamp=True,use_column_width=True)

    # --- 指標グラフ：Matplotlib 4in1 ---
    fig, axs = plt.subplots(2,2,figsize=(6,6))
    # 1. 空間占有率
    axs[0,0].bar([0], [occ]); axs[0,0].set_title("Occupancy %"); axs[0,0].set_ylim(0,100)
    # 2. ヒスト均一度
    axs[0,1].bar([0],[uni]); axs[0,1].set_title("Histogram Uniformity"); axs[0,1].set_ylim(0,0.2)
    # 3. 平均粒径
    axs[1,0].bar([0],[mps]); axs[1,0].set_title("Mean Particle Size (px)")
    # 4. フラクタル次元 & Box-count 曲線
    if frac_dim:
        axs[1,1].plot(np.log(sizes),np.log(counts),"o-")
        axs[1,1].set_title(f"Fractal Dim ≈ {frac_dim:.3f}")
    axs[1,1].set_xlabel("log(box)"); axs[1,1].set_ylabel("log(count)")
    plt.tight_layout()
    st.pyplot(fig)

    # --- 結果テーブル ---
    st.subheader("数値結果")
    st.write(
        f"""
| 指標 | 値 |
|------|-----|
| 空間占有率 | **{occ:.2f} %** |
| ヒストグラム均一度 | **{uni:.4f}** |
| 平均粒径 | **{mps:.2f} px** |
| フラクタル次元 | **{frac_dim:.3f}** |
| **清浄度(4段階)** | **{grade}** |
        """, unsafe_allow_html=True)

else:
    st.info("サイドバーでパラメータを調整し、画像をアップロードしてください。")
