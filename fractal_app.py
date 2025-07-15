import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="フラクタル解析Webアプリ", layout="wide")
st.title("フラクタル解析Webアプリ")

# --- 画像リサイズ補助関数（アスペクト比維持） ---
def resize_keep(img, max_size=800):
    h, w = img.shape[:2]
    scale = max_size / max(h, w) if max(h, w) > max_size else 1
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# --- Box Counting法の関数 ---
def box_count(img, size):
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
        np.arange(0, img.shape[1], size), axis=1)
    return np.count_nonzero(S)

# --- フラクタル次元計算 ---
def fractal_dimension(binary_img):
    max_size = min(binary_img.shape)//2
    if max_size < 2:
        return None
    sizes = np.unique(np.logspace(1, np.log2(max_size), num=10, base=2, dtype=int))
    counts = [box_count(binary_img, s) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts

# --- 空間占有率 ---
def occupancy(binary_img):
    return np.count_nonzero(binary_img == 255) / binary_img.size * 100

# --- ヒストグラム均一度（ヒストグラムのエントロピー的指標） ---
def histogram_uniformity(gray_img):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0,256])
    hist_norm = hist.ravel()/hist.sum()
    uniformity = np.sum(hist_norm**2)
    return uniformity

# --- 平均粒径（白い領域の平均サイズ） ---
def mean_particle_size(edges_img):
    contours, _ = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    areas = [cv2.contourArea(c) for c in contours]
    mean_area = np.mean(areas)
    mean_diameter = np.sqrt(4*mean_area/np.pi)  # 円換算直径
    return mean_diameter

# --- 清浄度判定（4段階） ---
def evaluate_cleanliness(occ, uniform, mpsize, fractal):
    score = 0
    score += 1 if occ < 5 else 0
    score += 1 if uniform > 0.05 else 0
    score += 1 if mpsize < 10 else 0
    score += 1 if fractal and fractal > 1.5 else 0
    grades = ["汚い", "やや汚い", "普通", "綺麗", "非常に綺麗"]
    return grades[score]

# --- Streamlit側 UI設定 ---
with st.sidebar:
    st.header("2値化設定")
    bin_mode = st.radio("2値化モード選択", ["Otsu(自動)", "手動閾値", "Adaptive"], horizontal=True)

    if bin_mode == "手動閾値":
        man_thr = st.slider("手動閾値", 0, 255, 128, 1)
    elif bin_mode == "Adaptive":
        block = st.slider("blockSize (奇数)", 3, 51, 11, 2)
        C_val = st.slider("C (補正値)", -10, 10, 2, 1)

    st.header("前処理設定")
    blur_sigma = st.slider("Gaussian Blur σ", 0.0, 5.0, 1.0, 0.1)
    canny1 = st.slider("Canny閾値1", 0, 200, 100)
    canny2 = st.slider("Canny閾値2", 0, 300, 200)

uploaded_file = st.file_uploader("画像ファイルを選択してください", type=["png", "jpg", "jpeg", "bmp"])

# --- キャッシュを利用した前処理 ---
@st.cache_data(hash_funcs={bytes: hash})
def preprocess(file_bytes, blur_sigma, can1, can2):
    arr = np.frombuffer(file_bytes, np.uint8)
    col = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if col is None:
        return None, None, None
    col_s = resize_keep(col)
    gray_s = cv2.cvtColor(col_s, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_s, (0,0), blur_sigma)
    edges = cv2.Canny(blur, can1, can2)
    return col_s, gray_s, edges

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    col_s, gray_s, edges = preprocess(image_bytes, blur_sigma, canny1, canny2)
    if col_s is None:
        st.error("画像の読み込みに失敗しました。別の画像でお試しください。")
        st.stop()

    # 2値化処理を選択肢によって変更
    if bin_mode == "Otsu(自動)":
        _, binary = cv2.threshold(gray_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif bin_mode == "手動閾値":
        _, binary = cv2.threshold(gray_s, man_thr, 255, cv2.THRESH_BINARY)
    else:  # Adaptive
        binary = cv2.adaptiveThreshold(gray_s, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       block, C_val)

    # 各種指標計算
    fractal_dim, sizes, counts = fractal_dimension(binary)
    occ_rate = occupancy(binary)
    uniformity = histogram_uniformity(gray_s)
    mean_size = mean_particle_size(edges)
    cleanliness = evaluate_cleanliness(occ_rate, uniformity, mean_size, fractal_dim)

    # 表示部分
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像（カラー）")
        st.image(cv2.cvtColor(col_s, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col2:
        st.subheader("2値化画像")
        st.image(binary, clamp=True, use_column_width=True)

    # フラクタル次元のグラフ表示
    with st.expander("フラクタル次元・Box count 曲線を表示"):
        fig, ax = plt.subplots()
        ax.plot(np.log(sizes), np.log(counts), "o-", label="Box Counting")
        ax.set_xlabel("log(Box size)")
        ax.set_ylabel("log(Count)")
        if fractal_dim is not None:
            ax.set_title(f"Fractal Dimension: {fractal_dim:.4f}")
        ax.legend()
        st.pyplot(fig)

    # 指標一覧表示
    st.markdown("---")
    st.subheader("評価指標")
    st.write(f"- **空間占有率:** {occ_rate:.2f} %")
    st.write(f"- **ヒストグラム均一度:** {uniformity:.5f}")
    st.write(f"- **平均粒径 (白領域):** {mean_size:.2f} px")
    st.write(f"- **フラクタル次元:** {fractal_dim:.4f}" if fractal_dim is not None else "- フラクタル次元の計算に失敗")

    st.markdown(f"### 清浄度評価：**{cleanliness}**")

else:
    st.info("左のサイドバーから画像をアップロードしてください。")
