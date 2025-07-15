# -*- coding: utf-8 -*-
"""
Fractal Analyzer – UI/UX 改良版
--------------------------------
✓ 元画像 / 2値化画像 を横並び表示
✓ 占有率・清潔度・フラクタル次元をカードレイアウトで表示
✓ ボックスカウントは Plotly グラフに変更（拡大やホバーが可能）
"""

import streamlit as st
import cv2
import numpy as np
import plotly.express as px


# ──────────────────────────── 基本設定 ────────────────────────────
st.set_page_config(page_title="Fractal Analyzer", layout="wide")
st.title("🌀 フラクタル解析 Web アプリ")


# ──────────────────────────── ユーティリティ ─────────────────────────
def box_count(img: np.ndarray, size: int) -> int:
    """size×size のグリッドで白画素を含むブロック数を数える"""
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
        np.arange(0, img.shape[1], size), axis=1
    )
    return np.count_nonzero(S)


def evaluate_cleanliness(rate: float) -> str:
    """白画素率[%]から清潔度を簡易分類"""
    return "汚い" if rate >= 10 else "やや汚い" if rate >= 1 else "綺麗"


def analyze_image(image_bytes: bytes):
    """画像解析メイン処理"""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img_gray is None or img_color is None:
        return None  # 失敗

    # 1) 2 値化
    _, binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    # 2) 白画素率 & 清潔度
    occupancy = np.count_nonzero(binary == 255) / binary.size * 100
    cleanliness = evaluate_cleanliness(occupancy)

    # 3) ボックスカウント
    max_size = min(binary.shape) // 2
    if max_size < 2:
        return None
    sizes = np.unique(
        np.logspace(1, np.log2(max_size), num=10, base=2, dtype=int)
    )
    counts = [box_count(binary, s) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = -coeffs[0]

    return img_color, binary, occupancy, cleanliness, fractal_dim, sizes, counts


# ──────────────────────────── UI 部品 ─────────────────────────────
uploaded_file = st.file_uploader(
    "📂 画像ファイルを選択してください（png / jpg / bmp）",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is None:
    st.info("左上のファイル選択ボックスから画像をアップロードしてください。")
    st.stop()

result = analyze_image(uploaded_file.read())
if result is None:
    st.error("⚠️ 画像の解析に失敗しました。別の画像でお試しください。")
    st.stop()

# アンパック
img_color, binary, occupancy, cleanliness, fractal_dim, sizes, counts = result

# ── (1) 画像を横並びで表示 ──
col_left, col_right = st.columns(2, gap="small")
with col_left:
    st.subheader("🖼️ 元画像（カラー）")
    st.image(
        cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB),
        caption="Original",
        use_column_width=True,
    )
with col_right:
    st.subheader("⬛⬜ 2値化画像")
    st.image(
        binary,
        caption="Binarized",
        clamp=True,
        use_column_width=True,
    )

st.markdown("---")

# ── (2) 算出値をカード風に 3 列表示 ──
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📏 空間占有率", value=f"{occupancy:.2f} %")
with col2:
    st.metric(label="🧹 清潔度評価", value=cleanliness)
with col3:
    st.metric(label="🌀 フラクタル次元", value=f"{fractal_dim:.4f}")

st.markdown("---")

# ── (3) Box-Counting グラフ (Plotly) ──
fig = px.scatter(
    x=np.log(sizes),
    y=np.log(counts),
    trendline="ols",
    labels={"x": "log(Box Size)", "y": "log(Count)"},
    title=f"Fractal Dimension (傾き) ≒ {fractal_dim:.4f}",
)
fig.update_traces(mode="lines+markers")
st.plotly_chart(fig, use_container_width=True)
