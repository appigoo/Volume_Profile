import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 股票成交量分布 / 筹码分布 (Volume Profile)")

# =============================
# 参数区
# =============================

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("股票代码", value="TSLA")

with col2:
    period = st.selectbox("数据周期", ["3mo", "6mo", "1y", "2y"])

with col3:
    bins = st.slider("价格分桶数量", 20, 200, 80)

# =============================
# 下载数据
# =============================

@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d")
    df = df.dropna()
    return df

df = load_data(ticker, period)

if df.empty:
    st.error("数据为空")
    st.stop()

# =============================
# 计算成交量分布
# =============================

price_min = df["Low"].min()
price_max = df["High"].max()

price_bins = np.linspace(price_min, price_max, bins)
volume_profile = np.zeros(len(price_bins))

for i in range(len(df)):
    low = df["Low"].iloc[i]
    high = df["High"].iloc[i]
    vol = df["Volume"].iloc[i]

    mask = (price_bins >= low) & (price_bins <= high)
    volume_profile[mask] += vol / max(mask.sum(), 1)

# =============================
# 计算关键区域
# =============================

vp_df = pd.DataFrame({
    "price": price_bins,
    "volume": volume_profile
})

vp_df = vp_df.sort_values("volume", ascending=False)

# POC
poc_price = vp_df.iloc[0]["price"]

# Value Area 70%
total_volume = vp_df["volume"].sum()
cum_volume = 0
value_prices = []

for _, row in vp_df.iterrows():
    cum_volume += row["volume"]
    value_prices.append(row["price"])
    if cum_volume >= total_volume * 0.7:
        break

value_low = min(value_prices)
value_high = max(value_prices)

# =============================
# 绘图
# =============================

fig = go.Figure()

# K线
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="K线"
))

# POC
fig.add_hline(y=poc_price, line_dash="dash", 
              annotation_text="POC", 
              annotation_position="top right")

# Value Area
fig.add_hrect(
    y0=value_low,
    y1=value_high,
    fillcolor="lightblue",
    opacity=0.2,
    line_width=0,
    annotation_text="Value Area 70%"
)

fig.update_layout(height=700)

st.plotly_chart(fig, use_container_width=True)

# =============================
# 横向成交量图
# =============================

st.subheader("横向成交量分布")

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=volume_profile,
    y=price_bins,
    orientation='h'
))

fig2.update_layout(
    height=700,
    yaxis_title="价格",
    xaxis_title="成交量",
)

st.plotly_chart(fig2, use_container_width=True)

# =============================
# 输出关键数据
# =============================

st.markdown("### 关键筹码区域")

colA, colB, colC = st.columns(3)

colA.metric("POC (最大成交量价位)", round(poc_price, 2))
colB.metric("Value Area Low", round(value_low, 2))
colC.metric("Value Area High", round(value_high, 2))
