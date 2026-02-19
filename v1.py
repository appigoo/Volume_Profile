# -*- coding: utf-8 -*-
"""
Streamlit 专业股票筹码分布 / 成交量价格分布 应用
支持 A股市场（需替换数据源）

两种算法可选：
1. 简单平均分布（适合快速查看）
2. 带换手衰减的迭代累积法（更接近主流软件）

依赖库：
pip install streamlit pandas numpy plotly yfinance akshare
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─── 数据获取部分 ───
# 你可以替换成自己的数据源，例如 akshare、tushare、baostock 等
# 这里用 yfinance 作为演示（美股），A股请换成 akshare 或 tushare

@st.cache_data(ttl=3600)
def load_stock_data(ticker: str, days: int = 500):
    """获取历史 OHLCV 数据"""
    try:
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=days + 100)  # 多取一点做缓冲
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index.name = 'date'
        df.reset_index(inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        st.error(f"数据下载失败：{e}")
        return None


# ─── 筹码分布核心计算 ───

def calc_chip_simple(df: pd.DataFrame, lookback_days: int = 250):
    """
    方法1：简单平均分布法（最常用入门算法）
    假设每天换手筹码在最高-最低价之间均匀分布
    """
    df = df.tail(lookback_days).copy()
    if len(df) == 0:
        return None, None

    prices = []
    volumes = []

    for _, row in df.iterrows():
        if row['Volume'] <= 0:
            continue
        low, high, vol = row['Low'], row['High'], row['Volume']
        if high <= low:
            continue
        # 均匀分布
        price_step = np.linspace(low, high, 50)  # 每根K线分50份
        vol_per_step = vol / len(price_step)
        prices.extend(price_step)
        volumes.extend([vol_per_step] * len(price_step))

    if not prices:
        return None, None

    # 按价格聚合
    price_bins = np.arange(min(prices) * 0.95, max(prices) * 1.05, (max(prices) - min(prices)) / 200)
    hist, bin_edges = np.histogram(prices, bins=price_bins, weights=volumes)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    chip_df = pd.DataFrame({'price': bin_centers, 'chip': hist})
    chip_df = chip_df[chip_df['chip'] > 0]
    
    return chip_df.sort_values('price'), "简单平均分布法"


def calc_chip_decay(df: pd.DataFrame, lookback_days: int = 210, decay_factor: float = 0.92):
    """
    方法2：带换手衰减的迭代累积法（更接近专业软件）
    每天新换手筹码覆盖旧筹码的一部分
    """
    df = df.tail(lookback_days * 2).copy()  # 多取一点做初始化
    if len(df) < 30:
        return None, None

    # 初始化一个价格区间 → 权重字典
    min_p = df['Low'].min() * 0.85
    max_p = df['High'].max() * 1.15
    bins = np.linspace(min_p, max_p, 301)  # 300个区间
    bin_centers = (bins[:-1] + bins[1:]) / 2
    chip_dist = np.zeros(len(bin_centers))  # 当前筹码权重

    total_circulation = 0  # 累计有效流通量（用于归一）

    for i in range(len(df)):
        row = df.iloc[i]
        vol = row['Volume']
        if vol <= 0:
            continue

        turnover = vol  # 这里简化，实际应除以流通股本得到换手率

        # 新增筹码 - 三角分布（低-均-高）
        low, high, close = row['Low'], row['High'], row['Close']
        mean_p = (low + high + close) / 3

        mask_low = (bin_centers >= low) & (bin_centers <= mean_p)
        mask_high = (bin_centers >= mean_p) & (bin_centers <= high)

        if mask_low.any():
            peak_low = np.argmax(mask_low)
            dist_low = np.linspace(0, 1, mask_low.sum())
            dist_low /= dist_low.sum() or 1
            chip_dist[mask_low] += turnover * dist_low * 0.5

        if mask_high.any():
            peak_high = np.argmax(mask_high)
            dist_high = np.linspace(1, 0, mask_high.sum())
            dist_high /= dist_high.sum() or 1
            chip_dist[mask_high] += turnover * dist_high * 0.5

        # 旧筹码衰减
        chip_dist *= decay_factor ** (turnover / 10000)  # 粗略模拟衰减

        total_circulation += turnover

    if total_circulation == 0:
        return None, None

    chip_dist /= chip_dist.sum() or 1  # 归一化成百分比
    chip_df = pd.DataFrame({
        'price': bin_centers,
        'chip': chip_dist * 100   # 百分比
    }).query("chip > 0.01")

    return chip_df.sort_values('price'), "迭代衰减累积法"


# ─── Streamlit 主程序 ───

def main():
    st.set_page_config(page_title="股票筹码分布分析", layout="wide")
    st.title("股票筹码分布 / 成本分布分析工具")
    st.caption("支持简单平均法 与 迭代衰减法两种主流算法（2026版）")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        ticker = st.text_input("股票代码（美股示例：AAPL, TSLA；A股请替换数据源）", "AAPL").upper().strip()

    with col2:
        lookback = st.slider("回看交易日数", 30, 600, 250, step=10)

    with col3:
        method = st.radio("计算方法", ["简单平均分布法", "迭代衰减累积法"], index=0)

    if st.button("开始计算", type="primary"):
        with st.spinner("正在下载数据并计算筹码分布..."):
            df = load_stock_data(ticker, lookback + 100)
            if df is None or len(df) < 30:
                st.error("数据不足或下载失败，请检查代码/网络")
                return

            if method == "简单平均分布法":
                chip_df, method_name = calc_chip_simple(df, lookback)
            else:
                chip_df, method_name = calc_chip_decay(df, lookback)

            if chip_df is None:
                st.warning("筹码计算结果为空")
                return

            # ─── 绘图 ───
            fig = go.Figure()

            # 筹码分布（横向柱状图）
            fig.add_trace(go.Bar(
                y=chip_df['price'],
                x=chip_df['chip'],
                orientation='h',
                name='筹码量',
                marker_color='rgba(50, 171, 96, 0.7)',
                hovertemplate='价格: %{y:.2f}<br>筹码占比: %{x:.2f}%<extra></extra>'
            ))

            # 当前价格参考线
            current_price = df['Close'].iloc[-1]
            fig.add_vline(x=0, line_width=0)
            fig.add_hline(y=current_price, line_dash="dash", line_color="red",
                          annotation_text=f"现价 {current_price:.2f}", 
                          annotation_position="right",
                          annotation_font_color="red")

            fig.update_layout(
                title=f"{ticker} 筹码分布图（{method_name}） - 最近 {lookback} 天",
                xaxis_title="筹码占比 (%)",
                yaxis_title="价格区间",
                height=700,
                showlegend=False,
                bargap=0.05,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

            # 显示主要筹码峰
            st.subheader("主要筹码峰（前3）")
            peaks = chip_df.nlargest(3, 'chip').round(2)
            st.dataframe(peaks, hide_index=True)

            # 原始数据预览
            with st.expander("最近30日K线数据"):
                st.dataframe(df.tail(30))


if __name__ == "__main__":
    main()
