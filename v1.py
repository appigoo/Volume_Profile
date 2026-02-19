import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(layout="wide", page_title="Stock Chips Distribution")

st.title("📊 股票成交量分布 (筹码分布) 分析工具")

# 侧边栏参数设置
with st.sidebar:
    st.header("参数设置")
    symbol = st.text_input("股票代码 (如 AAPL, TSLA, 0700.HK)", value="AAPL")
    period = st.selectbox("时间范围", ["3mo", "6mo", "1y", "2y", "5y"], index=1)
    bins = st.slider("价格档位数量 (Bins)", min_value=20, max_value=100, value=50)

# 获取数据
@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period)
    # 处理 MultiIndex 列名（新版 yfinance 特性）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

try:
    data = load_data(symbol, period)

    if data.empty:
        st.error("未找到数据，请检查股票代码。")
    else:
        # 计算成交量分布 (Volume Profile)
        # 我们取收盘价作为参考点，统计每个价格区间的成交量总和
        price_min = data['Low'].min()
        price_max = data['High'].max()
        
        # 创建价格区间
        data['bin'] = pd.cut(data['Close'], bins=bins)
        volume_profile = data.groupby('bin', observed=True)['Volume'].sum().reset_index()
        
        # 提取区间的中点用于绘图
        volume_profile['price_mid'] = volume_profile['bin'].apply(lambda x: x.mid)

        # --- 开始绘图 ---
        # 创建子图：左侧为主图 (K线)，右侧为成交量分布
        fig = make_subplots(
            rows=1, cols=2, 
            shared_yaxes=True, 
            column_widths=[0.8, 0.2],
            horizontal_spacing=0.01,
            subplot_titles=(f"{symbol} K线图", "筹码分布")
        )

        # 1. 添加 K 线图
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="K线"
            ),
            row=1, col=1
        )

        # 2. 添加筹码分布 (水平柱状图)
        fig.add_trace(
            go.Bar(
                x=volume_profile['Volume'],
                y=volume_profile['price_mid'],
                orientation='h',
                marker_color='rgba(100, 149, 237, 0.6)',
                name="成交量分布"
            ),
            row=1, col=2
        )

        # 更新布局
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=False,
            template="plotly_dark",
            yaxis_title="价格",
            xaxis_title="日期"
        )
        
        fig.update_xaxes(title_text="成交量累积", row=1, col=2)

        # 展示图表
        st.plotly_chart(fig, use_container_width=True)

        # 数据指标展示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("当前价格", f"${data['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("最高价 (区间)", f"${price_max:.2f}")
        with col3:
            st.metric("最低价 (区间)", f"${price_min:.2f}")

except Exception as e:
    st.error(f"发生错误: {e}")
