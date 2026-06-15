import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Helper function for styled headers (duplicated from app.py to avoid circular import)
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True) 

def tech_trend(df):
    styled_header("Technical Charting & Trend Architecture", "Price action structure and moving average overlays")
    #col5, col6 = st.columns(2)
   
    fig12 = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'])])
    fig12.update_layout(title='Tesla OHLC Candlestick Chart',
                    xaxis_title='Year', yaxis_title='Price ($)', template='plotly_dark')
    st.plotly_chart(fig12, use_container_width=True)
    
    df1=df.copy()
    df1['MA_3'] = df1['Close'].rolling(window=3).mean()
    df1['MA_5'] = df1['Close'].rolling(window=5).mean()
    df1['MA_10'] = df1['Close'].rolling(window=10).mean()
    df1['MA_20'] = df1['Close'].rolling(window=20).mean()

    fig13 = go.Figure()
    fig13.add_trace(go.Scatter(x=df1.index, y=df1['Close'], name='Close', line=dict(color='#FF6B6B', width=1)))
    fig13.add_trace(go.Scatter(x=df1.index, y=df1['MA_3'], name='MA 3', line=dict(color="#702AE2", width=1.5)))
    fig13.add_trace(go.Scatter(x=df1.index, y=df1['MA_5'], name='MA 5', line=dict(color='#4ECDC4', width=1.5)))
    fig13.add_trace(go.Scatter(x=df1.index, y=df1['MA_10'], name='MA 10', line=dict(color="#E4BD20", width=1.5)))
    fig13.add_trace(go.Scatter(x=df1.index, y=df1['MA_20'], name='MA 20', line=dict(color='#BB8FCE', width=1.5)))
    fig13.update_layout(title='Close Price with Moving Averages',
                    xaxis_title='Date', yaxis_title='Price ($)', template='plotly_dark')
    st.plotly_chart(fig13, use_container_width=True)