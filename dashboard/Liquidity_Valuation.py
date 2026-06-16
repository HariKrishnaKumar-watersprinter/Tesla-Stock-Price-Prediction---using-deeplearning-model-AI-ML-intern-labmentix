import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Helper function for styled headers (duplicated from app.py to avoid circular import)
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True) 

def liq_val(df):
    styled_header("Liquidity vs. Valuation Dynamics", "Volume interaction with price returns and closing valuation")
    col3, col4 = st.columns(2)
    
    with col3:
        fig11 = px.scatter(df, x='Volume', y='Close', title='Volume vs Close Price',
                   color='Year', opacity=0.5, color_continuous_scale='Inferno')
        fig11.update_layout(xaxis_title='Volume', yaxis_title='Close Price ($)', template='plotly_dark')
        st.plotly_chart(fig11, use_container_width=True)
    with col4:
        fig15 = px.scatter(df, x='Daily_Return', y='Volume', color='Year', opacity=0.5, color_continuous_scale='Turbo', template='plotly_dark')
        fig15.update_layout(title='Return Anomaly vs. Liquidity', xaxis_title='Daily Return', yaxis_title='Volume')
        st.plotly_chart(fig15, width='stretch')
