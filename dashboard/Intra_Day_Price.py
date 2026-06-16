import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Helper function for styled headers (duplicated from app.py to avoid circular import)
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)
def intra_day(df):
    styled_header("Intra-Day Price Correlation", "Open-to-Close and High-to-Low interdependency mapping")
    col1, col2 = st.columns(2)
    with col1:   
        fig9 = px.scatter(df, x='Open', y='Close', title='Open vs Close Price',
                  color='Year', opacity=0.6, color_continuous_scale='redor',
                  trendline='ols')
        fig9.update_layout(xaxis_title='Open Price ($)', yaxis_title='Close Price ($)', template='plotly_dark')
        st.plotly_chart(fig9, use_container_width=True)
    with col2:
        fig10 = px.scatter(df, x='Low', y='High', title='Chart 10: High vs Low Price',
                   color='Year', opacity=0.6, color_continuous_scale='Plasma',
                   trendline='ols')
        fig10.update_layout(xaxis_title='Low Price ($)', yaxis_title='High Price ($)', template='plotly_dark')
        st.plotly_chart(fig10, width='stretch')
        
