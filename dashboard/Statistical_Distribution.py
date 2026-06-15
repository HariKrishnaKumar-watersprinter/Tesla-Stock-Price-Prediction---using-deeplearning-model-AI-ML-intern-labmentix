import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Helper function for styled headers (duplicated from app.py to avoid circular import)
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)
def stat(df):
    styled_header("Statistical Distribution Profile", "Probability density and frequency analysis of returns and pricing")
    col3, col4 = st.columns(2)
    with col3:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=df['Close'], nbinsx=50, name='Close Price Distribution',
                             marker_color="#866BFF", opacity=0.75))
        fig3.update_traces(texttemplate='%{x:.2f}',textfont_size=20,textposition='auto')
        fig3.update_layout(title='Distribution of Close Price',
                   xaxis_title='Close Price ($)', yaxis_title='Frequency', template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(x=df['Daily_Return'].dropna(), nbinsx=100, 
                             name='Daily Returns', marker_color="#47DF7A", opacity=0.75))

        fig4.update_layout(title='Distribution of Daily Returns',
                   xaxis_title='Daily Return', yaxis_title='Frequency', template='plotly_dark')
        st.plotly_chart(fig4, use_container_width=True)