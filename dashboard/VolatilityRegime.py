import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Helper function for styled headers (duplicated from app.py to avoid circular import)
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)
def vol_reg(df):
    styled_header("Volatility & Regime Density", "Quarterly price density and intra-day volatility ranges")
    col7, col8 = st.columns(2)
    with col7:
        fig7 = px.violin(df, x='Quarter', y='Close', title='Close Price Violin Plot by Quarter',
                 box=True, points='outliers', color='Quarter',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
        fig7.update_layout(xaxis_title='Quarter', yaxis_title='Close Price ($)', template='plotly_dark',
                   showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)
    with col8:
        fig8 = px.histogram(df, x='Daily_Range', nbins=80, title='Distribution of Daily Price Range',
                    color_discrete_sequence=['#F7DC6F'], opacity=0.8)
        fig8.update_traces(texttemplate='%{x:.2f}',textfont_size=50,textposition='auto')
        fig8.update_layout(xaxis_title='Daily Range ($)', yaxis_title='Frequency', template='plotly_dark')
        st.plotly_chart(fig8, width='stretch')
