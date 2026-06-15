import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Helper function for styled headers (duplicated from app.py to avoid circular import)
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)


def price(df): 
    styled_header("Price Action & Liquidity Trajectory", "Historical time-series analysis of closing valuation and market participation")
    col1, col2= st.columns(2)
    col3,col4 = st.columns(2)
    with col1:
        fig1 = px.line(df, x=df.index, y='Close', title='Tesla Close Price Over Time',
               color_discrete_sequence=['#FF6B6B'])
        fig1.update_layout(xaxis_title='Years', yaxis_title='Close Price ($)', template='plotly_dark')
        st.plotly_chart(fig1, use_container_width=True)
    with col4:
        fig2 = px.area(df, x=df.index, y='Volume', color_discrete_sequence=['#FF6B6B'], template='plotly_dark')
        fig2.update_layout(title='Liquidity & Volume Dynamics', xaxis_title='Date', yaxis_title='Trade Volume')
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.line(df, x=df.index, y='High', title='Tesla High Price Over Time',
               color_discrete_sequence=["#6EDD13"])#fig3.update_traces(texttemplate='%{x:.2f}',textfont_size=20,textposition='auto')
        fig3.update_layout(xaxis_title='Years', yaxis_title='High Price ($)', template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)
    with col3:
        fig4 = px.line(df, x=df.index, y='Low', title='Tesla Low Price Over Time',
               color_discrete_sequence=["#250BBB"])

        fig4.update_layout(xaxis_title='Years', yaxis_title='Low Price ($)', template='plotly_dark')
        st.plotly_chart(fig4, use_container_width=True)