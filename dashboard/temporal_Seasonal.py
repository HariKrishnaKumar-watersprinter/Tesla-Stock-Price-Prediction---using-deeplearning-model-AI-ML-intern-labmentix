import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Helper function for styled headers (duplicated from app.py to avoid circular import)
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)

def temp_sea(df):
    styled_header("Temporal & Seasonal Boundaries", "Yearly statistical boundaries for price and volume")
    col5, col6 = st.columns(2)
    with col5:
        fig5 = px.box(df, x='Year', y='Close', title='Close Price Distribution by Year',
              color='Year', color_discrete_sequence=px.colors.qualitative.Plotly)
        fig5.update_layout(xaxis_title='Year', yaxis_title='Close Price ($)', template='plotly_dark',
                   showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)
    with col6:
        fig6 = px.box(df, x='Year', y='Volume', title='Volume Distribution by Year',
              color='Year', color_discrete_sequence=px.colors.qualitative.Vivid)
        fig6.update_layout(xaxis_title='Year', yaxis_title='Volume', template='plotly_dark',
                   showlegend=False)
        st.plotly_chart(fig6, width='stretch')
