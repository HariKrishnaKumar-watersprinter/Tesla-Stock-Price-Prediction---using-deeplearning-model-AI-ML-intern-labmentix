import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Helper function for styled headers (duplicated from app.py to avoid circular import)
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True) 

def multi(df):
    styled_header("Spatial & Multi-Dimensional Mapping", "3D valuation landscape and multi-feature trajectory map")
    col1, col2 = st.columns(2)
    with col1:
        fig17 = px.scatter_3d(df, x='Open', y='Close', z='Volume', color='Year', color_continuous_scale='Viridis', opacity=0.5, template='plotly_dark')
        fig17.update_layout(title='3D Valuation Landscape')
        st.plotly_chart(fig17, use_container_width=True)
    with col2:
        fig20 = px.scatter(df, x='Close', y='Volume', size='Daily_Range', color='Year',
                   title='Bubble Chart - Close vs Volume (Size=Daily Range)',
                   color_continuous_scale='Plasma', opacity=0.6, size_max=30)
        fig20.update_layout(xaxis_title='Close Price ($)', yaxis_title='Volume', template='plotly_dark')
        st.plotly_chart(fig20, use_container_width=True)
    

    fig19 = px.scatter_matrix(df, dimensions=['Open', 'High', 'Low', 'Close', 'Volume'],
                          color='Year', title='Scatter Matrix',
                          color_continuous_scale='mint', height=800)
    fig19.update_traces(diagonal_visible=True, showupperhalf=False)
    fig19.update_layout(template='plotly_dark')
    st.plotly_chart(fig19, width='stretch')
