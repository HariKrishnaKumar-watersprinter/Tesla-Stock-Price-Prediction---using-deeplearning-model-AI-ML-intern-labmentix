import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Helper function for styled headers (duplicated from app.py to avoid circular import)
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True) 

def macro_price_corr(df):
    styled_header("Macro Price Indexing & Correlation", "Monthly aggregation index and cross-factor Pearson correlation")
    col7, col8 = st.columns(2)
    with col7:
        monthly_avg = df.groupby(['Year', 'Month'])['Close'].mean().reset_index()
        monthly_avg['YearMonth'] = monthly_avg['Year'].astype(str) + '-' + monthly_avg['Month'].astype(str).str.zfill(2)
  
        fig14 = px.bar(monthly_avg, x='YearMonth', y='Close', title='Monthly Average Close Price',
               color='Close', color_continuous_scale='RdYlGn')
        fig14.update_layout(xaxis_title='Year-Month', yaxis_title='Avg Close Price ($)', 
                    template='plotly_dark', xaxis_tickangle=-45)
        st.plotly_chart(fig14, use_container_width=True)
    with col8:
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Daily_Return', 'Daily_Range']
        corr_matrix = df[numeric_cols].corr()
        z_values = corr_matrix.copy()
        for i in range(len(corr_matrix.columns)):
             for j in range(len(corr_matrix.columns)):
                  if i <= j:
                      z_values.iloc[i, j] = None

        fig16 = px.imshow(z_values, 
                color_continuous_scale='peach', 
                zmin=-1, zmax=1,
                text_auto='.3f',
                 aspect="auto")
        fig16.update_layout(title='Correlation Heatmap', template='plotly_dark')
        st.plotly_chart(fig16, width='stretch')
