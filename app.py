import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.Price_Action import price
from dashboard.Statistical_Distribution import stat
from dashboard.temporal_Seasonal import temp_sea
from dashboard.VolatilityRegime import vol_reg
from dashboard.Intra_Day_Price import intra_day
from dashboard.Liquidity_Valuation import liq_val
from dashboard.Technical_trend import tech_trend
from dashboard.Macro_Price import macro_price_corr
from dashboard.Spatial_MultiDimensional import multi
#from dashboard.corr_heat import corr_heat
#from dashboard.anomallydetection import fraud_det
from src.data_loader import load_data as fetch_raw_data
from src.feature_engineering import create_features
from prediction.predict import pred
import datetime

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================

st.set_page_config(layout="wide", page_title="Tesla Stock Price Prediction", page_icon="💸📈")
# Industry-Grade CSS Styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d2ff;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #a6a6a6;
        text-transform: uppercase;
    }
    /* Professional Section Headers */
    .quant-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #ffffff;
        margin-top: 2.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #00d2ff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .quant-subheader {
        font-size: 0.95rem;
        font-weight: 500;
        color: #a6a6a6;
        margin-bottom: 1rem;
    }
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #262730;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for styled headers
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #00d2ff; padding-bottom: 10px; margin-bottom: 30px;">
    <h1 style="margin:0; color:#ffffff; font-weight:800;">TSLA STOCK ANALYTICS AND PREDICTION</h1>
    <span style="color:#00d2ff; font-weight:600; font-size:1.2rem;">EQUITY DERIVATIVES ANALYTICS</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    uploaded_file = st.file_uploader("Import Market Data (CSV)", type=['csv'], label_visibility="collapsed")
    options = ["🏠 Analytics", #'📊 Transaction Dynamics', #'📱 Device Dominance', 
           #'💰 Insurance Penetration', '📈 Market Expansion', '💼 User Hotspots','🗺️ Payment Performance'
           #,"📈 Trend Analysis","🚨 Competitive Benchmarking","📈 Product Development",'📊 Correlation Heatmap',"🚨Fraud Detection (ML)",
           "📈 Prediction"]
    selection = st.sidebar.radio("Select Business Case:", options)
#selection = st.sidebar.radio
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = create_features(df)
    else:
        df = create_features(fetch_raw_data())
    return df
df = load_data(uploaded_file)
if selection == "🏠 Home & Executive Summary" and not df.empty:
    #st.title("📈 Tesla Stock Price Analytics and Deep Learning Predictor ")
    #st.write("""
    #This dashboard presents findings from the PhonePe Pulse GitHub data analysis. 
    #It focuses on 5 key business cases: Transaction Growth, Device Market Share, 
    #Insurance Opportunities, Geographical Expansion, and User Registration patterns.
    #""")
    #st.markdown("### End-to-End Analysis: ETL -> SQL -> ML -> Strategy")
    # Key Metrics Row

    
    col1, col2, col3,col4 = st.columns(4)
    
    # Fetching quick stats
    
    
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Date Range", f"{df.index.min().year} - {df.index.max().year}")
    col3.metric("Avg Close Price", f"${df['Close'].mean():.2f}")
    col4.metric("Max Volume", f"{df['Volume'].max():,.0f}")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9,tab10 = st.tabs([
           "📊 Key Market Statistics",
           "💰 Price Action & Liquidity Trajectory",
           "📈 Statistical Distribution Profile",
           "📈 Temporal & Seasonal Boundaries",
           "📈 Volatility & Regime Density",
           "📊 Intra-Day Price Correlation",
           "📈 Liquidity vs. Valuation Dynamics",
           "📊 Technical Charting & Trend Architecture",
           '📊 Macro Price Indexing & Correlation',
           "📊 Spatial & Multi-Dimensional Mapping",
           
           ])
    with tab1:
        styled_header("Key Market Statistics", "Snapshot of current valuation and liquidity metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Close", f"${df['Close'].iloc[-1]:.2f}")
        col2.metric("Historical High", f"${df['High'].max():.2f}")
        col3.metric("Mean Daily Return", f"{df['Daily_Return'].mean()*100:.4f}%")
        col4.metric("Avg. Daily Volume", f"{df['Volume'].mean()/1e6:.2f}M")
    
        styled_header("Raw Data Terminal", "Historical OHLCV data feed")
        st.dataframe(df.style.format({
        'Open': '${:.2f}', 'High': '${:.2f}', 'Low': '${:.2f}', 
        'Close': '${:.2f}', 'Adj Close': '${:.2f}', 'Volume': '{:,.0f}',
        'Daily_Return': '{:.4f}'}), use_container_width=True, height=400)
    with tab2:
        price(df)
    with tab3:
        stat(df)
    with tab4:
        temp_sea(df)
    with tab5:
        vol_reg(df)
    with tab6:
        intra_day(df)
    with tab7:
        liq_val(df)
    with tab8:
        tech_trend(df)
    with tab9:
        macro_price_corr(df)
    with tab10:
        multi(df)
    #with tab11:
        #product_dep()
#elif selection == "📊 Transaction Dynamics":
    #trans_dy()
#elif selection == "📱 Device Dominance":
    #device_dominance()
#elif selection == "💰 Insurance Penetration":
    #insurance_pen()
#elif selection == "📈 Market Expansion":
    #mar_exp()
#elif selection == "💼 User Hotspots":
    #user_reg()
#elif selection == "🗺️ Payment Performance":
    #pay_cat()
#elif selection == "📈 Trend Analysis":
    #trend_anay()
#elif selection == "🚨 Competitive Benchmarking":
    #competitiveBenchmarking()
#elif selection == "📈 Product Development":
    #product_dep()
#elif selection == '📊 Correlation Heatmap':
    #corr_heat()
#elif selection == "🚨Fraud Detection (ML)":
    #fraud_det()
elif selection == "📈 Prediction":
    pred()
