import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from datetime import timedelta
import datetime

# Import your preprocessing
try:
    from src.Data_split import datascale
except ImportError:
    st.error("Could not import datascale from src.Data_split")
    st.stop()

def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)
def pred():  
    styled_header("📈 TSLA Stock Closing Price Predictor (1, 5, 10 Days)")
    
    uploaded_file = st.file_uploader("Upload your TSLA.csv file", type=["csv", "xlsm"])

    if uploaded_file is not None:
        # ==================== LOAD & PREPROCESS ====================
        Data = pd.read_csv(uploaded_file)
        Data2 = Data.copy()                     # Original for display
        
        Data['Date'] = pd.to_datetime(Data['Date'], errors='coerce')
        Data = Data.dropna(subset=['Date']).sort_values('Date')
        Data.set_index('Date', inplace=True)

        # Feature Engineering
        Data['Year'] = Data.index.year
        Data['Daily_Return'] = Data['Close'].pct_change()
        Data['Daily_Range'] = Data['High'] - Data['Low']
        Data['MACD'] = Data['Close'].ewm(span=12, adjust=False).mean() - Data['Close'].ewm(span=26, adjust=False).mean()
        Data['OBV'] = (np.sign(Data['Close'].diff()) * Data['Volume']).cumsum()
        Data['Volume_Rolling_Mean_20'] = Data['Volume'].rolling(20).mean()
        Data['Volume_Rolling_Mean_10'] = Data['Volume'].rolling(10).mean()
        Data['Volume_Rolling_Mean_5'] = Data['Volume'].rolling(5).mean()
        Data['Volatility_20'] = Data['Close'].rolling(20).std()
        Data['Volatility_5'] = Data['Close'].rolling(5).std()
        Data['Volatility_10'] = Data['Close'].rolling(10).std()
        Data['Price_Rate_Of_Change_20'] = Data['Close'].pct_change(20)
        Data['Close_Open_Ratio'] = Data['Close'] / Data['Open']
        Data['TR'] = np.maximum(Data['High'] - Data['Low'],
                                np.maximum(abs(Data['High'] - Data['Close'].shift(1)),
                                           abs(Data['Low'] - Data['Close'].shift(1))))
        Data['ATR_14'] = Data['TR'].rolling(14).mean()
        
        Data = Data.dropna()

        possible_col = ['Open','MACD','OBV','Volume_Rolling_Mean_20','ATR_14',
                        'Volatility_20','Year','Volume_Rolling_Mean_10','Close_Open_Ratio',
                        'Volatility_5','Price_Rate_Of_Change_20','Volatility_10',
                        'Daily_Return','Daily_Range','Volume_Rolling_Mean_5']
        
        Data1 = Data.reindex(columns=possible_col)
        preprocess_pipeline = datascale()[1]
        X_scaled = preprocess_pipeline.transform(Data1)
        X_lstm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Load Model
        model_path = os.path.join(os.getcwd(), "model", "LSTM.pkl")
        if not os.path.exists(model_path):
            st.error(f"Model not found: {model_path}")
            st.stop()
        model = joblib.load(model_path)

        # Historical predictions
        with st.spinner("Generating predictions..."):
            historical_preds = model.predict(X_lstm).flatten()

        # ==================== DATE SELECTION & VALIDATION ====================
        min_date = Data2['Date'].min().date()
        max_date = Data2['Date'].max().date()   # Last date in uploaded file

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            selected_date = st.date_input("Prediction Start Date", 
                                        value=max_date, 
                                        min_value=min_date, max_value=datetime.date.today() + timedelta(days=30))

        # Handle future or invalid selected date
        end_dt = pd.to_datetime(selected_date)
        if end_dt not in Data.index:
            end_dt = Data.index.max()   # Use the latest available date
            st.warning(f"⚠️ Selected date **{selected_date}** is not available in your data. "
                      f"Using the latest available date: **{end_dt.date()}**")

        current_price = Data.loc[end_dt, 'Close']
        end_idx = Data.index.get_loc(end_dt)

        # ==================== FUTURE PREDICTIONS (Iterative) ====================
        def predict_future_steps(start_idx, steps):
            current_features = X_scaled[start_idx].copy().reshape(1, -1)
            predictions = []
            
            for _ in range(steps):
                pred = model.predict(current_features.reshape(1, current_features.shape[1], 1))[0][0]
                predictions.append(pred)
                # Update Close (simple approximation)
                current_features[0, 0] = pred   # Adjust index if 'Close' is not first feature after scaling
            return predictions

        pred_1d = predict_future_steps(end_idx, 1)[0]
        pred_5d = predict_future_steps(end_idx, 5)[-1]
        pred_10d = predict_future_steps(end_idx, 10)[-1]

        # ==================== METRICS ====================
        st.subheader(f"🚀 Prediction from {end_dt.date()}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("1-Day Ahead", f"${pred_1d:.2f}")
        col3.metric("5-Days Ahead", f"${pred_5d:.2f}")
        col4.metric("10-Days Ahead", f"${pred_10d:.2f}")

        # ==================== CHART ====================
        st.subheader("📊 Historical + Future Forecast")
        fig = go.Figure()

        mask = (Data2['Date'] >= pd.to_datetime(start_date)) & (Data2['Date'] <= end_dt)
        hist_df = Data2[mask]

        fig.add_trace(go.Scatter(
            x=hist_df['Date'], y=hist_df['Close'],
            mode='lines', name='Historical Close',
            line=dict(color='royalblue', width=2)
        ))

        future_dates = [
            end_dt + timedelta(days=1),
            end_dt + timedelta(days=5),
            end_dt + timedelta(days=10)
        ]
        future_prices = [pred_1d, pred_5d, pred_10d]

        fig.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            mode='markers+text', name='Predicted Future',
            marker=dict(color='crimson', size=12, symbol='star'),
            text=[f"1d: ${pred_1d:.2f}", f"5d: ${pred_5d:.2f}", f"10d: ${pred_10d:.2f}"],
            textposition="top center"
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("☝️ Please upload your TSLA.csv file to get predictions.")
