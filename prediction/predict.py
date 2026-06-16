import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from datetime import timedelta
import datetime

# Placeholder for your preprocessing pipeline (update import if needed)
try:
    from src.Data_split import datascale
except ImportError:
    st.warning("Could not import datascale. Using identity transform.")
    def datascale():
        class Dummy:
            def transform(self, X): return X.values
        return None, Dummy()

def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)

def pred():  
    styled_header("📈 TSLA Stock Closing Price Predictor (1, 5, 10 Days)")
    uploaded_file = st.file_uploader("Upload your TSLA.csv file", type=["csv", "xlsm"])

    if uploaded_file is not None:
        # Load data
        Data = pd.read_csv(uploaded_file)
        Data2 = Data.copy()  # Keep original for display
        
        Data['Date'] = pd.to_datetime(Data['Date'], errors='coerce')
        Data2 = Data.copy()
        Data = Data.dropna(subset=['Date'])
        Data.set_index('Date', inplace=True)
        Data.sort_index(inplace=True)

        # Feature Engineering
        Data['Year'] = Data.index.year
        Data['Daily_Return'] = Data['Close'].pct_change()
        Data['Daily_Range'] = Data['High'] - Data['Low']
        Data['MACD'] = Data['Close'].ewm(span=12, adjust=False).mean() - Data['Close'].ewm(span=26, adjust=False).mean()
        Data['OBV'] = (np.sign(Data['Close'].diff()) * Data['Volume']).cumsum()
        Data['Volume_Rolling_Mean_20'] = Data['Volume'].rolling(window=20).mean()
        Data['Volume_Rolling_Mean_10'] = Data['Volume'].rolling(window=10).mean()
        Data['Volume_Rolling_Mean_5'] = Data['Volume'].rolling(window=5).mean()
        Data['Volatility_20'] = Data['Close'].rolling(window=20).std()
        Data['Volatility_5'] = Data['Close'].rolling(window=5).std()
        Data['Volatility_10'] = Data['Close'].rolling(window=10).std()
        Data['Price_Rate_Of_Change_20'] = Data['Close'].pct_change(20)
        Data['Close_Open_Ratio'] = Data['Close'] / Data['Open']
        Data['TR'] = np.maximum(Data['High'] - Data['Low'],
                                np.maximum(abs(Data['High'] - Data['Close'].shift(1)),
                                           abs(Data['Low'] - Data['Close'].shift(1))))
        Data['ATR_14'] = Data['TR'].rolling(window=14).mean()
        Data = Data.dropna()

        possible_col = ['Open','MACD','OBV','Volume_Rolling_Mean_20','ATR_14',
                        'Volatility_20','Year','Volume_Rolling_Mean_10','Close_Open_Ratio',
                        'Volatility_5','Price_Rate_Of_Change_20','Volatility_10',
                        'Daily_Return','Daily_Range','Volume_Rolling_Mean_5']
        
        Data1 = Data.reindex(columns=possible_col)
        preprocess_pipeline = datascale()[1]
        single = preprocess_pipeline.transform(Data1)
        single_lstm = single.reshape(single.shape[0], single.shape[1], 1)

        # Load model
        with st.spinner("Loading model..."):
            model_path = os.path.join(os.getcwd(), "model", "LSTM.pkl")
            if not os.path.exists(model_path):
                st.error(f"Model not found at {model_path}")
                st.stop()
            model = joblib.load(model_path)
            st.success("Model loaded.")

        # Generate predictions for all rows
        predict = model.predict(single_lstm)

        # Date selection
        min_date = Data2['Date'].min().date()
        max_date = Data2['Date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date (Prediction starts from here)", 
                                   value=min(max_date, datetime.date.today()), 
                                   min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.error("End Date must be after Start Date.")
            st.stop()

        # Filter for chart
        mask = (Data2['Date'] >= pd.to_datetime(start_date)) & (Data2['Date'] <= pd.to_datetime(end_date))
        display_df = Data2[mask]

        # Selected end date
        end_dt = pd.to_datetime(end_date)
        selected_end_row = Data2[Data2['Date'] == end_dt]
        if selected_end_row.empty:
            st.warning("Selected End Date is not a trading day. Please pick a valid date.")
            st.stop()

        current_price = selected_end_row['Close'].values[0]

        # Index in processed data
        if end_dt not in Data.index:
            st.warning("Selected date dropped during preprocessing (NaNs).")
            st.stop()
            
        end_date_idx = Data.index.get_loc(end_dt)

        # Predictions DataFrame
        pred_df = pd.DataFrame(predict.reshape(-1, 1), 
                               columns=['Predicted Close'], 
                               index=Data.index)

        # Safe getters
        def safe_pred(offset):
            idx = end_date_idx + offset
            if idx < len(pred_df):
                return pred_df.iloc[idx]['Predicted Close']
            return np.nan

        def safe_actual(offset):
            idx = end_date_idx + offset
            if idx < len(Data):
                return Data['Close'].iloc[idx]
            return np.nan

        pred_1d = safe_pred(1)
        pred_5d = safe_pred(5)
        pred_10d = safe_pred(10)

        actual_1d = safe_actual(1)
        actual_5d = safe_actual(5)
        actual_10d = safe_actual(10)

        def format_metric(pred, actual):
            if pd.isna(pred):
                return "Predicted: N/A (Beyond data)", ""
            pred_str = f"Predicted: ${pred:.2f}"
            if pd.isna(actual):
                return pred_str, "Actual: Not available (future date)"
            else:
                err = ((pred - actual) / actual) * 100
                return pred_str, f"Actual: ${actual:.2f} ({err:+.2f}%)"

        p1, a1 = format_metric(pred_1d, actual_1d)
        p5, a5 = format_metric(pred_5d, actual_5d)
        p10, a10 = format_metric(pred_10d, actual_10d)

        st.subheader(f"🚀 Prediction from {end_date.strftime('%Y-%m-%d')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${current_price:.2f}")
        c2.metric("1-Day Ahead", p1, a1 if "Not available" not in a1 else None)
        c3.metric("5-Days Ahead", p5, a5 if "Not available" not in a5 else None)
        c4.metric("10-Days Ahead", p10, a10 if "Not available" not in a10 else None)

        # Chart
        st.subheader("📊 Historical Chart & Forecast")
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=display_df['Date'], y=display_df['Close'],
                                 mode='lines', name='Historical Close',
                                 line=dict(color='royalblue', width=2)))

        # Predictions
        future_dates, future_vals, labels = [], [], []
        for days, p in zip([1, 5, 10], [pred_1d, pred_5d, pred_10d]):
            if not pd.isna(p):
                future_dates.append(end_dt + timedelta(days=days))
                future_vals.append(p)
                labels.append(f"{days}d: ${p:.2f}")

        if future_vals:
            fig.add_trace(go.Scatter(x=future_dates, y=future_vals,
                                     mode='markers+text', name='Predicted',
                                     marker=dict(color='crimson', size=12, symbol='star'),
                                     text=labels, textposition="top center"))

        # Actuals (if available)
        act_dates, act_vals = [], []
        for days, a in zip([1, 5, 10], [actual_1d, actual_5d, actual_10d]):
            if not pd.isna(a):
                act_dates.append(end_dt + timedelta(days=days))
                act_vals.append(a)

        if act_vals:
            fig.add_trace(go.Scatter(x=act_dates, y=act_vals,
                                     mode='markers', name='Actual',
                                     marker=dict(color='limegreen', size=10, symbol='circle',
                                                line=dict(width=2, color='black'))))

        fig.update_layout(
            xaxis_title="Date", yaxis_title="Price (USD)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please upload a CSV file to proceed.")
