import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from datetime import timedelta, datetime

# Assuming this exists
from src.Data_split import datascale

def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)

def pred():
    styled_header("📈 TSLA Stock Closing Price Predictor (1, 5, 10 Days)")
    
    uploaded_file = st.file_uploader("Upload your TSLA.csv file", type=["csv", "xlsm"])
   
    if uploaded_file is not None:
        # ====================== DATA LOADING & FEATURE ENGINEERING ======================
        Data = pd.read_csv(uploaded_file)
        Data['Date'] = pd.to_datetime(Data['Date'], format='%Y-%m-%d')
        Data2 = Data.copy()  # Keep original for display
        
        Data.set_index('Date', inplace=True)
        Data.sort_index(inplace=True)
        
        # Feature Engineering
        Data['Year'] = Data.index.year
        Data['Daily_Return'] = Data['Close'].pct_change()
        Data['Daily_Range'] = Data['High'] - Data['Low']
        Data['MACD'] = (Data['Close'].ewm(span=12, adjust=False).mean() - 
                       Data['Close'].ewm(span=26, adjust=False).mean())
        Data['OBV'] = (np.sign(Data['Close'].diff()) * Data['Volume']).cumsum()
        
        for window in [5, 10, 20]:
            Data[f'Volume_Rolling_Mean_{window}'] = Data['Volume'].rolling(window=window).mean()
            Data[f'Volatility_{window}'] = Data['Close'].rolling(window=window).std()
        
        Data['Price_Rate_Of_Change_20'] = Data['Close'].pct_change(20)
        Data['Close_Open_Ratio'] = Data['Close'] / Data['Open']
        
        Data['TR'] = np.maximum(Data['High'] - Data['Low'],
                      np.maximum(abs(Data['High'] - Data['Close'].shift(1)),
                                 abs(Data['Low'] - Data['Close'].shift(1))))
        Data['ATR_14'] = Data['TR'].rolling(window=14).mean()
        
        Data = Data.dropna()

        # ====================== PREPROCESSING & PREDICTION ======================
        possible_col = [
            'Open', 'MACD', 'OBV', 'Volume_Rolling_Mean_20', 'ATR_14', 'Volatility_20',
            'Year', 'Volume_Rolling_Mean_10', 'Close_Open_Ratio', 'Volatility_5',
            'Price_Rate_Of_Change_20', 'Volatility_10', 'Daily_Return', 'Daily_Range',
            'Volume_Rolling_Mean_5'
        ]
        
        Data1 = Data[possible_col]  # Safer than reindex
        
        preprocess_pipeline = datascale()[1]
        single = preprocess_pipeline.transform(Data1)
        single_lstm = single.reshape(single.shape[0], single.shape[1], 1)

        # Load Model
        with st.spinner("Processing data and loading models..."):
            model_path = os.path.join(os.getcwd(), "model", "LSTM.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.success("Model loaded successfully.")
            else:
                st.error(f"Model not found at: {model_path}")
                st.stop()

        predict = model.predict(single_lstm)

        # ====================== DATE SELECTION ======================
        st.subheader("📅 Select Date Range")
        min_date = Data2['Date'].min().date()
        max_date = Data2['Date'].max().date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, 
                                     min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date (Prediction starts from here)", 
                                   value=max_date, 
                                   min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.error("❌ End Date must be after Start Date.")
            st.stop()

        # Find closest date in dataset
        target_date = pd.to_datetime(end_date)
        closest_idx = Data.index.get_indexer([target_date], method='nearest')[0]
        end_date_actual = Data.index[closest_idx].date()

        if end_date_actual != end_date:
            st.warning(f"Selected date not in dataset. Using closest date: {end_date_actual}")

        # ====================== PREDICTIONS ======================
        pred_df = pd.DataFrame(
            predict.reshape(-1, 1).astype(float),  # Keep as float!
            columns=['Predicted Close price'],
            index=Data.index
        )

        end_date_idx = closest_idx

        def safe_get(idx_offset, series):
            try:
                return series.iloc[end_date_idx + idx_offset]
            except IndexError:
                return np.nan

        pred_1d = safe_get(1, pred_df['Predicted Close price'])
        pred_5d = safe_get(5, pred_df['Predicted Close price'])
        pred_10d = safe_get(10, pred_df['Predicted Close price'])

        actual_1d = safe_get(1, Data['Close'])
        actual_5d = safe_get(5, Data['Close'])
        actual_10d = safe_get(10, Data['Close'])

        current_price = Data['Close'].iloc[end_date_idx]

        def format_metric(pred, actual):
            if pd.isna(pred):
                return "N/A", "Out of range"
            if pd.isna(actual):
                return f"Predicted: ${pred:.2f}", "Actual: Future"
            err = ((pred - actual) / actual) * 100
            return f"Predicted: ${pred:.2f}", f"Actual: ${actual:.2f} ({err:+.2f}%)"

        p1, a1 = format_metric(pred_1d, actual_1d)
        p5, a5 = format_metric(pred_5d, actual_5d)
        p10, a10 = format_metric(pred_10d, actual_10d)

        st.subheader(f"🚀 Prediction from {end_date_actual}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("1-Day Ahead", p1, a1)
        col3.metric("5-Days Ahead", p5, a5)
        col4.metric("10-Days Ahead", p10, a10)

        # ====================== VISUALIZATION ======================
        st.subheader("📊 Historical Chart & Future Forecast")
        
        mask = (Data2['Date'].dt.date >= start_date) & (Data2['Date'].dt.date <= end_date_actual)
        display_df = Data2[mask].copy()

        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=display_df['Date'],
            y=display_df['Close'],
            mode='lines',
            name='Historical Close',
            line=dict(color='royalblue', width=2)
        ))

        # Predictions
        future_dates = [
            pd.to_datetime(end_date_actual) + timedelta(days=1),
            pd.to_datetime(end_date_actual) + timedelta(days=5),
            pd.to_datetime(end_date_actual) + timedelta(days=10)
        ]
        future_preds = [pred_1d, pred_5d, pred_10d]

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_preds,
            mode='markers+text',
            name='Predicted',
            marker=dict(color='crimson', size=12, symbol='star'),
            text=[f"${p:.2f}" for p in future_preds],
            textposition="top center"
        ))

        # Actuals (if available)
        actual_dates = future_dates
        actual_vals = [actual_1d, actual_5d, actual_10d]
        valid = [(d, v) for d, v in zip(actual_dates, actual_vals) if not pd.isna(v)]
        
        if valid:
            fig.add_trace(go.Scatter(
                x=[d for d, v in valid],
                y=[v for d, v in valid],
                mode='markers',
                name='Actual',
                marker=dict(color='limegreen', size=10, symbol='circle', 
                           line=dict(width=2, color='darkgreen'))
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
        st.info("☝️ Please upload your TSLA.csv file to begin.")

if __name__ == "__main__":
    pred()
