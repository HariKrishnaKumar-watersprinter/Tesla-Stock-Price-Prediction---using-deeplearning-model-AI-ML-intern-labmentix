import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import joblib
import os
from datetime import timedelta
from src.Data_split import datascale
import numpy as np
import datetime
def styled_header(title, subtitle=""):
    st.markdown(f'<div class="quant-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="quant-subheader">{subtitle}</div>', unsafe_allow_html=True)

def pred():  
    styled_header("📈 TSLA Stock Closing Price Predictor (1, 5, 10 Days)")
    uploaded_file = st.file_uploader("Upload your TSLA.csv file", type=["csv", 'xlsm'])

    if uploaded_file is not None:
        Data = pd.read_csv(uploaded_file)
        Data['Date'] = pd.to_datetime(Data['Date'], format='%Y-%m-%d')
        Data2 = Data.copy()
        Data.set_index('Date', inplace=True)
        Data.sort_index(inplace=True)

        # ---- Feature engineering ----
        Data['Year'] = Data.index.year
        Data['Daily_Return'] = Data['Close'].pct_change()
        Data['Daily_Range'] = Data['High'] - Data['Low']
        Data['MACD'] = Data['Close'].ewm(span=12, adjust=False).mean() - Data['Close'].ewm(span=26, adjust=False).mean()
        Data['OBV'] = (np.sign(Data['Close'].diff()) * Data['Volume']).cumsum()
        Data['Volume_Rolling_Mean_20'] = Data['Volume'].rolling(window=20).mean()
        Data['Volume_Rolling_Mean_10'] = Data['Volume'].rolling(window=10).mean()
        Data['Volume_Rolling_Mean_5']  = Data['Volume'].rolling(window=5).mean()
        Data['Volatility_20'] = Data['Close'].rolling(window=20).std()
        Data['Volatility_5']  = Data['Close'].rolling(window=5).std()
        Data['Volatility_10'] = Data['Close'].rolling(window=10).std()
        Data['Price_Rate_Of_Change_20'] = Data['Close'].pct_change(20)
        Data['Close_Open_Ratio'] = Data['Close'] / Data['Open']
        Data['TR'] = np.maximum(Data['High'] - Data['Low'],
                       np.maximum(abs(Data['High'] - Data['Close'].shift(1)),
                                  abs(Data['Low']  - Data['Close'].shift(1))))
        Data['ATR_14'] = Data['TR'].rolling(window=14).mean()
        Data = Data.dropna()

        possible_col=['Open','Volume_Rolling_Mean_20','Year','Volume_Rolling_Mean_10','Daily_Range','Volume_Rolling_Mean_5']
        Data1 = Data.reindex(columns=possible_col)

        preprocess_pipeline = datascale()[1]
        single = preprocess_pipeline.transform(Data1)
        single_lstm = single.reshape(single.shape[0], single.shape[1], 1)

        with st.spinner("Processing data and loading models... Please wait."):
            model_path = os.path.join(os.getcwd(), "model", "LSTM.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.success("Model loaded successfully.")
            else:
                st.error(f"Error: Model file not found at: {model_path}. Please ensure 'LSTM.pkl' exists in the 'model' directory.")
                st.stop()

        predict = model.predict(single)
        pred_df = pd.DataFrame(predict.reshape(-1, 1).astype(float),
                               columns=['Predicted Close price'],
                               index=Data.index)

        # ---- Date range selection ----
        st.subheader("📅 Select Date Range")
        min_date = Data2['Date'].min().date()
        max_date = Data2['Date'].max().date()
        today    = datetime.datetime.now().date()

        # Clamp default end_date to the dataset's max_date so we never pick a future date
        default_end = min(today, max_date)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date (Prediction starts from here)",
                                     value=default_end,
                                     min_value=min_date,
                                     max_value=max_date)

        if start_date > end_date:
            st.error("❌ Error: End Date must fall after Start Date.")
            st.stop()

        # ---- Safe index lookup ----
        end_ts = pd.Timestamp(end_date)

        if end_ts not in pred_df.index:
            available_dates = pred_df.index[pred_df.index <= end_ts]
            if len(available_dates) == 0:
                st.error("Selected date is not available in dataset.")
                st.stop()

            end_ts = available_dates[-1]

        end_date_idx = pred_df.index.get_loc(end_ts)
        data_len = len(pred_df)

        # ---- Build prediction DataFrame aligned to Data.index ----
        

        # ---- Extract 1d / 5d / 10d predictions & actuals ----
        def safe_get(df_or_series, pos):
            return df_or_series.iloc[pos] if 0 <= pos < len(df_or_series) else np.nan

        pred_1d   = safe_get(pred_df['Predicted Close price'], end_date_idx + 1)
        pred_5d   = safe_get(pred_df['Predicted Close price'], end_date_idx + 5)
        pred_10d  = safe_get(pred_df['Predicted Close price'], end_date_idx + 10)

        actual_1d = safe_get(Data['Close'], end_date_idx + 1)
        actual_5d = safe_get(Data['Close'], end_date_idx + 5)
        actual_10d= safe_get(Data['Close'], end_date_idx + 10)

        # Current price from the (possibly shifted) end_ts
        current_price = Data['Close'].iloc[end_date_idx]

        def format_metric(pred, actual):
            if pd.isna(pred):
                return "Predicted: N/A", "Actual: N/A"
            if pd.isna(actual):
                return f"Predicted: ${pred:.2f}", "Actual: Future Date"
            err = ((pred - actual) / actual) * 100
            return f"Predicted: ${pred:.2f}", f"Actual: ${actual:.2f} ({err:+.2f}%)"

        p1, a1   = format_metric(pred_1d,  actual_1d)
        p5, a5   = format_metric(pred_5d,  actual_5d)
        p10, a10 = format_metric(pred_10d, actual_10d)

        st.subheader(f"🚀 Prediction from {end_ts.strftime('%Y-%m-%d')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price (End Date)", f"${current_price:.2f}")
        c2.metric("1-Day Ahead",  p1,  a1)
        c3.metric("5-Days Ahead", p5,  a5)
        c4.metric("10-Days Ahead", p10, a10)

        # ---- Visualization ----
        st.subheader("📊 Historical Chart & Future Forecast Points")
        mask = (Data2['Date'] >= pd.to_datetime(start_date)) & (Data2['Date'] <= end_ts)
        display_df = Data2[mask]   # ✅ Fixed: Data2 is the DataFrame, not `pred`

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=display_df['Date'],
            y=display_df['Close'],
            mode='lines',
            name='Historical Close',
            line=dict(color='royalblue', width=2)
        ))

        # Predicted future points — use the index dates so they fall on real trading days
        future_idx   = [end_date_idx + 1, end_date_idx + 5, end_date_idx + 10]
        future_dates = [Data.index[i] for i in future_idx if i < data_len]
        future_preds = [pred_df['Predicted Close price'].iloc[i] for i in future_idx if i < data_len]

        if future_dates:
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
        actual_pairs = [(Data.index[i], Data['Close'].iloc[i])
                        for i in future_idx
                        if 0 <= i < data_len and not pd.isna(Data['Close'].iloc[i])]
        if actual_pairs:
            fig.add_trace(go.Scatter(
                x=[d for d, _ in actual_pairs],
                y=[v for _, v in actual_pairs],
                mode='markers',
                name='Actual',
                marker=dict(color='limegreen', size=10, symbol='circle',
                            line=dict(width=2, color='DarkSlateGrey'))
            ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("☝️ Please upload a CSV file to proceed.")
