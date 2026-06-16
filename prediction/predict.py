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
    uploaded_file = st.file_uploader("Upload your TSLA.csv file", type=["csv",'xlsm'])
    
    if uploaded_file is not None:
        # Load and process data
        Data = pd.read_csv(uploaded_file)
       
        Data['Date'] = pd.to_datetime(Data['Date'], format='%Y-%m-%d')
        Data2 = Data.copy()
        Data.set_index('Date', inplace=True)
        Data.sort_index(inplace=True)
        
        Data['Year'] = Data.index.year
        Data['Daily_Return'] = Data['Close'].pct_change()
        Data['Daily_Range'] = Data['High'] - Data['Low']
        Data['MACD'] = Data['Close'].ewm(span=12, adjust=False).mean() - Data['Close'].ewm(span=26, adjust=False).mean()
        Data['OBV'] = (np.sign(Data['Close'].diff()) * Data['Volume']).cumsum()
        Data[f'Volume_Rolling_Mean_20'] = Data['Volume'].rolling(window=20).mean()
        Data[f'Volume_Rolling_Mean_10'] = Data['Volume'].rolling(window=10).mean()
        Data[f'Volume_Rolling_Mean_5'] = Data['Volume'].rolling(window=5).mean()
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
        
        possible_col=['Open','MACD','OBV','Volume_Rolling_Mean_20','ATR_14','Volatility_20','Year','Volume_Rolling_Mean_10','Close_Open_Ratio','Volatility_5','Price_Rate_Of_Change_20','Volatility_10','Daily_Return','Daily_Range','Volume_Rolling_Mean_5']
        Data1 = Data.reindex(columns=possible_col)
        
        preprocess_pipeline = datascale()[1]
        single = preprocess_pipeline.transform(Data1)
        single_lstm = single.reshape(single.shape[0], single.shape[1], 1)
        
        with st.spinner("Processing data and loading models... Please wait."):
            model_path = os.path.join(os.getcwd(), "model", "LSTM.pkl")
            model = None 
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.success("Model loaded successfully.")
            else:
                st.error(f"Error: Model file not found at: {model_path}. Please ensure 'LSTM.pkl' exists in the 'model' directory.")
                st.stop()
            
        # --- FIX: EXTEND DATA INTO THE FUTURE UP TO 2026-06-12 ---
        target_future_date = pd.to_datetime('2026-06-12')
        last_date_in_data = Data.index.max()
        
        # Generate business days up to the target date
        if last_date_in_data < target_future_date:
            future_dates = pd.bdate_range(start=last_date_in_data + pd.Timedelta(days=1), end=target_future_date)
            
            # Carry forward the last known technical indicators to estimate future features
            last_known = Data1.iloc[-1]
            future_features = pd.DataFrame(index=future_dates, columns=possible_col)
            
            for col in possible_col:
                if col == 'Open':
                    # Assume the stock opens at the previous day's close
                    future_features[col] = Data['Close'].iloc[-1]
                elif col == 'Year':
                    future_features[col] = future_dates.year
                else:
                    # Carry forward the last known technical indicator
                    future_features[col] = last_known[col]
                    
            # Append future features to historical features
            extended_Data1 = pd.concat([Data1, future_features])
            extended_single = preprocess_pipeline.transform(extended_Data1)
            extended_single_lstm = extended_single.reshape(extended_single.shape[0], extended_single.shape[1], 1)
            
            # Predict on the extended dataset
            predict = model.predict(extended_single_lstm)
            
            # Create prediction DataFrame using the extended index
            pred_df = pd.DataFrame(predict.reshape(-1,1).astype(int), 
                                   columns=['Predicted Close price'], 
                                   index=extended_Data1.index)
        else:
            # If the data already goes beyond 2026-06-12, just predict normally
            predict = model.predict(single_lstm)
            pred_df = pd.DataFrame(predict.reshape(-1,1).astype(int), 
                                   columns=['Predicted Close price'], 
                                   index=Data.index)
            
        # --- USER INTERFACE ---
        st.subheader("📅 Select Date Range")
        min_date = Data2['Date'].min().date()
        max_date = target_future_date.date() # Allow date picker to go up to 2026-06-12
        
        col1, col2 = st.columns(2)
        with col1:
           start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
           default_end = datetime.datetime.now().date()
           if default_end > max_date:
               default_end = max_date
           elif default_end < min_date:
               default_end = min_date
           end_date = st.date_input("End Date (Prediction starts from here)", value=default_end, min_value=min_date, max_value=max_date)
           
        if start_date > end_date:
            st.error("❌ Error: End Date must fall after Start Date.")
            st.stop()

        # Find the closest valid trading day on or before the selected end_date
        valid_dates = Data.index[Data.index <= pd.to_datetime(end_date)]
        
        if valid_dates.empty:
            st.warning("⚠️ No trading data available on or before the selected End Date.")
            st.stop()
            
        base_trade_date = valid_dates[-1]
        
        if base_trade_date.date() != end_date:
            st.info(f"ℹ️ Selected date is not a trading day. Using last available trading day: **{base_trade_date.strftime('%Y-%m-%d')}** as base.")

        end_date_idx = pred_df.index.get_loc(base_trade_date)
        data_len = len(pred_df)
        
        # Extract the specific predictions for 1d, 5d, and 10d ahead
        # The prediction at row `i` represents the forecasted price 1 day ahead of features at row `i`.
        # Therefore, the 5-day ahead prediction is located at row `i + 4`.
        pred_1d = pred_df.iloc[end_date_idx + 1]['Predicted Close price'] if end_date_idx + 1 < data_len else np.nan
        pred_5d = pred_df.iloc[end_date_idx + 5]['Predicted Close price'] if end_date_idx + 5 < data_len else np.nan
        pred_10d = pred_df.iloc[end_date_idx + 10]['Predicted Close price'] if end_date_idx + 10 < data_len else np.nan
            
        # Extract the actual prices for comparison
        actual_1d = Data['Close'].iloc[end_date_idx + 1] if end_date_idx + 1 < len(Data) else np.nan
        actual_5d = Data['Close'].iloc[end_date_idx + 5] if end_date_idx + 5 < len(Data) else np.nan
        actual_10d = Data['Close'].iloc[end_date_idx + 10] if end_date_idx + 10 < len(Data) else np.nan
            
        current_price = Data['Close'].iloc[end_date_idx]
        
        def format_metric(pred, actual):
            if pd.isna(pred):
                return "Predicted: N/A", "Actual: N/A"
            if pd.isna(actual):
                return f"Predicted: ${pred:.2f}", f"Actual: Future Date"
            else:
                err = ((pred - actual)/actual)*100
                return f"Predicted: ${pred:.2f}", f"Actual: ${actual:.2f} ({err:+.2f}%)"
                
        p1, a1 = format_metric(pred_1d, actual_1d)
        p5, a5 = format_metric(pred_5d, actual_5d)
        p10, a10 = format_metric(pred_10d, actual_10d)
            
        st.subheader(f"🚀 Prediction from {base_trade_date.strftime('%Y-%m-%d')}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price (End Date)", f"${current_price:.2f}")
        col2.metric("1-Day Ahead", p1, a1)
        col3.metric("5-Days Ahead", p5, a5)
        col4.metric("10-Days Ahead", p10, a10)
                        
        # --- VISUALIZATION ---
        st.subheader("📊 Historical Chart & Future Forecast Points")
        
        # Filter the DataFrame properly for the chart display
        mask = (Data2['Date'] >= pd.to_datetime(start_date)) & (Data2['Date'] <= pd.to_datetime(base_trade_date))
        display_df = Data2[mask]
        
        fig = go.Figure()
            
        # Plot historical close prices for selected range
        fig.add_trace(go.Scatter(
            x=display_df['Date'], 
            y=display_df['Close'], 
            mode='lines', 
            name='Historical Close',
            line=dict(color='royalblue', width=2)
        ))
            
        # Use Business Days (BDay) to skip weekends for future dates
        future_dates = [
            base_trade_date + pd.tseries.offsets.BDay(1), 
            base_trade_date + pd.tseries.offsets.BDay(5), 
            base_trade_date + pd.tseries.offsets.BDay(10)
        ]
        future_preds = [pred_1d, pred_5d, pred_10d]
        
        # Filter out NaN predictions so they don't plot empty markers
        valid_future = [(d, p) for d, p in zip(future_dates, future_preds) if not pd.isna(p)]
        
        if valid_future:
            # Add a connecting line from the last historical point to the first prediction
            fig.add_trace(go.Scatter(
                x=[display_df['Date'].iloc[-1], valid_future[0][0]],
                y=[display_df['Close'].iloc[-1], valid_future[0][1]],
                mode='lines',
                name='Forecast Connection',
                line=dict(color='crimson', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=[d for d,p in valid_future], 
                y=[p for d,p in valid_future], 
                mode='markers+text', 
                name='Predicted',
                marker=dict(color='crimson', size=10, symbol='star'),
                text=[f"${p:.2f}" for d,p in valid_future],
                textposition="top center"
            ))
            
        # If actuals exist, plot them to compare
        actual_dates = [
            base_trade_date + pd.tseries.offsets.BDay(1), 
            base_trade_date + pd.tseries.offsets.BDay(5), 
            base_trade_date + pd.tseries.offsets.BDay(10)
        ]
        actual_vals = [actual_1d, actual_5d, actual_10d]
            
        valid_actuals = [(d, v) for d, v in zip(actual_dates, actual_vals) if not pd.isna(v)]
        if valid_actuals:
            fig.add_trace(go.Scatter(
                x=[d for d,v in valid_actuals], 
                y=[v for d,v in valid_actuals], 
                mode='markers', 
                name='Actual',
                marker=dict(color='limegreen', size=10, symbol='circle', line=dict(width=2, color='DarkSlateGrey'))
            ))
                
        # Explicitly set x-axis range so future prediction markers are not cut off
        all_chart_dates = list(display_df['Date']) + [d for d, p in valid_future] + [d for d, v in valid_actuals]
        if all_chart_dates:
            min_chart_date = min(all_chart_dates) - timedelta(days=1)
            max_chart_date = max(all_chart_dates) + timedelta(days=3) # Padding for text labels
            
            fig.update_layout(
                xaxis_range=[min_chart_date, max_chart_date],
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
        else:
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("☝️ Please upload a CSV file to proceed.")
