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
        Data2=Data.copy()
        Data.set_index('Date', inplace=True)
        Data.sort_index(inplace=True)
        Data['Year'] = Data.index.year
        Data['Daily_Return'] = Data['Close'].pct_change()#(Current_Value - Previous_Value) / Previous_Value
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
        preprocess_pipeline=datascale()[1]
        single=preprocess_pipeline.transform(Data1)
        single_lstm = single.reshape(single.shape[0],single.shape[1],1)
        with st.spinner("Processing data and loading models... Please wait."):
            model_path = os.path.join(os.getcwd(), "model", "LSTM.pkl")
            model = None # Initialize model to None
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.success("Model loaded successfully.")
            else:
                st.error(f"Error: Model file not found at: {model_path}. Please ensure 'LSTM.pkl' exists in the 'model' directory.")
                return # Stop execution if model is not found
            
        predict=model.predict(single_lstm ) # Changed from 'single' to 'single_lstm'
        
        
        st.subheader("📅 Select Date Range")
        min_date = Data2['Date'].min().date()
        max_date = Data2['Date'].max().date()
        col1, col2 = st.columns(2)
        with col1:
           start_date = st.date_input("Start Date", value=min_date , min_value=min_date, max_value=max_date)
        with col2:
           # FIX: Default end_date is set 15 days BEFORE max_date to guarantee 10 future trading days exist in the dataset
           default_end =  start_date + timedelta(days=30)
           if default_end < min_date:
               default_end = min_date
           end_date = st.date_input("End Date (Prediction starts from here)", value=default_end, min_value=min_date, max_value=max_date)
        if start_date > end_date:
            st.error("❌ Error: End Date must fall after Start Date.")
        else:
        # Filter data for chart display
            mask = (Data2['Date'] >= pd.to_datetime(start_date)) & (Data2['Date'] <= pd.to_datetime(end_date))
            display_df = Data2[mask]
        
        
        # Get the row for the selected END DATE to make predictions
        selected_end_row = Data2[Data2['Date'] == pd.to_datetime(end_date)]
        if selected_end_row.empty:
            st.warning("⚠️ The selected End Date is not a trading day (e.g., weekend or holiday). Please select a valid trading day.")
        else:
             # 1. Create a DataFrame for all 1-day predictions
            pred_df = pd.DataFrame(predict.reshape(-1,1).astype(int), 
                                   columns=['Predicted Close price'], 
                                   index=Data.index)
            
            # 2. Get the index position of the selected end date
            # We use Data.index because that's what we set as the index for pred_df
            if pd.to_datetime(end_date) not in Data.index:
                st.warning("⚠️ The selected End Date is not available in the processed dataset (might have been dropped due to NaNs).")
                return

            end_date_idx = Data.index.get_loc(pd.to_datetime(end_date))
            data_len = len(Data)
            # 3. Extract the specific predictions for 1d, 5d, and 10d ahead
            pred_1d = pred_df.iloc[end_date_idx + 1]['Predicted Close price'] if end_date_idx + 1 < data_len else np.nan
            pred_5d = pred_df.iloc[end_date_idx + 5]['Predicted Close price'] if end_date_idx + 5 < data_len else np.nan
            pred_10d = pred_df.iloc[end_date_idx + 10]['Predicted Close price'] if end_date_idx + 10 < data_len else np.nan
            
            # 4. Extract the actual prices for comparison
            # Using min() to prevent IndexError if we are near the end of the dataset
            actual_1d = Data['Close'].iloc[end_date_idx + 1] if end_date_idx + 1 < len(Data) else np.nan
            actual_5d = Data['Close'].iloc[end_date_idx + 5] if end_date_idx + 5 < len(Data) else np.nan
            actual_10d = Data['Close'].iloc[end_date_idx + 10] if end_date_idx + 10 < len(Data) else np.nan
            
            current_price = selected_end_row['Close'].values[0]
            def format_metric(pred, actual):
                if pd.isna(actual):
                    return f"Predicted: ${pred:.2f}", f"Actual: Future Date"
                else:
                    err = ((pred - actual)/actual)*100
                    return f"Predicted: ${pred:.2f}", f"Actual: ${actual:.2f} ({err:+.2f}%)"
            p1, a1 = format_metric(pred_1d, actual_1d)
            p5, a5 = format_metric(pred_5d, actual_5d)
            p10, a10 = format_metric(pred_10d, actual_10d)
            
            st.subheader("🚀 Future Price Prediction")
            st.subheader(f"🚀 Prediction from {end_date.strftime('%Y-%m-%d')}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price (End Date)", f"${current_price:.2f}")
            col2.metric("1-Day Ahead", p1, a1)
            col3.metric("5-Days Ahead", p5, a5)
            col4.metric("10-Days Ahead", p10, a10)
                        # --- VISUALIZATION ---
            st.subheader("📊 Historical Chart & Future Forecast Points")
            
            fig = go.Figure()
            
            # Plot historical close prices for selected range
            fig.add_trace(go.Scatter(
                x=display_df['Date'], 
                y=display_df['Close'], 
                mode='lines', 
                name='Historical Close',
                line=dict(color='royalblue', width=2)
            ))
            
            # Plot prediction markers
            future_dates = [
                pd.to_datetime(end_date) + pd.Timedelta(days=1), 
                pd.to_datetime(end_date) + pd.Timedelta(days=5), 
                pd.to_datetime(end_date) + pd.Timedelta(days=10)
            ]
            future_preds = [pred_1d, pred_5d, pred_10d]
            
            fig.add_trace(go.Scatter(
                x=future_dates, 
                y=future_preds, 
                mode='markers+text', 
                name='Predicted',
                marker=dict(color='crimson', size=10, symbol='star'),
                text=[f"1d: ${p:.2f}" for p in future_preds],
                textposition="top center"
            ))
            
            # If actuals exist, plot them to compare
            actual_dates = [
                pd.to_datetime(end_date) + pd.Timedelta(days=1), 
                pd.to_datetime(end_date) + pd.Timedelta(days=5), 
                pd.to_datetime(end_date) + pd.Timedelta(days=10)
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
                
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, width='stretch')

    else:
        st.info("☝️ Please upload a CSV file to proceed.")
            
       
       
