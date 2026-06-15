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
        st.subheader("📅 Select Date Range")
        Data['Date'] = pd.to_datetime(Data['Date'], format='%Y-%m-%d')
        min_date = Data['Date'].min().date()
        max_date = Data['Date'].max().date()
        col1, col2 = st.columns(2)
        with col1:
        # Default start date: 30 days before end date
           start_date = st.date_input("Start Date", value=max_date - timedelta(days=30), min_value=min_date, max_value=max_date)
        with col2:
        # Prediction will be made FROM this end date
           end_date = st.date_input("End Date (Prediction starts from here)", value=max_date, min_value=min_date, max_value=max_date)
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

        model_path = os.path.join(os.getcwd(), "model", "LSTM.pkl")
        model = None # Initialize model to None
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.success("Model loaded successfully.")
        else:
            st.error(f"Error: Model file not found at: {model_path}. Please ensure 'LSTM.pkl' exists in the 'model' directory.")
            return # Stop execution if model is not found
            
        predict=model.predict(single_lstm ) # Changed from 'single' to 'single_lstm'
        
        predict=pd.DataFrame(predict.reshape(-1,1).astype(int), columns=['Predicted Close price'], index=Data.index)
        pred_5d = predict['Predicted Close price'].shift(-5)
        pred_5d = pred_5d.dropna()
        pred_10d = predict['Predicted Close price'].shift(-10)
        pred_10d = pred_10d.dropna()
        actual_1d = Data['Close'].shift(-1)
        actual_1d = actual_1d.dropna()
        actual_5d = Data['Close'].shift(-5)
        actual_5d = actual_5d.dropna()  
        actual_10d = Data['Close'].shift(-10)
        actual_10d = actual_10d.dropna()
        def format_metric(pred, actual):
                if pd.isna(actual):
                    return f"Predicted: ${pred:.2f}", f"Actual: Future Date"
                else:
                    err = ((pred - actual)/actual)*100
                    return f"Predicted: ${pred:.2f}", f"Actual: ${actual:.2f} ({err:+.2f}%)"
        p1, a1 = format_metric(pred_1d, actual_1d)
        p5, a5 = format_metric(pred_5d, actual_5d)
        p10, a10 = format_metric(pred_10d, actual_10d)
        
            
        with st.spinner("Processing data and loading models... Please wait."):
            st.subheader("🚀 Future Price Prediction")
            st.subheader(f"🚀 Prediction from {end_date.strftime('%Y-%m-%d')}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price (End Date)", f"${current_price:.2f}")
            col2.metric("1-Day Ahead", p1, a1)
            col3.metric("5-Days Ahead", p5, a5)
            col4.metric("10-Days Ahead", p10, a10)
            st.subheader("🚀 Future Price Prediction")
       
       
