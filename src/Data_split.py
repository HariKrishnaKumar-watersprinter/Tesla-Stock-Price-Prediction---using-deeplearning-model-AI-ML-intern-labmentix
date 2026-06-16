import pandas as pd
from src.data_loader import load_data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.pipeline import Pipeline
def datascale():
    df=load_data()
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

# Create Year and Month columns for aggregation
    df['Daily_Return'] = df['Close'].pct_change()#(Current_Value - Previous_Value) / Previous_Value
    df['Daily_Range'] = df['High'] - df['Low']
    df['Year'] = df.index.year
    df['Month'] = df.index.month_name()
    df['DayOfWeek'] = df.index.dayofweek
    df['Quarter'] = df.index.quarter
    df['DayOfYear'] = df.index.dayofyear
    df['WeekOfYear'] = df.index.isocalendar().week.astype(int)
#  Technical Indicator Features
# Simple Moving Averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()

# Exponential Moving Averages
    for window in [5, 10, 20, 50]:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

# Momentum Indicators
# RSI (Relative Strength Index)
    def compute_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['RSI_7'] = compute_rsi(df['Close'], 7)

# MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

#  Volatility Features
    df['Volatility_5'] = df['Close'].rolling(window=5).std()
    df['Volatility_10'] = df['Close'].rolling(window=10).std()
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + 2 * df['Volatility_20']
    df['BB_Lower'] = df['SMA_20'] - 2 * df['Volatility_20']
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

#  Price-based Features
    df['Price_Rate_Of_Change_5'] = df['Close'].pct_change(5)
    df['Price_Rate_Of_Change_10'] = df['Close'].pct_change(10)
    df['Price_Rate_Of_Change_20'] = df['Close'].pct_change(20)
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']

#  Lag Features
    for lag in [1, 2, 3, 5, 7, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)

#  Rolling Statistics
    for window in [5, 10, 20]:
        df[f'Close_Rolling_Min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'Close_Rolling_Max_{window}'] = df['Close'].rolling(window=window).max()
        df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
# Cyclical encoding for time features
    df['Month'] = pd.to_datetime(df['Month'], format='%B').dt.month
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 5)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 5)

# 5.8 Volume-based Features
    df['Volume_Rate_Of_Change'] = df['Volume'].pct_change()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()

# 5.9 ATR (Average True Range)
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    possible_col=['Open','Volume_Rolling_Mean_20','Year','Volume_Rolling_Mean_10','Daily_Range','Volume_Rolling_Mean_5']
    df=df.reindex(columns=possible_col)
    pipeline_steps = [
               ('imputer', SimpleImputer(strategy='mean')), # Safety for any NaNs
               ('scaler', MinMaxScaler())]
    preprocess_pipeline = Pipeline(pipeline_steps)
    df=preprocess_pipeline.fit_transform(df)
    df1=df.copy()
    
    return df1,preprocess_pipeline
