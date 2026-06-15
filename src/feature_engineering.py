import pandas as pd
from src.data_loader import load_data

def create_features(df):
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
    
    df1=df.copy()
    
    return df1