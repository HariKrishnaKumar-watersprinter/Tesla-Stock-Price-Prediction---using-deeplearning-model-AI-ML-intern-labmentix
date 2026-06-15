import pandas as pd
from src.data_loader import load_data
from src.feature_engineering import create_features

def data_quality_preprocessing():
    df=create_features()
    df.fillna(df.isnull().sum().mean(), inplace=True)
    df = df.dropna()
    return df

def data_quality_report():
    df = data_quality_preprocessing()
    report=pd.DataFrame()
    report["Column"] = df.columns
    report["Data Type"] = df.dtypes.astype(str).values
    report["Missing Values"] = df.isnull().sum().values
    report["Missing %"] = (df.isnull().sum().values / len(df)) * 100
    report["Unique Values"] = df.nunique().values
    report['duplicates'] = df.duplicated().sum()
    
    return report

def detect_outliers():
    outlier_summary = {}
    df = data_quality_preprocessing()
    num_cols = df.select_dtypes(include=['int64','float64']).columns

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        outlier_summary[col] = len(outliers)

    return pd.DataFrame(outlier_summary,index=['Count']).T