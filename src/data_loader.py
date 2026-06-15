import pandas as pd
import os

def load_data():
    os.chdir('F:\\Project\Labmantix\\tesla stock price prediction\\streamlit')
    path = os.path.join(os.getcwd(), "data", "TSLA.csv")
    
    df = pd.read_csv(path)
    
    return df