import pandas as pd
import os

def load_data():
    # Direct raw GitHub URL (most reliable for Streamlit)
    url = "https://raw.githubusercontent.com/HariKrishnaKumar-watersprinter/Tesla-Stock-Price-Prediction---using-deeplearning-model-AI-ML-intern-labmentix/main/data/TSLA.csv"
    
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None
