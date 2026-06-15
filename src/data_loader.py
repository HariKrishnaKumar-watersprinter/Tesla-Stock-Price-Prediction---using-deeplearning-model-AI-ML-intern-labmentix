import pandas as pd
import os

def load_data():
    #os.chdir(r'https://raw.github.com/HariKrishnaKumar-watersprinter/Tesla-Stock-Price-Prediction---using-deeplearning-model-AI-ML-intern-labmentix/main')
    path = os.path.join(os.getcwd(), "data", "TSLA.csv")
    
    df = pd.read_csv(path)
    
    return df
