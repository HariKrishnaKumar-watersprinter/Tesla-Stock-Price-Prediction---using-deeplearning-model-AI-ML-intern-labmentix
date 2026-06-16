# ⌛ TSLA Quantitative Research Desk | Deep Learning Stock Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

An industry-grade, end-to-end quantitative framework for predicting Tesla (TSLA) stock prices using advanced Deep Learning architectures (SimpleRNN, LSTM, Transformer). This project features a complete pipeline from high-frequency data wrangling and 20-chart interactive EDA to hyperparameter-tuned multi-horizon forecasting and SHAP-based model explainability, all deployed within a professional Streamlit dashboard.

---

## 📑 Table of Contents
- [🎯 Problem Statement](#-problem-statement)
- [✨ Key Features](#-key-features)
- [🛠 Tech Stack](#-tech-stack)
- [📊 Pipeline Architecture](#-pipeline-architecture)
- [🚀 Installation & Setup](#-installation--setup)
- [💻 Running the Dashboard](#-running-the-dashboard)
- [📁 Project Structure](#-project-structure)
- [🔮 Future Scope](#-future-scope)
- [📜 License](#-license)

---

## 🎯 Problem Statement

The stock market, particularly high-growth equities like Tesla (TSLA), is characterized by extreme volatility, non-linear dynamics, and complex temporal dependencies. Traditional statistical models (e.g., ARIMA) often fail to capture these intricate patterns. Furthermore, deep learning models in finance are frequently treated as "black boxes," making it difficult for quantitative analysts to trust and interpret the driving factors behind predictions. 

This project addresses these challenges by developing a robust, multi-horizon deep learning framework capable of capturing sequential dependencies, optimizing hyperparameters systematically via GridSearchCV, and providing explainable, actionable insights using SHAP values.

---

## ✨ Key Features

- **📊 20-Chart Interactive EDA:** Professional Plotly visualizations categorized into *Price & Distribution Dynamics*, *Correlation & Regime Analysis*, and *Multi-Factor Analytics*.
- **🧪 Statistical Hypothesis Testing:** Validates mean returns, volume-price correlation, and distribution normality (Shapiro-Wilk, Pearson, T-Test).
- **⚙️ Advanced Feature Engineering:** Generates 50+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR) and cyclical time features.
- **🧠 Deep Learning Architectures:** Implements SimpleRNN, LSTM, and custom Transformer models for time-series forecasting.
- **🎯 Multi-Horizon Forecasting:** Predicts stock behavior across **1-day, 5-day, and 10-day** windows.
- **🔧 GridSearchCV Integration:** Systematically tunes units, dropout rates, and learning rates using `scikeras` and `TimeSeriesSplit` cross-validation.
- **💡 Model Explainability (XAI):** Uses SHAP (DeepExplainer/KernelExplainer) to quantify feature importance and temporal influence.
- **📅 2026 Iterative Forecasting:** Generates out-of-sample predictions for the year 2026 using sliding window inference.
- **🖥 Professional Streamlit UI:** Terminal-style dashboard with custom CSS, metric cards, and responsive layouts.

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.9+ |
| **Deep Learning** | TensorFlow / Keras, Scikeras |
| **Machine Learning** | Scikit-Learn (GridSearchCV, MinMaxScaler), SHAP |
| **Data Manipulation** | Pandas, NumPy, SciPy |
| **Visualization** | Plotly, Matplotlib |
| **Dashboard** | Streamlit |
| **Serialization** | Joblib, Keras H5 |

---

## 📊 Pipeline Architecture

The project follows a rigorous 13-step quantitative pipeline:

1. **Data Ingestion:** Load OHLCV data.
2. **Data Wrangling:** Handle missing values (ffill/bfill), sort chronologically, remove duplicates.
3. **EDA:** 20 interactive charts (Univariate, Bivariate, Multivariate).
4. **Hypothesis Testing:** 3 statistical tests on returns, correlation, and normality.
5. **Feature Engineering:** Create lag features, rolling stats, momentum indicators, and cyclical encodings.
6. **Outlier Handling:** Winsorization on volatility and volume features.
7. **Feature Selection:** Correlation filtering (>0.95) + Random Forest feature importance.
8. **Stationarity Test:** ADF and KPSS tests; differencing applied if needed.
9. **Data Transformation:** Log transformation for highly skewed features.
10. **Data Scaling:** `MinMaxScaler` applied separately to features and target.
11. **Data Splitting:** Time-series train/val/test split with 60-day sequence generation.
12. **Deep Learning Modeling:** Train SimpleRNN, LSTM, Transformer (Baseline + GridSearchCV).
13. **Explainability & Inference:** SHAP analysis, 2026 forecast generation, Joblib model saving.

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/tsla-quant-predictor.git
   cd tsla-quant-predictor
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

<details>
<summary>📄 View requirements.txt</summary>

```text
streamlit==1.31.0
pandas==2.1.4
numpy==1.23.5
plotly==5.18.0
scipy==1.11.4
scikit-learn==1.3.2
tensorflow==2.15.0
scikeras==0.12.0
shap==0.44.1
joblib==1.3.2
statsmodels==0.14.1
matplotlib==3.8.2
```
</details>

---

## 💻 Running the Dashboard

To launch the interactive Streamlit dashboard, run the following command in your terminal:

```bash
streamlit run app.py
```

*Note: If you do not upload a `TSLA.csv` file, the application will automatically generate mathematical mock data so you can explore the dashboard's functionality.*

---

## 📁 Project Structure

```text
tsla-quant-predictor/
│
├── app.py                     # Main Streamlit application script
├── app_core.py                # Core caching functions & DL model architectures
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── models/                    # Directory for saved deep learning models
│   ├── LSTM_Tuned_h1.h5
│   └── Transformer_Baseline_h5.h5
│
├── scalers/                   # Saved MinMaxScaler objects
│   ├── feature_scaler.pkl
│   └── target_scaler.pkl
│
└── assets/                    # Images/GIFs for README
```

---

## 🔮 Future Scope

- **Sentiment Analysis Integration:** Incorporate NLP models to parse Twitter/X and financial news sentiment as exogenous features.
- **Real-Time API Streaming:** Connect to Alpha Vantage or Yahoo Finance APIs for live tick-level inference.
- **Reinforcement Learning:** Transition from pure prediction to action-based trading environments using RL agents (PPO, DQN).
- **Generative Adversarial Networks (GANs):** Use TimeGAN to synthesize realistic financial time-series for data augmentation.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<p align="center">
  Built with ❤️ and Quantitative Rigor
</p>
