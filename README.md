# 📈 Stock Market Prediction Using Machine Learning

> Predicting closing prices of top-10 S&P 500 equities using classical econometric models and deep learning — enabling data-driven investment insights at scale.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🚀 Business Impact

Financial markets generate billions of data points daily. The ability to accurately forecast stock prices translates directly into:

- **Reduced investment risk** through model-informed entry/exit signals
- **Portfolio optimization** by ranking securities on predicted performance
- **Automated trading support** — LSTM predictions can feed real-time algorithmic trading pipelines
- **Quantifiable alpha generation** — even marginal improvements in prediction accuracy (e.g., 1–2% MAPE reduction) can represent millions in avoided losses at institutional scale
- **Cost savings on research** — automating price forecasting reduces dependency on manual equity analysis

This project demonstrates an end-to-end machine learning pipeline that a quant team or fintech startup could deploy to power trading dashboards, robo-advisors, or risk management systems.

---

## 📌 Project Overview

This project applies a suite of statistical and deep learning models to forecast the **daily closing prices** of 10 major U.S. equities traded on NASDAQ and NYSE, using 5+ years of historical market data (2017–2022).

### Stocks Analyzed

| Ticker | Company              | Sector        |
|--------|----------------------|---------------|
| AAPL   | Apple Inc.           | Technology    |
| MSFT   | Microsoft Corp.      | Technology    |
| AMZN   | Amazon.com Inc.      | Consumer      |
| GOOGL  | Alphabet Inc.        | Technology    |
| TSLA   | Tesla Inc.           | Automotive    |
| NVDA   | NVIDIA Corp.         | Semiconductors|
| META   | Meta Platforms Inc.  | Technology    |
| WMT    | Walmart Inc.         | Retail        |
| TSM    | Taiwan Semiconductor | Semiconductors|
| JNJ    | Johnson & Johnson    | Healthcare    |

---

## 🗂️ Repository Structure
```
Stock-data-science-project/
│
├── Final Project code .ipynb          # End-to-end ML pipeline (EDA → Modeling → Evaluation)
├── PDS Final Project Report.pdf       # Full academic project report
├── Stock Market Prediction Project Report.docx
├── ZAll_Combine_Stock_History.csv     # Consolidated OHLCV dataset (2017–2022)
├── predicted_data.csv                 # Model output — predicted vs. actual prices
│
├── FE plot.png                        # Fixed Effects model visualization
├── FEavvspvplot.png                   # Fixed Effects actual vs. predicted plot
├── FEcoef.png                         # Fixed Effects coefficients
├── LSTM.png                           # LSTM training/prediction plot
├── Linearmodelcoef.png                # Linear Regression coefficients
├── Ridgeplot.png                      # Ridge Regression regularization plot
├── corr.png                           # Correlation heatmap
├── linearplot.png                     # Linear model actual vs. predicted
├── ridgecoef.png                      # Ridge model coefficients
└── README.md
```

---

## 📊 Dataset

- **Source:** Yahoo Finance historical OHLCV data
- **Time Period:** 2017 – 2022 (~5 years)
- **Stocks:** 10 large-cap U.S. equities
- **Features:**
  - `Open`, `High`, `Low`, `Close` (price fields)
  - `Volume` (daily trading volume)
  - `Ticker` (stock identifier)
  - `Date` (time index)
- **Target Variable:** `Close` — daily closing price

---

## 🔬 Methodology

### 1. Data Preparation
- Merged individual stock CSV files into a unified panel dataset
- Handled missing values and ensured date continuity across all tickers
- Applied feature engineering: lag features, rolling statistics, and normalized price scales

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics (mean, std, min/max) across all 10 stocks
- Correlation heatmap revealing inter-stock price relationships
- Time-series visualizations of price trends and volume patterns
- Volatility analysis across sectors (Technology vs. Healthcare vs. Retail)

### 3. Models Implemented

| Model | Type | Key Strength |
|---|---|---|
| **Linear Regression** | Baseline / Statistical | Interpretable coefficients; fast inference |
| **Fixed Effects Model** | Panel Econometrics | Controls for stock-specific unobserved factors |
| **Ridge Regression** | Regularized Linear | Handles multicollinearity in price features |
| **LSTM (Long Short-Term Memory)** | Deep Learning | Captures long-range temporal dependencies in time series |

### 4. Evaluation Metrics
- **RMSE** (Root Mean Squared Error) — penalizes large prediction errors
- **MAE** (Mean Absolute Error) — average prediction deviation in dollars
- **R²** (Coefficient of Determination) — proportion of variance explained

---

## 📈 Key Results

- The **LSTM model** achieved the strongest predictive performance by learning sequential price patterns across multi-day windows, making it most suitable for production forecasting systems
- **Fixed Effects model** provided statistically significant insights into stock-specific pricing dynamics, controlling for individual equity characteristics
- **Ridge Regression** outperformed standard Linear Regression by mitigating multicollinearity among correlated financial features
- All models demonstrated strong R² values on test data, validating the predictive power of historical OHLCV features

> **Business Takeaway:** The LSTM model's superior accuracy positions it as the recommended engine for a real-time stock price prediction microservice, while the Fixed Effects and Ridge models offer faster, interpretable alternatives suitable for regulatory reporting or explainable AI (XAI) use cases.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (LinearRegression, Ridge) |
| **Econometrics** | Statsmodels (Fixed Effects / Panel OLS) |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Environment** | Jupyter Notebook |

---

## ⚡ Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow jupyter
```

### Run the Notebook

```bash
git clone https://github.com/ManojMareedu/Stock-data-science-project.git
cd Stock-data-science-project
jupyter notebook "Final Project code .ipynb"
```

The notebook is self-contained — run all cells sequentially from data loading through model evaluation.

---

## 🔭 Future Enhancements

- [ ] Integrate **sentiment analysis** from financial news (NLP-based feature engineering)
- [ ] Add **transformer-based models** (e.g., Temporal Fusion Transformer) for improved sequence modeling
- [ ] Build a **real-time prediction API** using FastAPI + deployed LSTM model
- [ ] Expand to **100+ stocks** using automated Yahoo Finance data ingestion
- [ ] Implement **portfolio backtesting** framework using predicted signals
- [ ] Deploy interactive dashboard using **Streamlit or Dash**

---

## 👤 Author

**Manoj Mareedu**
- 🎓 M.S. Business Analytics — University of Texas at Dallas
- 💼 [LinkedIn](https://www.linkedin.com/in/manojmareedu)
- 🐙 [GitHub](https://github.com/ManojMareedu)

---

## 📄 License

This project was developed for academic purposes as part of a graduate-level Data Science course. 

---

*If you found this project valuable, please ⭐ star the repository — it helps others discover it!*
