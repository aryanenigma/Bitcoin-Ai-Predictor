# Bitcoin AI Predictor – Data Analytics Case Study

## Overview

This project is **not** a trading bot or a guaranteed profit system. It is a **data analytics and applied machine learning case study** built to answer a specific question:

> **Can historical market data and engineered features provide statistically meaningful signals for short‑term Bitcoin price movement?**

The goal of this project is to demonstrate my ability as a **data analytics fresher** to:

* Collect and clean real‑world financial data
* Perform exploratory data analysis (EDA)
* Engineer relevant features
* Build and evaluate predictive models
* Interpret results honestly, including limitations

---

## Problem Statement

Cryptocurrency prices are highly volatile and often close to random in the short term. Many "AI trading" projects ignore this and overclaim results.

This project takes a **realistic analytics approach**:

* Use historical Bitcoin OHLCV data
* Predict **short‑term price direction** (up/down), not exact prices
* Benchmark models against simple baselines
* Evaluate whether performance exceeds random guessing

---

## Data Source

* **Asset:** Bitcoin (BTC)
* **Data Type:** OHLCV (Open, High, Low, Close, Volume)
* **Frequency:** Daily (can be adapted to hourly)
* **Source:** Public cryptocurrency market API

### Data Cleaning Steps

* Removed missing and inconsistent records
* Converted timestamps to proper datetime format
* Ensured numeric consistency across price and volume columns
* Verified continuity of time series

---

## Exploratory Data Analysis (EDA)

Key analyses performed:

* Price trend visualization over time
* Daily returns distribution
* Volatility analysis
* Correlation matrix of engineered features
* Volume vs price‑movement relationships

### Key Observations

* Returns are heavily skewed with fat tails (non‑normal distribution)
* High volatility periods cluster together
* Volume alone is not a reliable predictor of next‑day direction
* Strong autocorrelation is rare, confirming market efficiency

---

## Feature Engineering

The following features were created from raw price data:

* Daily returns
* Rolling moving averages (short & long windows)
* Rolling volatility
* Momentum indicators
* Lagged price and return values

All features were scaled where required and aligned to prevent data leakage.

---

## Modeling Approach

This is a **classification problem** (price up vs down).

### Models Used

* Baseline model (random / naive predictor)
* Logistic Regression
* Tree‑based models (where applicable)

### Evaluation Metrics

* Accuracy
* Precision & Recall
* Confusion Matrix
* Cross‑validation performance

Models were evaluated on **out‑of‑sample data** to avoid overfitting.

---

## Results & Interpretation

* Models achieved **slightly better than random accuracy** in some periods
* Performance was **not stable across all market regimes**
* Feature importance showed momentum‑based features had limited but non‑zero impact

### Critical Insight

> The model does **not** consistently outperform a naive baseline.

This confirms an important real‑world analytics lesson:
**Most financial time series are extremely hard to predict, and honest evaluation matters more than flashy claims.**

---

## Limitations

* Market behavior changes over time (concept drift)
* No transaction cost or slippage modeling
* Short‑term prediction in crypto is near‑efficient
* Results should not be used for real trading without deeper risk controls

---

## What This Project Demonstrates

✔ Real‑world data cleaning and preprocessing
✔ Exploratory data analysis with financial time series
✔ Feature engineering from raw data
✔ Model benchmarking and evaluation
✔ Honest interpretation of results

This project is intended as a **data analytics portfolio piece**, not a trading promise.

---

## How to Run

1. Clone the repository
2. Install required Python libraries
3. Run data preparation scripts
4. Execute analysis and modeling notebooks

---

## Future Improvements

* Add more robust backtesting
* Include macro or sentiment data
* Regime‑based modeling
* Improved visualization dashboards

---

## Author

**Aryan Kaushik**
BCA Student | Aspiring Data Analyst
GitHub: [https://github.com/aryanenigma](https://github.com/aryanenigma)

---

## Disclaimer

This project is for **educational and analytical purposes only**. It does not provide financial advice or guaranteed trading strategies.
