# Financial-Time-Series-Analytics-Platform

An end-to-end financial time-series analytics system for model
comparison, out-of-sample evaluation, and interactive forecasting.

------------------------------------------------------------------------

## 🚀 Quick Start

``` bash
pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```

Open in browser:

http://127.0.0.1:8000

------------------------------------------------------------------------

## 🎥 Demo

### 🎬 Demo Video

[Watch Demo Video](https://youtu.be/WfOaw9U3D-0)

------------------------------------------------------------------------

### 📊 Dashboard Overview


![Dashboard](Financial-Time-Series-Analytics-Platform/datapp/images/dashboard.png)
![alt text](Financial-Time-Series-Analytics-Platform/datapp/images/dashboard1.png)

------------------------------------------------------------------------

### 📈 LSTM Forecast (Test Set)


![LSTM Forecast](Financial-Time-Series-Analytics-Platform/datapp/images/lstm.png)

------------------------------------------------------------------------

### 📉 ARIMA Forecast (Test Set)

![ARIMA Forecast](Financial-Time-Series-Analytics-Platform/datapp/images/arima.png)

![alt text](Financial-Time-Series-Analytics-Platform/datapp/images/home.png)

![alt text](Financial-Time-Series-Analytics-Platform/datapp/images/collect.png)

------------------------------------------------------------------------

## 📌 Project Overview

This project implements a full-stack financial analytics platform for
stock time-series modeling and evaluation using:

-   Deep learning (LSTM)
-   Statistical modeling (ARIMA)

The system emphasizes:

-   Model comparison
-   Out-of-sample testing
-   Performance evaluation
-   Analytical reasoning under financial uncertainty

Rather than attempting to "beat the market," the platform demonstrates
how different modeling paradigms behave under realistic financial
volatility.

------------------------------------------------------------------------

## 🧠 Why This Project Matters

Financial time-series typically exhibit:

-   High volatility
-   Non-stationarity
-   Regime shifts
-   Partial randomness

Short-term forecasting performance is inherently constrained.

This system is designed to:

-   Compare nonlinear vs linear modeling frameworks
-   Evaluate predictive stability
-   Analyze error behavior
-   Understand structural forecasting limits


------------------------------------------------------------------------

## 🏗 System Architecture

Backend: Django\
Database: SQLite\
Data Processing: Pandas / NumPy\
Deep Learning: TensorFlow (LSTM)\
Statistical Modeling: ARIMA (Statsmodels + auto_arima)\
Visualization: pyecharts

### Workflow

Data Collection\
→ Data Cleaning\
→ Train/Test Split\
→ Model Training\
→ Out-of-Sample Forecast\
→ Performance Evaluation\
→ Interactive Dashboard Display

------------------------------------------------------------------------

## 📦 Project Structure

Financial-Time-Series-Analytics-Platform/

├── datapp/ \# Core modeling logic (LSTM, ARIMA, evaluation)\
├── templates/ \# Frontend templates\
├── static/ \# Static assets\
├── images/ \# README screenshots\
├── manage.py\
├── requirements.txt\
└── README.md

------------------------------------------------------------------------

## 📊 Modeling Components

### LSTM (Deep Learning Model)

-   3-layer architecture
-   Sliding-window sequence modeling
-   Nonlinear temporal dependency capture
-   Stochastic training behavior

Captures nonlinear patterns but exhibits stochastic variability across
runs.

------------------------------------------------------------------------

### ARIMA (Statistical Benchmark)

-   Stationarity testing (ADF)
-   Automatic order selection (auto_arima)
-   Confidence interval estimation
-   Linear time-series baseline

Provides a structured benchmark for comparison with nonlinear models.

------------------------------------------------------------------------

## 📈 Evaluation Framework

Out-of-sample (test set) evaluation using:

-   MAE (Mean Absolute Error)
-   RMSE (Root Mean Squared Error)
-   R² (Coefficient of Determination)
-   MAPE (Mean Absolute Percentage Error)

Evaluation metrics vary depending on:

-   Selected stock
-   Time horizon
-   Market regime
-   Model initialization (LSTM stochasticity)

Negative R² values may occur in volatile regimes, reflecting structural
prediction limits rather than implementation errors.

------------------------------------------------------------------------

## 🔍 Analytical Insights

-   Deep learning models adapt better to nonlinear financial dynamics.
-   ARIMA provides a stable statistical baseline.
-   Financial forecasting accuracy is fundamentally bounded by market
    randomness.
-   Model comparison reveals trade-offs between flexibility and
    stability.


------------------------------------------------------------------------

## ⚠️ Reproducibility Note

Due to stochastic weight initialization and training randomness, LSTM
evaluation metrics may vary slightly across runs.

This reflects realistic model behavior rather than instability in
implementation.

------------------------------------------------------------------------

## 📉 Limitations

-   Financial markets approximate random-walk behavior.
-   Short-term forecasting remains unstable.
-   No transaction cost modeling included.
-   No portfolio-level risk adjustment implemented.
-   Single-asset time-series modeling only.

------------------------------------------------------------------------

## 🔮 Future Enhancements

-   GRU / Transformer comparison
-   Walk-forward validation
-   Hyperparameter optimization
-   Portfolio-level modeling
-   Risk-adjusted performance metrics

------------------------------------------------------------------------

## 🛠 Skills Demonstrated

-   Time-series analysis
-   Statistical modeling
-   Deep learning implementation
-   Financial data reasoning
-   Model evaluation and comparison
-   Backend development (Django)
-   Data pipeline design
-   Analytical interpretation under uncertainty

------------------------------------------------------------------------
