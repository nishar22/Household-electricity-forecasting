# Household Electricity Demand Forecasting

A time-series forecasting pipeline that compares a seasonal-naive baseline, ARIMA, SARIMA, LSTM, and GRU models on real-world household electricity consumption data, with full statistical diagnostics (stationarity testing, ACF/PACF, residual analysis).

📄 **[Full Project Report (PDF)](./Electricity_Forecasting_Project_Report.pdf)** — detailed writeup with all figures and interpretation.

---

## Overview

This project builds an end-to-end forecasting pipeline on the [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) dataset — 2M+ minute-level readings from a single household in Sceaux, France (Dec 2006 – Nov 2010). The goal: forecast hourly `Global_active_power` (kW) 24 hours ahead, and rigorously compare five modeling approaches using both accuracy metrics and residual diagnostics.

## Dataset

| | |
|---|---|
| Source | UCI Machine Learning Repository |
| Raw size | 2,075,259 minute-level rows, 9 columns |
| Missing data | ~1.25% per measurement column, handled via time-interpolation |
| Resampled to | Hourly mean (34,589 observations) |
| Split | 70% train / 15% validation / 15% test (chronological, no shuffling) |

## Methodology

1. **Data cleaning** — datetime indexing, numeric coercion, hourly resampling, gap interpolation
2. **EDA** — full series, recent-window, and rolling statistics plots
3. **STL decomposition** — trend/seasonal/residual breakdown
4. **Stationarity & trend testing** — ADF, KPSS, Mann-Kendall + Sen's slope, on both raw and differenced series
5. **ACF/PACF analysis** — to justify model order selection
6. **Model fitting** — Seasonal Naive, ARIMA(2,1,2), SARIMA(1,1,1)(1,1,1,24), LSTM, GRU
7. **Evaluation** — MAE, RMSE, MAPE, MASE
8. **Residual diagnostics** — Ljung-Box (autocorrelation), ARCH-LM (heteroskedasticity)

## Results

| Model | MAE | RMSE | MAPE (%) | MASE |
|---|---|---|---|---|
| **LSTM** | **0.435** | **0.592** | 62.94 | **0.658** |
| GRU | 0.480 | 0.625 | 75.25 | 0.726 |
| ARIMA | 0.622 | 0.725 | 116.52 | 0.940 |
| Seasonal Naive | 0.505 | 0.753 | 65.67 | 0.763 |
| SARIMA | 1.635 | 1.886 | 331.27 | 2.471 |

LSTM achieved the best performance on every metric. See the [full report](./Electricity_Forecasting_Project_Report.pdf) for residual diagnostics, STL decomposition, ACF/PACF justification for model orders, and an important caveat on evaluation-protocol differences between the statistical and deep learning models (Report Section 8: Limitations).

## Tech Stack

- **Data:** pandas, numpy
- **Statistical testing:** statsmodels (ADF, KPSS, STL, Ljung-Box, ARCH-LM), pymannkendall
- **Classical forecasting:** statsmodels (ARIMA, SARIMAX)
- **Deep learning:** TensorFlow / Keras (LSTM, GRU)
- **Evaluation:** scikit-learn
- **Visualization:** matplotlib, seaborn

## Repository Structure

```
├── results/                                  # generated outputs (figures, CSVs, saved models)
├── Electricity_Forecasting_Project_Report.pdf  # full written report
├── electricity_forecasting_final.py          # main script (data pipeline + all models)
└── README.md
```

## Setup & Usage

```bash
git clone https://github.com/nishar22/Household-electricity-forecasting.git
cd Household-electricity-forecasting

pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install statsmodels pymannkendall tensorflow requests

python electricity_forecasting_final.py
```

The dataset auto-downloads from UCI on first run (or place `household_power_consumption.txt` in a `data/` folder manually before running). Note: fitting ARIMA/SARIMA and training the LSTM/GRU models on the full dataset takes significant time — see script comments for parameters (`ARIMA_MAX_TRAIN`, `EPOCHS`) that can be reduced for faster experimentation.

## Key Findings

- The series is **trend-stationary** with a statistically significant decreasing trend (Mann-Kendall p < 0.001); first/seasonal differencing achieves full stationarity per both ADF and KPSS.
- Strong **24-hour seasonality** is confirmed by ACF peaks at lags 24/48/72, motivating the SARIMA seasonal order.
- **LSTM outperformed all other models** on every metric, with GRU close behind using ~30% fewer parameters.
- **SARIMA underperformed even the naive baseline** in this setup — investigated and explained in the report as an artifact of unconditional long-horizon forecasting rather than a fundamentally weaker model.
- All models show significant residual autocorrelation (Ljung-Box) and volatility clustering (ARCH-LM), indicating room for further refinement (e.g. GARCH-based variance modeling, quantile loss for spike sensitivity).

## Limitations & Future Work

- ARIMA/SARIMA were evaluated via a single unconditional forecast over the full test period, while LSTM/GRU used rolling 24-hour windows with fresh history each time — not a fully matched comparison. A common **rolling-origin (walk-forward) evaluation** is the natural next step.
- No prediction intervals were produced; given confirmed heteroskedasticity, a GARCH-type or quantile-based approach would be needed for calibrated intervals.
- MAPE is unreliable on this data due to near-zero actual values in low-draw hours; MAE/RMSE/MASE are the more trustworthy metrics here.

## License

This project uses the UCI Individual Household Electric Power Consumption dataset, available under UCI's public dataset terms. Code in this repository is available under the MIT License.
