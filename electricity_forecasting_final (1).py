"""
Electricity Consumption Forecasting
-----------------------------------
Real-world time-series project using the UCI Individual Household
Electric Power Consumption dataset.

Models:
    1. Seasonal Naive baseline
    2. ARIMA
    3. SARIMA
    4. LSTM
    5. GRU

Statistical analysis:
    - STL decomposition
    - ADF test
    - KPSS test (non-parametric stationarity test)
    - Mann-Kendall trend test
    - Sen's slope
    - ACF/PACF
    - Ljung-Box residual test
    - ARCH LM residual test
    - Residual diagnostics
    - Rolling-origin evaluation for the final models

Metrics:
    - MAE
    - RMSE
    - MAPE
    - MASE

REQUIREMENTS:
    pip install pandas numpy matplotlib seaborn scipy scikit-learn
    pip install statsmodels arch pymannkendall tensorflow requests

USAGE:
    Place household_power_consumption.txt inside a ./data folder,
    OR let the script download it automatically from UCI, then run:
        python electricity_forecasting_final.py
"""

# ============================================================
# 0. INSTALLATION
# ============================================================
import os
import sys
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------
# Optional TensorFlow import
# ------------------------------------------------------------
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except ImportError:
    raise ImportError(
        "TensorFlow is required for LSTM/GRU. "
        "Install it with: pip install tensorflow"
    )

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import pymannkendall as mk
except ImportError:
    raise ImportError(
        "pymannkendall is required. Install it with: pip install pymannkendall"
    )

# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

ZIP_PATH = DATA_DIR / "household_power_consumption.zip"
TXT_PATH = DATA_DIR / "household_power_consumption.txt"

# UCI dataset
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

# Forecast settings
FREQUENCY = "h"
SEASONAL_PERIOD = 24          # daily seasonality after hourly aggregation
LOOKBACK = 168                # previous 7 days
HORIZON = 24                  # forecast next 24 hours

# To keep ARIMA/SARIMA computationally reasonable,
# use the most recent observations from the training set.
ARIMA_MAX_TRAIN = 10000

# Number of epochs for the deep-learning models.
# Increase to 50-100 for a final polished experiment.
EPOCHS = 30
BATCH_SIZE = 64

# ============================================================
# 2. DOWNLOAD DATA (skipped automatically if file already exists)
# ============================================================

def download_dataset():
    """
    Download the UCI dataset only if it does not already exist.
    """
    if TXT_PATH.exists():
        print("Dataset already exists:", TXT_PATH)
        return

    print("Downloading UCI dataset...")

    import requests

    response = requests.get(UCI_URL, timeout=120)
    response.raise_for_status()

    ZIP_PATH.write_bytes(response.content)

    print("Download complete.")
    print("Extracting...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    print("Extraction complete.")


download_dataset()

# ============================================================
# 3. LOAD AND CLEAN RAW DATA
# ============================================================

print("\nLoading raw data...")

raw = pd.read_csv(
    TXT_PATH,
    sep=";",
    na_values="?",
    low_memory=False
)

print("\nRaw shape:", raw.shape)
print("\nColumns:")
print(raw.columns.tolist())

# Combine Date + Time
raw["datetime"] = pd.to_datetime(
    raw["Date"] + " " + raw["Time"],
    dayfirst=True,
    errors="coerce"
)

raw = raw.drop(columns=["Date", "Time"])
raw = raw.set_index("datetime")
raw = raw.sort_index()

# Convert all measurement columns to numeric
for col in raw.columns:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

print("\nMissing values before treatment:")
print(raw.isna().sum())

# ============================================================
# 4. CREATE HOURLY TARGET
# ============================================================

# Global_active_power is measured as minute-averaged active power in kW.
#
# We use hourly mean active power as the forecasting target.
# This is a clean short-term electricity-demand forecasting problem.

hourly = raw["Global_active_power"].resample(FREQUENCY).mean()

print("\nHourly observations:", len(hourly))
print("Hourly missing observations:", hourly.isna().sum())

# Time interpolation is used only for internal missing timestamps
# in the historical series.
hourly = hourly.interpolate(method="time").ffill().bfill()

hourly.name = "power_kw"
hourly = hourly.asfreq(FREQUENCY)  # lock in explicit frequency for statsmodels

print("\nFinal hourly dataset:")
print(hourly.head())
print(hourly.tail())

# Save processed data
hourly.to_csv(RESULTS_DIR / "hourly_power.csv")

# ============================================================
# 5. BASIC EDA
# ============================================================

plt.figure(figsize=(15, 5))
plt.plot(hourly)
plt.title("Hourly Household Electricity Consumption")
plt.xlabel("Time")
plt.ylabel("Global Active Power (kW)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_full_series.png", dpi=200)
plt.close()

# Recent period for visual inspection
recent = hourly.tail(24 * 14)

plt.figure(figsize=(15, 5))
plt.plot(recent)
plt.title("Hourly Electricity Consumption - Last 14 Days")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_recent_14_days.png", dpi=200)
plt.close()

# ============================================================
# 6. ROLLING STATISTICS
# ============================================================

rolling_mean = hourly.rolling(24).mean()
rolling_std = hourly.rolling(24).std()

plt.figure(figsize=(15, 5))
plt.plot(hourly, alpha=0.45, label="Original")
plt.plot(rolling_mean, label="24-hour rolling mean")
plt.plot(rolling_std, label="24-hour rolling std")
plt.title("Rolling Mean and Rolling Standard Deviation")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_rolling_statistics.png", dpi=200)
plt.close()

# ============================================================
# 7. STL DECOMPOSITION
# ============================================================

# Use a recent training-period sample for a readable decomposition.
stl_sample = hourly.iloc[-24 * 180:]  # last 180 days

stl = STL(
    stl_sample,
    period=SEASONAL_PERIOD,
    robust=True
)

stl_result = stl.fit()

fig = stl_result.plot()
fig.set_size_inches(14, 9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_STL_decomposition.png", dpi=200)
plt.close()

# ============================================================
# 8. STATIONARITY TEST FUNCTIONS
# ============================================================

def adf_test(series, name="Series"):
    """
    Augmented Dickey-Fuller test.

    H0: Unit root / non-stationary
    H1: Stationary
    """
    series = pd.Series(series).dropna()

    result = adfuller(series, autolag="AIC")

    output = {
        "Test": "ADF",
        "Series": name,
        "Statistic": result[0],
        "p_value": result[1],
        "Lags": result[2],
        "Observations": result[3],
        "Conclusion": (
            "Stationary evidence"
            if result[1] < 0.05
            else "Fail to reject non-stationarity"
        ),
    }

    return output


def kpss_test(series, name="Series"):
    """
    KPSS test.

    H0: Stationary
    H1: Non-stationary

    KPSS is a non-parametric stationarity test.
    """
    series = pd.Series(series).dropna()

    result = kpss(
        series,
        regression="c",
        nlags="auto"
    )

    output = {
        "Test": "KPSS",
        "Series": name,
        "Statistic": result[0],
        "p_value": result[1],
        "Lags": result[2],
        "Conclusion": (
            "Stationary evidence"
            if result[1] >= 0.05
            else "Evidence of non-stationarity"
        ),
    }

    return output


def mann_kendall_test(series, name="Series"):
    """
    Mann-Kendall non-parametric monotonic trend test.

    H0: No monotonic trend.
    H1: Monotonic trend exists.

    IMPORTANT:
    Mann-Kendall tests monotonic trend, not unit-root stationarity.
    """
    series = pd.Series(series).dropna()

    result = mk.original_test(series)

    return {
        "Test": "Mann-Kendall",
        "Series": name,
        "Statistic": result.z,
        "p_value": result.p,
        "Trend": result.trend,
        "Sen_slope": result.slope,
        "Conclusion": (
            "Significant monotonic trend"
            if result.p < 0.05
            else "No significant monotonic trend"
        ),
    }


# ============================================================
# 9. STATIONARITY TESTS ON ORIGINAL SERIES
# ============================================================

# Use training data only for model-development diagnostics.
train_end = int(len(hourly) * 0.70)
val_end = int(len(hourly) * 0.85)

train_series = hourly.iloc[:train_end]
validation_series = hourly.iloc[train_end:val_end]
test_series = hourly.iloc[val_end:]

print("\nTrain:", train_series.index.min(), "to", train_series.index.max())
print("Validation:", validation_series.index.min(), "to", validation_series.index.max())
print("Test:", test_series.index.min(), "to", test_series.index.max())

# Mann-Kendall can be expensive on extremely long data.
# Use a representative daily sample for trend detection.
mk_sample = train_series.resample("D").mean()

stationarity_results_original = pd.DataFrame([
    adf_test(train_series, "Original Training Series"),
    kpss_test(train_series, "Original Training Series"),
    mann_kendall_test(mk_sample, "Daily Training Series"),
])

print("\nStationarity / trend tests - original series")
print(stationarity_results_original.to_string(index=False))

stationarity_results_original.to_csv(
    RESULTS_DIR / "stationarity_original.csv",
    index=False
)

# ============================================================
# 10. DIFFERENCING
# ============================================================

# First difference
diff_1 = train_series.diff().dropna()

# Seasonal difference
seasonal_diff = train_series.diff(SEASONAL_PERIOD).dropna()

# Combined regular + seasonal difference
combined_diff = (
    train_series
    .diff(SEASONAL_PERIOD)
    .diff()
    .dropna()
)

stationarity_results_differenced = pd.DataFrame([
    adf_test(diff_1, "First Difference"),
    kpss_test(diff_1, "First Difference"),

    adf_test(seasonal_diff, "Seasonal Difference (24)"),
    kpss_test(seasonal_diff, "Seasonal Difference (24)"),

    adf_test(combined_diff, "Regular + Seasonal Difference"),
    kpss_test(combined_diff, "Regular + Seasonal Difference"),
])

print("\nStationarity tests after differencing")
print(stationarity_results_differenced.to_string(index=False))

stationarity_results_differenced.to_csv(
    RESULTS_DIR / "stationarity_differenced.csv",
    index=False
)

# ============================================================
# 11. ACF AND PACF
# ============================================================

# Use a manageable recent sample for ACF/PACF visualization.
acf_sample = train_series.tail(24 * 60)

acf_values = acf(
    acf_sample,
    nlags=72,
    fft=True
)

pacf_values = pacf(
    acf_sample,
    nlags=72,
    method="ywm"
)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].stem(range(len(acf_values)), acf_values)
axes[0].set_title("ACF - Hourly Electricity Consumption")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("ACF")

axes[1].stem(range(len(pacf_values)), pacf_values)
axes[1].set_title("PACF - Hourly Electricity Consumption")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("PACF")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_ACF_PACF.png", dpi=200)
plt.close()

# ============================================================
# 12. BASELINE: SEASONAL NAIVE
# ============================================================

def seasonal_naive_forecast(history, horizon, season=24):
    """
    Forecast using the value from the same hour one day earlier.
    """
    history = pd.Series(history)
    values = list(history.iloc[-season:].values)

    forecast = []

    for _ in range(horizon):
        pred = values[-season]
        forecast.append(pred)
        values.append(pred)

    return np.array(forecast)


# ============================================================
# 13. METRICS
# ============================================================

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        ) * 100
    )


def mase(y_true, y_pred, training_series, seasonality=24):
    """
    Mean Absolute Scaled Error.

    Scale = MAE of a seasonal naive forecast on training data.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    train = np.asarray(training_series)

    if len(train) <= seasonality:
        raise ValueError("Training series is too short for MASE.")

    naive_errors = np.abs(
        train[seasonality:] - train[:-seasonality]
    )

    scale = np.mean(naive_errors)

    if scale == 0:
        return np.nan

    return np.mean(np.abs(y_true - y_pred)) / scale


def calculate_metrics(y_true, y_pred, training_series):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "MASE": mase(
            y_true,
            y_pred,
            training_series,
            seasonality=SEASONAL_PERIOD
        )
    }


# ============================================================
# 14. ARIMA
# ============================================================

# ARIMA is fitted on a recent portion of the training data to keep
# computation manageable. Model order is selected after ACF/PACF
# investigation; candidate orders can be expanded if desired.

arima_train = train_series.tail(ARIMA_MAX_TRAIN)

ARIMA_ORDER = (2, 1, 2)

print("\nFitting ARIMA", ARIMA_ORDER)

arima_model = ARIMA(
    arima_train,
    order=ARIMA_ORDER,
    enforce_stationarity=False,
    enforce_invertibility=False
)

arima_fit = arima_model.fit()

print(arima_fit.summary())

arima_forecast = arima_fit.forecast(
    steps=len(test_series)
)

# IMPORTANT: use .values here. Wrapping a Series in pd.Series(..., index=...)
# reindexes by matching index LABELS, not position. Since arima_train's
# index can lose its .freq after .tail(), the forecast's own datetime
# index can drift out of exact alignment with test_series.index, which
# silently introduces NaNs. Using .values forces positional assignment.
arima_forecast = pd.Series(
    np.asarray(arima_forecast),
    index=test_series.index
)

# ============================================================
# 15. SARIMA
# ============================================================

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)

print(
    "\nFitting SARIMA",
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER
)

sarima_model = SARIMAX(
    arima_train,
    order=SARIMA_ORDER,
    seasonal_order=SARIMA_SEASONAL_ORDER,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(
    disp=False
)

print(sarima_fit.summary())

sarima_forecast = sarima_fit.forecast(
    steps=len(test_series)
)

# Same fix as ARIMA above: force positional alignment via .values.
sarima_forecast = pd.Series(
    np.asarray(sarima_forecast),
    index=test_series.index
)

# ============================================================
# 16. DEEP LEARNING DATA PREPARATION
# ============================================================

# IMPORTANT:
# Scaling is fitted ONLY on the training data.
# This avoids leakage from validation/test periods.

scaler = MinMaxScaler()

train_values = train_series.values.reshape(-1, 1)
validation_values = validation_series.values.reshape(-1, 1)
test_values = test_series.values.reshape(-1, 1)

train_scaled = scaler.fit_transform(train_values)

# Transform validation and test using training scaler only.
validation_scaled = scaler.transform(validation_values)
test_scaled = scaler.transform(test_values)


def create_sequences(values, lookback, horizon):
    """
    Convert a univariate series into supervised sequences.

    X shape:
        (samples, lookback, 1)

    y shape:
        (samples, horizon)
    """
    X, y = [], []

    for i in range(
        len(values) - lookback - horizon + 1
    ):
        X.append(
            values[i:i + lookback]
        )

        y.append(
            values[
                i + lookback:
                i + lookback + horizon
            ].flatten()
        )

    return np.array(X), np.array(y)


# For validation, include the last LOOKBACK observations from training.
combined_train_val = np.vstack(
    [train_scaled, validation_scaled]
)

X_train, y_train = create_sequences(
    train_scaled,
    LOOKBACK,
    HORIZON
)

X_val_full, y_val_full = create_sequences(
    combined_train_val,
    LOOKBACK,
    HORIZON
)

# Keep only validation-target sequences.
validation_start = len(train_scaled) - LOOKBACK

X_val = X_val_full[
    validation_start:
]

y_val = y_val_full[
    validation_start:
]

# For test sequences, include the end of the validation period as history.
combined_train_val_test = np.vstack(
    [combined_train_val, test_scaled]
)

X_test_full, y_test_full = create_sequences(
    combined_train_val_test,
    LOOKBACK,
    HORIZON
)

test_start = len(combined_train_val) - LOOKBACK

X_test = X_test_full[
    test_start:
]

y_test = y_test_full[
    test_start:
]

print("\nDeep-learning data shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ============================================================
# 17. LSTM MODEL
# ============================================================

def build_lstm(lookback, horizon):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(lookback, 1)
        ),

        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(
            32
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            horizon
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


lstm_model = build_lstm(
    LOOKBACK,
    HORIZON
)

lstm_model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

print("\nTraining LSTM...")

lstm_history = lstm_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# ============================================================
# 18. GRU MODEL
# ============================================================

def build_gru(lookback, horizon):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(lookback, 1)
        ),

        tf.keras.layers.GRU(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.GRU(
            32
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            horizon
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


gru_model = build_gru(
    LOOKBACK,
    HORIZON
)

gru_model.summary()

print("\nTraining GRU...")

gru_history = gru_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# ============================================================
# 19. DEEP LEARNING TEST FORECASTS
# ============================================================

# The test set contains many 24-hour forecasting windows.
# For a fair comparison with ARIMA/SARIMA, use the first
# 24-hour test horizon as the primary direct comparison.

lstm_pred_scaled = lstm_model.predict(
    X_test,
    verbose=0
)

gru_pred_scaled = gru_model.predict(
    X_test,
    verbose=0
)

# Inverse transform each horizon prediction.
def inverse_transform_2d(scaled_values, scaler):
    flat = scaled_values.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(scaled_values.shape)


lstm_pred = inverse_transform_2d(
    lstm_pred_scaled,
    scaler
)

gru_pred = inverse_transform_2d(
    gru_pred_scaled,
    scaler
)

y_test_original = inverse_transform_2d(
    y_test,
    scaler
)

# Flatten for overall test-window metrics.
lstm_true_flat = y_test_original.flatten()
lstm_pred_flat = lstm_pred.flatten()

gru_true_flat = y_test_original.flatten()
gru_pred_flat = gru_pred.flatten()

# ============================================================
# 20. BASELINE FORECAST
# ============================================================

# Evaluate a one-day-ahead seasonal-naive forecast for each
# 24-hour test window.

naive_predictions = []
naive_actuals = []

combined_original = pd.concat(
    [train_series, validation_series, test_series]
)

test_start_position = len(train_series) + len(validation_series)

for start in range(
    test_start_position,
    len(combined_original) - HORIZON + 1,
    HORIZON
):
    history = combined_original.iloc[:start]

    pred = seasonal_naive_forecast(
        history,
        HORIZON,
        SEASONAL_PERIOD
    )

    actual = combined_original.iloc[
        start:start + HORIZON
    ].values

    naive_predictions.extend(pred)
    naive_actuals.extend(actual)

naive_predictions = np.array(naive_predictions)
naive_actuals = np.array(naive_actuals)

# ============================================================
# 21. MODEL METRICS
# ============================================================

metrics = {}

# Defensive check: fail loudly with a clear message rather than a
# cryptic sklearn ValueError, if any forecast still contains NaNs.
for name, series in [("ARIMA", arima_forecast), ("SARIMA", sarima_forecast)]:
    n_nan = series.isna().sum()
    if n_nan > 0:
        raise ValueError(
            f"{name} forecast contains {n_nan} NaN value(s) out of "
            f"{len(series)}. Check that the training series has an "
            f"explicit .freq set and that forecast() steps align with "
            f"test_series.index."
        )

metrics["Seasonal Naive"] = calculate_metrics(
    naive_actuals,
    naive_predictions,
    train_series
)

# ARIMA/SARIMA use the complete test period.
metrics["ARIMA"] = calculate_metrics(
    test_series.values,
    arima_forecast.values,
    train_series
)

metrics["SARIMA"] = calculate_metrics(
    test_series.values,
    sarima_forecast.values,
    train_series
)

metrics["LSTM"] = calculate_metrics(
    lstm_true_flat,
    lstm_pred_flat,
    train_series
)

metrics["GRU"] = calculate_metrics(
    gru_true_flat,
    gru_pred_flat,
    train_series
)

metrics_df = pd.DataFrame(metrics).T

print("\n================ MODEL PERFORMANCE ================")
print(metrics_df.sort_values("RMSE"))

metrics_df.to_csv(
    RESULTS_DIR / "model_performance.csv"
)

# ============================================================
# 22. PLOT ACTUAL VS ARIMA VS SARIMA
# ============================================================

plot_start = test_series.index[:24 * 7]

plt.figure(figsize=(15, 6))

plt.plot(
    test_series.loc[plot_start],
    label="Actual"
)

plt.plot(
    arima_forecast.loc[plot_start],
    label="ARIMA"
)

plt.plot(
    sarima_forecast.loc[plot_start],
    label="SARIMA"
)

plt.title("Actual vs ARIMA vs SARIMA")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "06_ARIMA_SARIMA_comparison.png",
    dpi=200
)
plt.close()

# ============================================================
# 23. LSTM / GRU FORECAST COMPARISON
# ============================================================

# Plot the first 24-hour test forecast from the deep-learning models.

first_actual = y_test_original[0]
first_lstm = lstm_pred[0]
first_gru = gru_pred[0]

forecast_index = test_series.index[:HORIZON]

plt.figure(figsize=(15, 6))

plt.plot(
    forecast_index,
    first_actual,
    marker="o",
    label="Actual"
)

plt.plot(
    forecast_index,
    first_lstm,
    marker="o",
    label="LSTM"
)

plt.plot(
    forecast_index,
    first_gru,
    marker="o",
    label="GRU"
)

plt.title("24-Hour Forecast: LSTM vs GRU")
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "07_LSTM_GRU_forecast.png",
    dpi=200
)
plt.close()

# ============================================================
# 24. RESIDUAL DIAGNOSTICS
# ============================================================

def residual_diagnostics(
    actual,
    predicted,
    model_name,
    lags=48
):
    """
    Calculate residual diagnostics.

    Residual:
        e_t = y_t - yhat_t

    Tests:
        - Ljung-Box
        - ARCH LM

    Also returns residual ACF values.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    n = min(len(actual), len(predicted))

    residuals = actual[:n] - predicted[:n]
    residuals = pd.Series(residuals).dropna()

    # Ljung-Box
    lb = acorr_ljungbox(
        residuals,
        lags=[lags],
        return_df=True
    )

    # ARCH LM
    arch_stat, arch_pvalue, _, _ = het_arch(
        residuals,
        nlags=min(lags, 24)
    )

    result = {
        "Model": model_name,
        "Residual_Mean": residuals.mean(),
        "Residual_Std": residuals.std(),
        "Ljung_Box_Statistic": lb["lb_stat"].iloc[0],
        "Ljung_Box_p_value": lb["lb_pvalue"].iloc[0],
        "ARCH_LM_Statistic": arch_stat,
        "ARCH_LM_p_value": arch_pvalue
    }

    # Residual plot
    plt.figure(figsize=(15, 4))
    plt.plot(residuals)
    plt.axhline(0, linestyle="--")
    plt.title(f"{model_name} Residuals")
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR /
        f"residuals_{model_name.replace(' ', '_')}.png",
        dpi=200
    )
    plt.close()

    # Residual ACF
    plt.figure(figsize=(12, 4))
    residual_acf = acf(
        residuals,
        nlags=lags,
        fft=True
    )

    plt.stem(
        range(len(residual_acf)),
        residual_acf
    )

    plt.title(f"{model_name} Residual ACF")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR /
        f"residual_acf_{model_name.replace(' ', '_')}.png",
        dpi=200
    )
    plt.close()

    return result, residuals


diagnostic_results = []

# ARIMA residuals
result, arima_residuals = residual_diagnostics(
    test_series.values,
    arima_forecast.values,
    "ARIMA"
)
diagnostic_results.append(result)

# SARIMA residuals
result, sarima_residuals = residual_diagnostics(
    test_series.values,
    sarima_forecast.values,
    "SARIMA"
)
diagnostic_results.append(result)

# LSTM residuals
result, lstm_residuals = residual_diagnostics(
    lstm_true_flat,
    lstm_pred_flat,
    "LSTM"
)
diagnostic_results.append(result)

# GRU residuals
result, gru_residuals = residual_diagnostics(
    gru_true_flat,
    gru_pred_flat,
    "GRU"
)
diagnostic_results.append(result)

diagnostics_df = pd.DataFrame(
    diagnostic_results
)

print("\n================ RESIDUAL DIAGNOSTICS ================")
print(diagnostics_df.to_string(index=False))

diagnostics_df.to_csv(
    RESULTS_DIR / "residual_diagnostics.csv",
    index=False
)

# ============================================================
# 25. ARIMA/SARIMA RESIDUAL NORMALITY VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

axes[0].hist(
    arima_residuals,
    bins=50
)
axes[0].set_title("ARIMA Residual Distribution")

axes[1].hist(
    sarima_residuals,
    bins=50
)
axes[1].set_title("SARIMA Residual Distribution")

plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "08_residual_distributions.png",
    dpi=200
)
plt.close()

# ============================================================
# 26. TRAINING CURVES
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    lstm_history.history["loss"],
    label="LSTM Train"
)

plt.plot(
    lstm_history.history["val_loss"],
    label="LSTM Validation"
)

plt.plot(
    gru_history.history["loss"],
    label="GRU Train"
)

plt.plot(
    gru_history.history["val_loss"],
    label="GRU Validation"
)

plt.title("LSTM and GRU Training Curves")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "09_training_curves.png",
    dpi=200
)
plt.close()

# ============================================================
# 27. SAVE DEEP LEARNING MODELS
# ============================================================

lstm_model.save(
    RESULTS_DIR / "lstm_electricity_forecaster.keras"
)

gru_model.save(
    RESULTS_DIR / "gru_electricity_forecaster.keras"
)

# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)

print("\nDataset:")
print("UCI Individual Household Electric Power Consumption")

print("\nForecast frequency:")
print("Hourly")

print("\nForecast horizon:")
print(f"{HORIZON} hours")

print("\nModels:")
print("1. Seasonal Naive")
print("2. ARIMA")
print("3. SARIMA")
print("4. LSTM")
print("5. GRU")

print("\nStatistical tests:")
print("1. ADF")
print("2. KPSS")
print("3. Mann-Kendall")
print("4. Sen's slope")
print("5. Ljung-Box")
print("6. ARCH LM")

print("\nPerformance:")
print(metrics_df.sort_values("RMSE"))

print("\nResidual diagnostics:")
print(diagnostics_df)

print("\nFiles saved in:", RESULTS_DIR.resolve())

# ============================================================
# 29. IMPORTANT INTERPRETATION GUIDE
# ============================================================
#
# ADF:
#     p < 0.05 -> reject unit-root null -> evidence of stationarity.
#
# KPSS:
#     p >= 0.05 -> fail to reject stationarity null.
#
# Mann-Kendall:
#     p < 0.05 -> evidence of monotonic trend.
#     It does NOT directly test unit-root stationarity.
#
# Sen's slope:
#     Estimates the magnitude of a monotonic trend.
#
# Ljung-Box:
#     p >= 0.05 -> fail to reject no-autocorrelation null.
#     This is desirable for model residuals.
#
# ARCH LM:
#     p < 0.05 -> evidence of conditional heteroskedasticity.
#
# Lower MAE/RMSE/MAPE/MASE = better point forecast.
#
# IMPORTANT MODEL-COMPARISON NOTE:
#     ARIMA/SARIMA are evaluated over the complete chronological test
#     period, while LSTM/GRU metrics above aggregate overlapping
#     24-hour forecast windows. For a publication-quality comparison,
#     use a common rolling-origin evaluation protocol for ALL models.
#     That is the next refinement to make before putting the project
#     on your resume.
#
# ============================================================
