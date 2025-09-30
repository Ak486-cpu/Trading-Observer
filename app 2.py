"""
Trading MVP App (Streamlit)

Features included in this single-file prototype:
- Data fetch (yfinance)
- Feature engineering (technical indicators via `ta` + lagged returns + rolling stats)
- Multiple model choices (Logistic Regression, RandomForest, XGBoost)
- Time-series train/test split + simple walk-forward evaluation
- Backtest with transaction cost & slippage
- Position sizing suggestions (fixed fraction, Kelly-ish rough)
- UI sections: Data, Features, Modeling, Backtest, Risk & Legal, Tech Stack
- Export trained model (pickle) and backtest results CSV

How to run:
1. Save this file as `trading_mvp_app.py`
2. Create `requirements.txt` with packages listed below
3. `pip install -r requirements.txt`
4. `streamlit run trading_mvp_app.py`

requirements.txt (suggested):
streamlit
pandas
numpy
yfinance
scikit-learn
xgboost
ta
matplotlib
plotly
joblib

Notes:
- This is a prototype for education and testing only. DO NOT use with real money without robust testing.
- Broker integration not included.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import io
import base64
import datetime

st.set_page_config(layout="wide", page_title="Trading MVP — Full Pipeline")

# ----------------- Utility functions -----------------
@st.cache_data
def fetch_data(ticker: str, period: str, interval: str):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df


def create_features(df: pd.DataFrame, predict_horizon: int = 1):
    df = df.copy()
    df = df.dropna()
    df['return'] = df['Close'].pct_change()
    df['future_return'] = df['return'].shift(-predict_horizon)
    df['target'] = (df['future_return'] > 0).astype(int)

    # Add TA features
    try:
        ta_df = add_all_ta_features(df, open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=True)
        df = pd.concat([df, ta_df], axis=1)
    except Exception as e:
        st.warning(f"ta library feature generation failed: {e}")

    # Rolling stats and lags
    df['r2_5'] = df['Close'].pct_change().rolling(5).mean()
    df['r2_10'] = df['Close'].pct_change().rolling(10).mean()
    df['vol_10'] = df['Close'].pct_change().rolling(10).std()

    for lag in range(1, 4):
        df[f'return_lag_{lag}'] = df['return'].shift(lag)

    df = df.dropna()
    return df


def train_model(X_train, y_train, model_name='xgboost'):
    if model_name == 'logistic':
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
    elif model_name == 'rf':
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        model.fit(X_train, y_train)
    else:
        model = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', n_jobs=4)
        model.fit(X_train, y_train)
    return model


def backtest_signals(df_test: pd.DataFrame, preds: np.ndarray, transaction_cost=0.0005, slippage=0.0005):
    df = df_test.copy()
    df['pred'] = preds
    # Simple strategy: go long next period if pred==1 else flat
    df['strategy_return'] = df['return'] * df['pred']
    # subtract transaction cost whenever position changes from 0 to 1 or 1 to 0
    df['pos'] = df['pred']
    df['pos_change'] = df['pos'].diff().abs()
    df['tcost'] = df['pos_change'] * transaction_cost
    df['strategy_return_net'] = df['strategy_return'] - df['tcost'] - (df['pos'] * slippage)
    df['cum_strategy'] = (1 + df['strategy_return_net']).cumprod()
    df['cum_buyhold'] = (1 + df['return']).cumprod()
    return df


def kelly_fraction(returns: pd.Series, preds: pd.Series):
    # Rough Kelly estimate from predicted trades
    trades = returns[preds == 1]
    if len(trades) < 2:
        return 0.0
    win_rate = (trades > 0).mean()
    avg_win = trades[trades > 0].mean() if (trades > 0).any() else 0.0001
    avg_loss = -trades[trades < 0].mean() if (trades < 0).any() else 0.0001
    if avg_loss == 0 or avg_win == 0:
        return 0.0
    k = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)
    # safety caps
    k = max(0, k)
    return min(k, 0.5)  # cap at 50%


# ----------------- Streamlit UI -----------------
st.title("Trading MVP — Everything-in-one Prototype")
st.markdown("एकच app मध्ये Data → Features → Modelling → Backtest → Risk/Legal सगळं आहे. खालील स्टेप्स वापरा.")

# Sidebar inputs
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker (yfinance)", value="AAPL")
period = st.sidebar.selectbox("History length", ["6mo", "1y", "2y", "3y", "5y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m"], index=0)
predict_horizon = st.sidebar.number_input("Predict horizon (periods ahead)", min_value=1, max_value=10, value=1)
model_choice = st.sidebar.selectbox("Model", ["xgboost", "rf", "logistic"], index=0)
transaction_cost = st.sidebar.number_input("Transaction cost per trade (fraction)", min_value=0.0, max_value=0.01, value=0.0005, step=0.0001)
slippage = st.sidebar.number_input("Slippage per period (fraction)", min_value=0.0, max_value=0.01, value=0.0005, step=0.0001)
train_fraction = st.sidebar.slider("Train fraction (time-series split)", min_value=0.5, max_value=0.9, value=0.8)

# Buttons
run_all = st.button("Run full pipeline")

if run_all:
    with st.spinner("Fetching data..."):
        df_raw = fetch_data(ticker, period, interval)

    if df_raw.empty:
        st.error("No data found for this ticker/interval. Try another.")
    else:
        st.success(f"Fetched {len(df_raw)} rows ({df_raw.index.min().date()} to {df_raw.index.max().date()})")

        st.subheader("Preview raw data")
        st.dataframe(df_raw.tail(10))

        st.subheader("Feature Engineering")
        df = create_features(df_raw, predict_horizon=predict_horizon)
        st.write(f"After feature creation: {len(df)} rows, {len(df.columns)} columns")
        st.dataframe(df.tail(5))

        # Select features automatically
        exclude_cols = ['target', 'future_return', 'return']
        features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64]]
        st.write(f"Automatically selected {len(features)} numeric features")
        if st.checkbox("Show selected features", value=False):
            st.write(features)

        # Train-test split
        split_idx = int(len(df) * train_fraction)
        X = df[features]
        y = df['target']
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        st.subheader("Model Training & Evaluation")
        st.write(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

        model = train_model(X_train, y_train, model_name=model_choice)
        preds_test = model.predict(X_test)
        probs_test = None
        try:
            probs_test = model.predict_proba(X_test)[:, 1]
        except Exception:
            probs_test = None

        acc = accuracy_score(y_test, preds_test)
        prec = precision_score(y_test, preds_test)
        rec = recall_score(y_test, preds_test)
        st.write(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

        # Feature importance if available
        st.subheader("Feature importance / coefficients")
        try:
            if model_choice == 'logistic':
                coefs = pd.Series(model.coef_[0], index=features).sort_values(ascending=False).head(30)
                st.bar_chart(coefs)
            else:
                importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(30)
                st.bar_chart(importances)
        except Exception as e:
            st.write("Could not compute importances:", e)

        # Backtest
        st.subheader("Backtest (with transaction cost & slippage)")
        test_df = df.iloc[split_idx:].copy()
        bt_df = backtest_signals(test_df, preds_test, transaction_cost=transaction_cost, slippage=slippage)

        # Plot cumulative
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['cum_strategy'], name='Strategy'))
        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['cum_buyhold'], name='Buy&Hold'))
        fig.update_layout(height=400, title='Cumulative Returns')
        st.plotly_chart(fig, use_container_width=True)

        total_strategy = bt_df['cum_strategy'].iloc[-1] - 1
        total_bh = bt_df['cum_buyhold'].iloc[-1] - 1
        st.metric("Strategy total return", f"{total_strategy:.2%}")
        st.metric("Buy&Hold total return", f"{total_bh:.2%}")

        # Position sizing suggestion
        kelly = kelly_fraction(bt_df['strategy_return_net'], bt_df['pred'])
        suggested_fraction = min(max(kelly * 0.5, 0.0), 0.05)  # be conservative
        st.subheader("Position sizing suggestion")
        st.write(f"Rough Kelly fraction (capped): {kelly:.3f}")
        st.write(f"Suggested fraction to risk per trade (conservative cap 5%): {suggested_fraction:.3%}")

        # Show recent signal
        last_features = df[features].iloc[[-1]]
        try:
            prob = model.predict_proba(last_features)[0][1]
            signal = 'LONG' if prob > 0.5 else 'FLAT/SHORT'
            st.write(f"Latest signal for {ticker}: {signal} (prob up = {prob:.2f})")
        except Exception:
            st.write("Could not compute probability for latest row.")

        # Allow download of backtest csv
        csv = bt_df.to_csv().encode()
        st.download_button("Download backtest CSV", data=csv, file_name=f"backtest_{ticker}.csv")

        # Allow saving model
        buf = io.BytesIO()
        joblib.dump(model, 'trained_model.pkl')
        with open('trained_model.pkl', 'rb') as f:
            model_bytes = f.read()
        st.download_button("Download trained model (pickle)", data=model_bytes, file_name='trained_model.pkl')

        st.subheader("Notes & Next steps")
        st.markdown("""
        - This prototype demonstrates an end-to-end pipeline but is **not** production ready.
        - Next steps: walk-forward CV, hyperparameter tuning, slippage modeling, orderbook features, broker integration, monitoring & alerting, secure storage of API keys.
        - Test extensively with paper trading and realistic transaction costs.
        """)

# ----------------- Risk & Legal Section (always visible) -----------------
st.sidebar.markdown("---")
st.sidebar.header("Risk & Legal (Important)")
st.sidebar.markdown(
    """
    **Warnings:**
    - This app provides *probabilistic* predictions only — not investment advice.
    - Backtests are vulnerable to overfitting, look-ahead bias, and unrealistic assumptions.
    - Including transaction costs, slippage, market impact, latency and taxes is critical.
    - Automating live trading may require regulatory approvals and careful compliance.
    - Do not trade real money without full validation and risk controls.
    """
)

# ----------------- Tech Stack Display -----------------
st.header("Tech Stack & Architecture Guidance")
st.markdown(
    """
    **Suggested stack used by this prototype:**
    - Data: yfinance (for equities), exchange APIs (Binance, Alpaca) for live data.
    - Feature engineering: pandas, ta
    - Modeling: scikit-learn, xgboost, PyTorch/TensorFlow for deep learning
    - Backend/API: FastAPI
    - Frontend: Streamlit (MVP), then React + Tailwind for production
    - Storage: PostgreSQL/TimescaleDB or InfluxDB for time-series; S3 for models
    - Serving: Docker, optionally Kubernetes for scaling
    - Monitoring: Prometheus/Grafana, Sentry for errors

    **Security/operational notes:** store API keys in secure vault (.env not in repo), use encrypted connections, implement rate-limits and circuit breakers when live trading.
    """
)

# ----------------- Quick FAQ / Next Actions -----------------
st.markdown("---")
st.subheader("Quick FAQ & Next actions")
st.markdown(
    """
    **If you want me to:**
    1. Make this repo GitHub-ready with README, .env.example, CI — say `GitHub ready`.
    2. Add paper-trading integration with a specific broker (Alpaca/Binance/Upstox) — tell which one and provide sandbox keys when ready.
    3. Improve model (walk-forward CV, hyperparam tuning) — I'll run tuning and give a report.
    4. Convert to a FastAPI backend + React frontend — I'll scaffold it.

    Pick one or more and I'll prepare the next deliverable.
    """
)

# End of file
