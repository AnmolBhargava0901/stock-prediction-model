import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Streamlit page config
st.set_page_config(page_title="TATA Global Beverages Stock Predictor", layout="wide")  

# Constants
MODEL_PATH = r"C:\Users\anmol\OneDrive\Desktop\all project\stock_model.h5"
DATA_PATH = r"C:\Users\anmol\OneDrive\Desktop\all project\NSE-Tata-Global-Beverages-Limited.csv"

# Sidebar info
st.sidebar.title("Configuration")
st.sidebar.write("**Model file:**", MODEL_PATH)
st.sidebar.write("**Data file:**", DATA_PATH)
st.sidebar.markdown("---")

# Load model
@st.cache_resource
def load_trained_model(path):
    return load_model(path)

model = load_trained_model(MODEL_PATH)

# Load and prepare data
@st.cache_data
def load_and_process_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', inplace=True)
    prices = df[['Close']].sort_index()
    return prices

prices = load_and_process_data(DATA_PATH)
train = prices[:987]
valid = prices[987:]

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train)

# Build test sequences
def make_sequences(data, window=60):
    seqs = []
    for i in range(window, len(data)):
        seqs.append(data[i-window:i, 0])
    return np.array(seqs)

# Prepare X_test
combined = np.vstack((scaled_train, np.zeros((len(valid), 1))))  # placeholder for shape
scaled_all = scaler.transform(prices)
X_test = make_sequences(scaled_all[-(len(valid)+60):])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Predict
pred_scaled = model.predict(X_test, verbose=0)
preds = scaler.inverse_transform(pred_scaled)

# Assemble results
valid = valid.copy()
valid['Predicted'] = preds

# Main page
st.title("TATA Global Beverages Stock Price Prediction")
st.markdown("### Actual vs Predicted Closing Prices")

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['Close'],
                         name='Training', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'],
                         name='Actual', line=dict(color='green')))
fig.add_trace(go.Scatter(x=valid.index, y=valid['Predicted'],
                         name='Predicted', line=dict(color='red')))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("**Model:** Pre-trained LSTM loaded from `stock_model.h5`")
st.write("**Dataset:** TATA Global Beverages Limited (NSE)")
st.write("**Window Size:** 60 days")
st.write("**Disclaimer:** For demonstration only. Not financial advice.")

