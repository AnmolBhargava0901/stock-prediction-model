import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# ——— Helper: build LSTM model architecture ———
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ——— Load & prepare data ———
DATA_PATH = r'C:\Users\anmol\OneDrive\Desktop\all project\NSE-Tata-Global-Beverages-Limited.csv'
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)

# Extract only closing prices
dataset = df[['Close']].sort_index()
train_df = dataset[:987]
valid_df = dataset[987:]

# Scale training data
scaler = MinMaxScaler((0,1))
scaled_train = scaler.fit_transform(train_df)

# Build training sequences (60 days → predict next)
X_train, y_train = [], []
for i in range(60, len(scaled_train)):
    X_train.append(scaled_train[i-60:i, 0])
    y_train.append(scaled_train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# ——— Train or load model ———
MODEL_FILE = "stock_model.h5"
if not os.path.exists(MODEL_FILE):
    st.sidebar.write("Training new LSTM model (may take a few minutes)…")
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    model.save(MODEL_FILE)
    st.sidebar.success("Model trained and saved")
else:
    model = load_model(MODEL_FILE)
    st.sidebar.info("Loaded existing model")

# ——— Prepare test data & predict ———
inputs = dataset[-(len(valid_df) + 60):].values
inputs = scaler.transform(inputs)
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test).reshape(-1, 60, 1)

preds = model.predict(X_test)
preds = scaler.inverse_transform(preds)

# ——— Display in Streamlit ———
valid_df = valid_df.copy()
valid_df['Predicted'] = preds

st.title("TATA Global Beverages Stock Price Prediction")
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'],
                         name='Training', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Close'],
                         name='Actual', line=dict(color='green')))
fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Predicted'],
                         name='Predicted', line=dict(color='red')))
fig.update_layout(xaxis_title='Date',
                  yaxis_title='Price (INR)',
                  legend_title='')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("**Model:** LSTM")
st.write("**Data:** TATA Global Beverages Limited (NSE)")
st.write("**Note:** Uses past 60 days to predict next closing price.")
st.write("**Disclaimer:** Predictions are for demo only. Always verify independently.")

