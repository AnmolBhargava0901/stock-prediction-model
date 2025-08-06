import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Load dataset (use raw string for file path to avoid escape issues)
df = pd.read_csv(r'C:\Users\anmol\OneDrive\Desktop\all project\NSE-Tata-Global-Beverages-Limited.csv')
df['Date'] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.index = df['Date']

# Prepare data
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

# Extract 'Date' and 'Close' for model
new_dataset = data[['Date', 'Close']].reset_index(drop=True)
new_dataset.index = new_dataset['Date']
new_dataset.drop('Date', axis=1, inplace=True)

# Split into training and validation
train_data = new_dataset[:987]
valid_data = new_dataset[987:]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train_data)



# Create training sequences
x_train = []
y_train = []

for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Load trained LSTM model
model_new = load_model("stock_model.h5")

# Prepare test data
inputs = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(60, len(inputs)):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
p_closing_price = model_new.predict(x_test)
p_closing_price = scaler.inverse_transform(p_closing_price)

# Visualization
train = new_dataset[:987]
valid = new_dataset[987:].copy()
valid['Predictions'] = p_closing_price

st.subheader('Stock Price Prediction')
fig = go.Figure()

# Plot training data
fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Training Data', line=dict(color='blue')))
# Plot actual validation data
fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual Stock Price', line=dict(color='green')))
# Plot predicted stock price
fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted Stock Price', line=dict(color='red')))

fig.update_layout(title='TATA Global Beverages Limited Stock Price Prediction',
                  xaxis_title='Date', yaxis_title='Stock Price (INR)')
st.plotly_chart(fig, use_container_width=True)

# Info panel
st.markdown("---")
st.write("Model : LSTM (Long Short-Term Memory)")
st.write("Dataset : TATA Global Beverages Limited Stock Price")
st.write("Note: this is a simple example using 60 days of past data to predict the next stock price. For better accuracy, consider using more complex models and additional features.")
st.markdown("For a more accurate prediction, consider using more complex models, additional features, and extensive datasets.")
st.write("Disclaimer: This is a demo application and the predictions made by the model are not guaranteed to be accurate. Always do your own research before making any investment decisions.")
st.markdown("---")
