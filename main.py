import asyncio
import io
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message

# Initialize bot
import os
PI_TOKEN = '7023105791:AAH4l7BZ4DessVguUpok0HNeJ88Zxr1Hl0E'  # Note: For production, use os.environ.get('BOT_TOKEN')
bot = Bot(token=API_TOKEN)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

logging.basicConfig(level=logging.INFO)

# Fetch market data
def get_market_data(asset):
    try:
        df = yf.download(asset, period="6mo", interval="1d")
        df.reset_index(inplace=True)
        df.rename(columns={"Close": "price", "Date": "timestamp"}, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        return pd.DataFrame()

# ARIMA Prediction
def predict_with_arima(data):
    if data.empty:
        return None
    model = ARIMA(data['price'], order=(5, 1, 2))
    model_fit = model.fit()
    return model_fit.forecast()[0]

# LSTM Prediction
def predict_with_lstm(data):
    if len(data) < 50:  # Need at least 50 data points
        return None
    
    dataset = data['price'].values.reshape(-1, 1)
    dataset = dataset[-100:]  # Use last 100 points

    # Normalize Data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_scaled = scaler.fit_transform(dataset)

    # Prepare data for LSTM
    X, y = [], []
    for i in range(50, len(dataset_scaled)):
        X.append(dataset_scaled[i-50:i, 0])
        y.append(dataset_scaled[i, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Define LSTM Model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train Model (Quick Training)
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    # Make Prediction
    last_50_days = dataset_scaled[-50:].reshape(1, 50, 1)
    prediction_scaled = model.predict(last_50_days)[0][0]
    
    return scaler.inverse_transform([[prediction_scaled]])[0][0]

# Calculate Indicators
def calculate_indicators(data):
    if data.empty:
        return data

    data['MA50'] = data['price'].rolling(window=50).mean()
    data['MA200'] = data['price'].rolling(window=200).mean()

    # Bollinger Bands
    data['BB_Upper'] = data['MA50'] + 2 * data['price'].rolling(window=50).std()
    data['BB_Lower'] = data['MA50'] - 2 * data['price'].rolling(window=50).std()

    # RSI
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    short_ema = data['price'].ewm(span=12, adjust=False).mean()
    long_ema = data['price'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema

    # Stochastic Oscillator
    data['L14'] = data['price'].rolling(window=14).min()
    data['H14'] = data['price'].rolling(window=14).max()
    data['Stochastic'] = 100 * (data['price'] - data['L14']) / (data['H14'] - data['L14'])

    return data

# Generate Plot
def generate_plot(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data['timestamp'], data['price'], label='Price')
    plt.plot(data['timestamp'], data['MA50'], label='MA50', linestyle='dashed')
    plt.plot(data['timestamp'], data['MA200'], label='MA200', linestyle='dashed')
    plt.fill_between(data['timestamp'], data['BB_Upper'], data['BB_Lower'], color='gray', alpha=0.3, label="Bollinger Bands")
    plt.title('Market Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# Start Command
@dp.message_handler(commands=['start'])
async def start_handler(message: Message):
    await message.answer("✅ Welcome! Use /trade <asset> to get market analysis. Example: /trade EURUSD=X")

# Trade Analysis Command
@dp.message_handler(commands=['trade'])
async def trade_analysis(message: Message):
    try:
        _, asset = message.text.split(" ")
    except ValueError:
        await message.answer("⚠️ Correct format: /trade <asset> (Example: /trade EURUSD=X)")
        return

    await message.answer(f"🔄 Fetching real-time data for {asset}...")
    data = get_market_data(asset)
    if data.empty:
        await message.answer("⚠️ No market data available.")
        return

    data = calculate_indicators(data)
    lstm_prediction = predict_with_lstm(data)
    arima_prediction = predict_with_arima(data)

    plot_buf = generate_plot(data)

    await message.answer_photo(plot_buf, caption=f"📊 Market Analysis for {asset}:\n"
                                                 f"- LSTM AI Prediction: ${lstm_prediction:.2f}\n"
                                                 f"- ARIMA Prediction: ${arima_prediction:.2f}\n"
                                                 f"- RSI: {data['RSI'].iloc[-1]:.2f}\n"
                                                 f"- MACD: {data['MACD'].iloc[-1]:.2f}\n"
                                                 f"- Stochastic: {data['Stochastic'].iloc[-1]:.2f}\n"
                                                 f"- MA50: ${data['MA50'].iloc[-1]:.2f}\n"
                                                 f"- MA200: ${data['MA200'].iloc[-1]:.2f}")

# Main Function
async def main():
    logging.info("🚀 Bot is running...")
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())
