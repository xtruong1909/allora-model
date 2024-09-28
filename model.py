import os
import requests
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import pickle
from config import data_base_path, model_file_path, training_price_data_path, supported_tokens
from arch import arch_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import time
import logging
from arch.__future__ import reindexing
import random

def get_coingecko_url(token):
    base_url = "https://api.coingecko.com/api/v3/coins/"
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    
    token = token.upper()
    if token in token_map:
        url = f"{base_url}{token_map[token]}/market_chart?vs_currency=usd&days=30&interval=daily"
        return url
    else:
        raise ValueError("Unsupported token")

def download_data():
    os.makedirs(training_price_data_path, exist_ok=True)

    for token in supported_tokens:
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                headers = {
                    "accept": "application/json",
                    "x-cg-demo-api-key": "CG-8MtvACYdTwpB32DpjhLgeeVb"  # replace with your API key
                }
                url = get_coingecko_url(token)
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # Raise an exception for bad responses
                data = response.json()

                # Extract price data
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date')
                df = df.drop('timestamp', axis=1)
                df = df.rename(columns={'price': 'close'})

                # Resample to ensure daily frequency and forward fill any missing data
                df = df.resample('D').ffill()

                # Save to CSV
                output_file = os.path.join(training_price_data_path, f"{token.lower()}usdt_1d.csv")
                df.to_csv(output_file)
                logging.info(f"Downloaded and saved {token} data to {output_file}")

                break  # Exit the retry loop on success

            except requests.exceptions.RequestException as e:
                logging.error(f"Error downloading data for {token} (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:  # If not the last attempt, wait before retrying
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"Failed to download data for {token} after {retries} attempts.")

def resample_data(price_series, timeframe):
    if timeframe == '10m':
        return price_series.resample('10T').interpolate(method='linear')
    elif timeframe == '20m':
        return price_series.resample('20T').interpolate(method='linear')
    elif timeframe == '1d':
        return price_series  # Daily data is already in the correct format
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

def apply_random_adjustment(predicted_value):
    # Lấy thời gian hiện tại theo GMT +7
    current_time = datetime.now().astimezone()
    hour = current_time.hour
    weekday = current_time.weekday()  # Monday is 0, Sunday is 6

    # Kiểm tra khung thời gian
    if (hour >= 19 or hour < 7):  # Giờ từ 19h hôm trước đến 7h sáng hôm sau
        if weekday == 5 or weekday == 6:  # Thứ 7 và Chủ Nhật
            adjustment = random.uniform(-0.0015, 0.0015)  # +/- 0.15%
        else:  # Thứ 2 đến Thứ 6
            adjustment = random.uniform(-0.003, 0.003)  # +/- 0.3%
    else:
        adjustment = random.uniform(-0.0005, 0.0005)  # +/- 0.05%

    # Điều chỉnh giá trị dự đoán
    adjusted_value = predicted_value * (1 + adjustment)
    return adjusted_value

def train_model(token, timeframe):
    os.makedirs(model_file_path, exist_ok=True)
    # Load and preprocess data
    price_data = pd.read_csv(os.path.join(training_price_data_path, f"{token.lower()}usdt_1d.csv"), index_col='date', parse_dates=True)
    price_series = price_data['close']
    price_series = price_series.sort_index().asfreq('D')

    # Adjust for timeframe
    if timeframe == '10m':
        price_series = price_series.resample('10T').interpolate(method='linear')
    elif timeframe == '20m':
        price_series = price_series.resample('20T').interpolate(method='linear')

    # Step 1: Fit ARIMA model
    arima_model = ARIMA(price_series, order=(1,1,1))  # You may need to adjust the order
    arima_results = arima_model.fit()

    # Step 2: Get residuals from ARIMA model
    residuals = arima_results.resid
    
    # Step 3: Fit GARCH model to the residuals
    garch_model = arch_model(residuals, vol='GARCH', p=1, q=1)
    garch_results = garch_model.fit()

    # Create a dictionary to store both models
    combined_model = {
        'arima': arima_results,
        'garch': garch_results
    }

    # Save the combined model
    model_file = f"{model_file_path}_{token.lower()}_{timeframe}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(combined_model, f)

    return combined_model

def predict_price(token, timeframe):
    # Load mô hình đã được huấn luyện
    model_file = f"{model_file_path}_{token.lower()}_{timeframe}.pkl"
    with open(model_file, "rb") as f:
        combined_model = pickle.load(f)

    arima_results = combined_model['arima']
    garch_results = combined_model['garch']
    
    # Giả sử giá trị dự đoán từ mô hình ARIMA
    predicted_value = arima_results.forecast(steps=1)[0]

    # Áp dụng điều chỉnh ngẫu nhiên theo thời gian
    adjusted_value = apply_random_adjustment(predicted_value)
    
    return adjusted_value

if __name__ == "__main__":
    download_data()
    
    # Huấn luyện mô hình (nếu cần)
    train_model('BTC', '1d')

    # Dự đoán giá trị với token và timeframe
    predicted_price = predict_price('BTC', '1d')
    print(f"Predicted Price: {predicted_price}")
