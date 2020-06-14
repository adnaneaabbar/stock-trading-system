import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime,timedelta
from keras.models import load_model
import numpy as np
import matplotlib as plt
from matplotlib.figure import Figure
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = load_model("stock_prediction.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text = request.form['Company']
    scaler = MinMaxScaler(feature_range=(0,1))
    dataframe = web.DataReader(text, data_source='yahoo', start='2012-01-01', end= datetime.today().strftime('%Y-%m-%d'))
    new_df = dataframe.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.fit_transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)[0][0]
    output_close = pred_price

    new_df_open = dataframe.filter(['Open'])
    last_60_days_open = new_df_open[-60:].values
    last_60_days_scaled_open = scaler.fit_transform(last_60_days_open)
    X_test_open = []
    X_test_open.append(last_60_days_scaled_open)
    X_test_open = np.array(X_test_open)
    X_test_open = np.reshape(X_test_open, (X_test_open.shape[0], X_test_open.shape[1], 1))
    pred_price_open = model.predict(X_test_open)
    pred_price_open = scaler.inverse_transform(pred_price_open)[0][0]
    output_open = pred_price_open

    new_df_high = dataframe.filter(['High'])
    last_60_days_high = new_df_high[-60:].values
    last_60_days_scaled_high = scaler.fit_transform(last_60_days_high)
    X_test_high = []
    X_test_high.append(last_60_days_scaled_high)
    X_test_high = np.array(X_test_high)
    X_test_high = np.reshape(X_test_high, (X_test_high.shape[0], X_test_high.shape[1], 1))
    pred_price_high = model.predict(X_test_high)
    pred_price_high = scaler.inverse_transform(pred_price_high)[0][0]
    output_high = pred_price_high

    new_df_low = dataframe.filter(['Low'])
    last_60_days_low = new_df_low[-60:].values
    last_60_days_scaled_low = scaler.fit_transform(last_60_days_low)
    X_test_low = []
    X_test_low.append(last_60_days_scaled_low)
    X_test_low = np.array(X_test_low)
    X_test_low = np.reshape(X_test_low, (X_test_low.shape[0], X_test_low.shape[1], 1))
    pred_price_low = model.predict(X_test_low)
    pred_price_low = scaler.inverse_transform(pred_price_low)[0][0]
    output_low = pred_price_low
    # apple_quote2 = web.DataReader('GS', data_source='yahoo', start=datetime.today().strftime('%Y-%m-%d'), end=datetime.today().strftime('%Y-%m-%d'))
    # today_price=apple_quote2['Close']
    pred_text = 'Predicted '+text+' prices for tomorrow'
    return render_template('index.html', close=output_close,open=output_open,high=output_high,low=output_low,text=pred_text)


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    scaler = MinMaxScaler(feature_range=(0,1))
    dataframe = web.DataReader(data.values(), data_source='yahoo', start='2012-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    new_df = dataframe.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.fit_transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)[0][0]
    output = pred_price
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

