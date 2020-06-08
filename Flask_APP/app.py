import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime,timedelta
from keras.models import load_model
import numpy as np
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
    output = pred_price

    return render_template('index.html', prediction_text='tomorrow`s stock price is {} $'.format(output))

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

