import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt

df = web.DataReader('GS', data_source='yahoo', start='2012-01-01', end='2019-12-17' )

data = df.filter(['Close'])
dataset = data.values

#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

from tqdm import tqdm
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in tqdm(range(60, len(train_data))):
 x_train.append(train_data[i-60:i, 0])
 y_train.append(train_data[i, 0])
 if i<=61 :
  print(x_train)
  print(y_train)

x_train, y_train = np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))

model.add(Dense(units = 1024))
model.add(Dense(units = 512))
model.add(Dense(units = 256))
model.add(Dense(units = 128))
model.add(Dense(units = 64))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

#create testing dataset

test_data = scaled_data[training_data_len-60: , :]

x_test=[]
y_test = dataset[training_data_len: , :]

for i in range(60, len(test_data)):
 x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

