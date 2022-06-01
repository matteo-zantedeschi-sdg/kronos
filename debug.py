import keras.layers
import keras as keras
import pandas as pd
import numpy as np
import datetime
import random
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# Config
n_days = 365
min_val = 0
max_val = 100

# Creation
df = pd.DataFrame(data={
    'key': ['1' for x in range(n_days)],
    'ds': [datetime.date.today() - datetime.timedelta(days=x) for x in range(n_days)],
    'x': [random.randint(min_val, max_val) for x in range(n_days)],
    'y': [random.randint(min_val, max_val) for x in range(n_days)]}
)

type(df.x)

# RNN #####
#Preprocess
df.drop(df.columns.difference(['ds', 'y']), axis=1, inplace=True)
df.set_index('ds', inplace=True)

# Train/test
df_train = df[:80]
df_test = df[80:]

scaler = MinMaxScaler()

scaler.fit(df_train)
scaled_train = scaler.transform(df_train)
scaled_test = scaler.transform(df_test)

n_input = scaled_train.shape[0]-1
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=n_features)

# Fit your model
model=Sequential([
    # keras.layers.LSTM(100, activation='relu', input_shape=(n_input, n_features)),
    keras.layers.RNN(100, activation='relu', input_shape=(n_input, n_features)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(generator, epochs=50)

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

last_train_batch = scaled_train[-n_input:]
last_train_batch = last_train_batch.reshape((1, n_input, 1))

model.predict(last_train_batch)

scaled_test[0]

test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(df_test)):
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]

    # append the prediction into the array
    test_predictions.append(current_pred)

    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

test_predictions

df_test.head()

true_predictions = scaler.inverse_transform(test_predictions)

df_test['Predictions'] = true_predictions

df_test.plot(figsize=(14,5))

# Caluclate error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(df_test['y'],df_test['Predictions']))
print(rmse)

# PMDARIMA #####
# Preprocess
df.drop(df.columns.difference(['ds', 'y']), axis=1, inplace=True)
df.set_index('ds', inplace=True)

# Train/test
df_train = df[:80]
df_test = df[80:]

# Fit your model
model = pm.auto_arima(df, seasonal=True, m=7)

# make your forecasts
n_days = 100
pred = pd.DataFrame(data={
    'ds': [datetime.date.today() + datetime.timedelta(days=x) for x in range(n_days)],
    'yhat': model.predict(n_periods=n_days)
})


# PROPHET #####
model = Prophet()
model.fit(df)

model_last_date = model.history_dates[0].date()
fcst_first_date = datetime.date(2022, 6, 1)
difference = (fcst_first_date - model_last_date).days
n_days = 5

pred_config = model.make_future_dataframe(periods=difference + n_days - 1, freq="d", include_history=False)

# make predictions
pred = model.predict(pred_config)

# Convert to date
pred['ds'] = pred['ds'].dt.date

pred = pred[pred['ds'] >= fcst_first_date]

print(pred)

class Test:
    def __init__(self, input: pd.DataFrame):
        self.input = input.copy()
        self.input.rename(columns={'col1': 'col1b'}, inplace=True)


d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

test = Test(input=df)

print(test.input)
print(df)