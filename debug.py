import keras.models
import tensorflow as tf
import pandas as pd
import datetime
import random
import pmdarima as pm
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator

# Config
n_days = 365
min_val = 0
max_val = 100

# Creation
df = pd.DataFrame(
    data={
        'key': ['1' for x in range(n_days)],
        'ds': [datetime.date.today() - datetime.timedelta(days=x) for x in range(n_days)],
        'x': [random.randint(min_val, max_val) for x in range(n_days)],
        'y': [random.randint(min_val, max_val) for x in range(n_days)]}
)

# Preprocess
# df = np.array(df['y'])


df.drop(df.columns.difference(['ds', 'y']),axis=1,inplace=True,)
df.set_index('ds', inplace=True)

model = pm.auto_arima(df)
df_1 = df[df['y'] == 1000]

model.update(df_1)
len(df_1)


model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(30, 1)),
        tf.keras.layers.GRU(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="mse")

# Define generator
ts_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    data=df,
    targets=df,
    length=30,
    batch_size=1,
)

# Fit the model
model.fit(
    ts_generator, steps_per_epoch=len(ts_generator), epochs=10
)

predictions = []
batch = df.astype('float32')[-30:].reshape((1, 30, 1))
for i in range(7):
    # Get the prediction value for the first batch
    pred_val = model.predict(batch)[0]

    # Append the prediction into the array
    predictions.append(pred_val[0])

    # Use the prediction to update the batch and remove the first value
    batch = np.append(batch[:, 1:, :], [[pred_val]], axis=1)


batch = df.astype('float32')[-30:].reshape((1, 30, 1))
pred_val = model.predict(batch)[0]
print(pred_val)

import os

path = os.path.join(os.getcwd(), "ciao")
tf.saved_model.save(model, path)


loaded = tf.saved_model.load(path)

loaded.signatures
loaded.tags

tf.saved_model.SERVING

# PMDARIMA #####
# Preprocess
df.drop(df.columns.difference(['ds', 'y']), axis=1, inplace=True)
df.set_index('ds', inplace=True)

# Train/test
df_train = df[-350:]
df_test = df[:-340]

# Fit your model
model = pm.auto_arima(df_train, seasonal=True, m=7)
model.last_day = df_train.index[0]

new_data = df_test[df_test.index > model.last_day]
len(new_data)
model.update(new_data)
pred_1 = model.predict(n_periods=7)
print(pred_1)



model.update(df_test)
pred_2 = model.predict(n_periods=7)

print(f"Pred_1: {pred_1}")
print(f"Pred_2: {pred_2}")

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
