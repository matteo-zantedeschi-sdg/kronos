import pandas as pd
import datetime
import random
import tensorflow as tf
import numpy as np

# Config
n_days = 365
min_val = 0
max_val = 100

# Creation data
df = pd.DataFrame(
    data={
        'key': ['1' for x in range(n_days)],
        'ds': [datetime.date.today() - datetime.timedelta(days=x) for x in range(n_days)],
        'x': [random.randint(min_val, max_val) for x in range(n_days)],
        'y': [random.randint(min_val, max_val) for x in range(n_days)]}
)

df_update = pd.DataFrame(
    data={
        'key': ['1' for x in range(n_days + 5)],
        'ds': [datetime.date.today() + datetime.timedelta(days=5) - datetime.timedelta(days=x) for x in range(n_days + 5)],
        'x': [random.randint(min_val, max_val) for x in range(n_days + 5)],
        'y': [random.randint(min_val, max_val) for x in range(n_days + 5)]}
)

# Preprocess
df_pr = np.array(df['y'])
df_update_pr = np.array(df_update['y'])

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 1), name="input"),
    tf.keras.layers.SimpleRNN(units=128, activation='relu', name="rnn_1"),
    tf.keras.layers.Dense(units=1, name="output"),
])

# Compile the model
model.compile(optimizer="adam", loss="mse")

# Define generator
ts_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    data=df_pr,
    targets=df_pr,
    length=30,
    batch_size=1,
)

# Fit the model
model.fit(ts_generator, steps_per_epoch=len(ts_generator), epochs=10)

# scenario: 1 - fcst_first_date <= last training day and difference < n_days (still something to forecast) ##### all data is predicted
# scenario: 2 - fcst_first_date << last training day and difference >= n_days (nothing to forecast) ##### all data is predicted
# scenario: 3 - fcst_first_date > last training day and some data in between available ##### no update strategy, intermediate data is used to feed the model
# scenario: 4 - fcst_first_date > last training day and no data in between available #####

# Since the first step is to keep only historic data (data previous of fcst_first_date), in every scenario all the forecasts are predicted by the model
# (difference have to be at least 1) by giving the last n_inputs of historic_data to the model fitted in a autoregressive way.
last_training_day = df['ds'].max()
fcst_first_date = last_training_day + datetime.timedelta(days=10)
n_days = 7
future_only = False

# Keep only historic data
historic_data = df_update[df_update['ds'] < fcst_first_date]

# Compute last observed historical day
last_observed_day = historic_data['ds'].max()

# Compute the difference between last_observed_day and fcst_first_date
difference = (fcst_first_date - last_observed_day).days

# Compute actual forecast horizon
fcst_horizon = max(difference + n_days - 1, 0)

# Preprocess historic data
historic_data = np.array(historic_data['y'])

# Autoregressive prediction
predictions = []
batch = historic_data.astype("float32")[-30 :].reshape((1, 30, 1))
for i in range(fcst_horizon):
    # Get the prediction value for the first batch: we need to differentiate when we directly use the model after training or when we load it from mlflow.
    if type(model) == tf.keras.Sequential:
        # Model directly used after training
        pred_val = model.predict(batch)[0]
    else:
        # Model loaded from mlflow model registry
        # Note: 'input' is the name of the first layer of the network, 'output' the name of the last one
        pred_val = model(input=batch)["output"].numpy()[0]

    # Append the prediction into the array
    predictions.append(pred_val[0])

    # Use the prediction to update the batch and remove the first value
    batch = np.append(batch[:, 1:, :], [[pred_val]], axis=1)

# Make predictions dataframe
pred = pd.DataFrame(
    data={
        'ds': [
            last_observed_day + datetime.timedelta(days=x)
            for x in range(1, fcst_horizon + 1)
        ],
        'yhat': predictions,
    }
)

# Keep relevant data
if future_only:
    pred = pred[pred["ds"] >= fcst_first_date]