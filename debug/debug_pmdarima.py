import pandas as pd
import datetime
import random
import pmdarima as pm

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
df.drop(df.columns.difference(['ds', 'y']), axis=1, inplace=True)
df.set_index('ds', inplace=True)
df_update.drop(df_update.columns.difference(['ds', 'y']), axis=1, inplace=True)
df_update.set_index('ds', inplace=True)

# Define the model
model = pm.auto_arima(df, seasonal=True, m=7)
model.last_training_day = df.index.max()

# scenario: 1 - fcst_first_date <= last training day and difference < n_days (still something to forecast) ##### actual data used as forecast
# scenario: 2 - fcst_first_date << last training day and difference >= n_days (nothing to forecast) ##### actual data used as forecast
# scenario: 3 - fcst_first_date > last training day and some data in between available #####
# scenario: 4 - fcst_first_date > last training day and no data in between available #####
last_training_day = model.last_training_day
fcst_first_date = last_training_day + datetime.timedelta(days=10)
n_days = 7
future_only = False

# Update model with last data (if any) and update last_training_day value
update_data = df_update[(last_training_day < df_update.index) & (df_update.index < fcst_first_date)]
if len(update_data) > 0:
    model.update(update_data)
    last_training_day = update_data.index.max()

# Compute the difference between last_training_day and fcst_first_date
difference = (fcst_first_date - last_training_day).days

# Compute actual forecast horizon
fcst_horizon = max(difference + n_days - 1, 0)

# Make predictions
if fcst_horizon > 0:
    pred = pd.DataFrame(
        data={
            'ds': [
                last_training_day + datetime.timedelta(days=x)
                for x in range(1, fcst_horizon + 1)
            ],
            'yhat': model.predict(n_periods=fcst_horizon),
        }
    )
else:
    pred = pd.DataFrame(data={'ds': [], 'yhat': []})

# Attach actual data on predictions
if difference < 0:
    # Keep last n actual data (n = difference - 1)
    actual_data = df.sort_values(by='ds', ascending=True).iloc[difference - 1:]
    # Reset index
    actual_data.reset_index(inplace=True)
    # Rename columns
    actual_data.rename(columns={'y': 'yhat'}, inplace=True)
    # Concat to pred and reset index
    pred = pd.concat([actual_data, pred])
    pred.reset_index(drop=True, inplace=True)

# Keep relevant data
if future_only:
    pred = pred[pred["ds"] >= fcst_first_date]

if difference < 0:
    pred = pred[pred["ds"] < fcst_first_date + datetime.timedelta(days=n_days)]
