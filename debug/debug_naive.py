import pandas as pd
import datetime
import random

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

last_training_day = df['ds'].max()
fcst_first_date = last_training_day - datetime.timedelta(days=10)
n_days = 7
future_only = False

# Keep only historic data
historic_data = df[df['ds'] < fcst_first_date]

# Compute last observed historical day
last_observed_day = historic_data['ds'].max()

# Compute the difference between last_observed_day and fcst_first_date
difference = (fcst_first_date - last_observed_day).days

# Compute actual forecast horizon
fcst_horizon = difference + n_days - 1

# Get last value
last_value = historic_data.sort_values(by='ds', ascending=False, inplace=False).iloc[0]['y']

# Create dummy pred df
pred = pd.DataFrame(
    {
        'ds': [last_observed_day + datetime.timedelta(days=x) for x in range(1, fcst_horizon + 1)],
        'yhat': [last_value for x in range(fcst_horizon)],
    }
)

# Keep relevant data
if future_only:
    pred = pred[pred["ds"] >= fcst_first_date]
