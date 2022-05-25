import pandas as pd
import datetime
import random
import pmdarima as pm
import numpy as np
from prophet import Prophet

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
