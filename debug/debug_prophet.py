import pandas as pd
import datetime
import random
from prophet import Prophet


# Python
def stan_init(m):
    """Retrieve parameters from a trained model.

    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.

    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res

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

# Define model
model = Prophet(
    interval_width=0.95,
    growth='logistic',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
)

# Add floor and cap
df["floor"] = 0
df["cap"] = df['y'].max() * 10
df_update["floor"] = 0
df_update["cap"] = df_update['y'].max() * 10

# Add country holidays
model.add_country_holidays(country_name='IT')

# Fit the model
model.fit(df)

# scenario: 1 - fcst_first_date <= last training day and difference < n_days (still something to forecast) ##### actual data used as forecast
# scenario: 2 - fcst_first_date << last training day and difference >= n_days (nothing to forecast) ##### actual data used as forecast
# scenario: 3 - fcst_first_date > last training day and some data in between available #####
# scenario: 4 - fcst_first_date > last training day and no data in between available #####
last_training_day = model.history_dates[0].date()
fcst_first_date = last_training_day - datetime.timedelta(days=5)
n_days = 7
future_only = False

# Update model with last data (if any)
n_update_rows = len(df_update[(last_training_day < df_update['ds']) & (df_update['ds'] < fcst_first_date)])
if n_update_rows > 0:
    update_data = df_update[df_update['ds'] < fcst_first_date]
    old_model = model
    model = Prophet(
        interval_width=old_model.interval_width,
        growth=old_model.growth,
        daily_seasonality=old_model.daily_seasonality,
        weekly_seasonality=old_model.weekly_seasonality,
        yearly_seasonality=old_model.yearly_seasonality,
        seasonality_mode=old_model.seasonality_mode
    )
    model.add_country_holidays(country_name=old_model.country_holidays)
    model.fit(update_data, init=stan_init(old_model))
    last_training_day = model.history_dates[0].date()

# Compute the difference between last_training_day and fcst_first_date
difference = (fcst_first_date - last_training_day).days

# Set include_history based on the fact if fcst_first_date is older or newer than
include_history = False if difference > 0 else True

# Compute actual forecast horizon
fcst_horizon = max(difference + n_days - 1, 0)

# Configure predictions
pred_config = model.make_future_dataframe(
    periods=fcst_horizon, freq="d", include_history=include_history
)

# Add floor and cap
pred_config["floor"] = 0
pred_config["cap"] = df['y'].max() * 10

# Make predictions
pred = model.predict(pred_config)

# Convert datetime to date
pred["ds"] = pred["ds"].dt.date

# Keep relevant data
if future_only:
    pred = pred[pred["ds"] >= fcst_first_date]

if difference < 0:
    pred = pred[pred["ds"] < fcst_first_date + datetime.timedelta(days=n_days)]