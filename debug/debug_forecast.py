from mlflow.tracking import MlflowClient
from pyspark.sql.types import StructType, StructField, DateType, FloatType, StringType
from kronos.forecast_udf import forecast_udf_gen

import sys
import datetime
from datetime import timedelta
import json
import kronos
import pandas as pd
import numpy as np
import mlflow

# set parameters
df_param = pd.read_csv("C:/data/hera/param.csv")
df_param = df_param[['chiave', 'valore']]

def get_param(param_name):
    return ((df_param[df_param['chiave'] == (param_name.upper())]['valore']).values[0])


df = pd.read_csv("C:/data/hera/df_03081001598913.csv")

df['year'] = df.giorno_gas.str[:4]
df['giorno_gas'] = df['giorno_gas'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
df['classe_desc'] = 'smooth'



# Init mlflow client
client = MlflowClient()

forecast_udf = forecast_udf_gen(
    client=client,
    key_col=get_param("key_col"),
    date_col=get_param("date_col"),
    metric_col=get_param("metric_col"),
    fcst_col=get_param("fcst_col"),
    quality_col=get_param("quality_col"),
    action_col=get_param("action_col"),
    models_col=get_param("models_col"),
    # models_config=ast.literal_eval(get_param("fcst_models_config")),
    # complete model dict
    # models_config= {"prophet_1":{"model_flavor":"prophet","interval_width":0.95,"growth":"logistic","yearly_seasonality":True,"weekly_seasonality":True,"daily_seasonality":False,"seasonality_mode":"multiplicative","floor":0,"country_holidays":"IT"},"prophet_2":{"model_flavor":"prophet","interval_width":0.95,"growth":"linear","yearly_seasonality":True,"weekly_seasonality":True,"daily_seasonality":False,"seasonality_mode":"additive","floor":0,"country_holidays":"IT"},"pmdarima_1":{"model_flavor":"pmdarima","m":7,"seasonal":True}},
    # models_config= {"prophet_1":{"model_flavor":"prophet","interval_width":0.95,"growth":"logistic","yearly_seasonality":True,"weekly_seasonality":True,"daily_seasonality":False,"seasonality_mode":"multiplicative","floor":0,"country_holidays":"IT"}},
    models_config= {"pmdarima_1":{"model_flavor":"pmdarima","m":7,"seasonal":True}},
    # models_config= {"lumpy":{"model_flavor":"lumpy", "start_P": 1, "max_P": 1, "start_D": 0, "max_D": 0, "start_Q": 1, "max_Q": 1, "m": 1, "start_p": 1, "max_p": 1, "start_d": 0, "max_d": 1, "start_q": 1, "max_q": 1}},
    # models_config= {"tensorflow_1":{"model_flavor":"tensorflow","nn_type":"rnn","n_units":128,"activation":"relu","epochs":10,"n_inputs":30}},
    # current_date=datetime.datetime.strptime('2022-11-16', '%Y-%m-%d').date(),
    today_date=datetime.datetime.strptime('2023-01-17', '%Y-%m-%d').date(),
    # fcst_first_date=datetime.datetime.strptime('2022-11-17', '%Y-%m-%d').date(),
    # fcst_horizon=int(get_param("pdr_fcst_horizon")),
    horizon=5,
    dt_creation_col=get_param("dt_creazione_col"),
    dt_reference_col=get_param("dt_riferimento_col"),
    fcst_competition_metrics=['max_perc_diff_3_days', 'max_perc_diff'],
    fcst_competition_metric_weights=[0.5, 0.5],
    future_only=True,
    x_reg_columns=['sabato','domenica','year', 'mean_temperatura', 'max_temperatura', 'min_temperatura', 'new_year', 'epiphany', 'local_holiday', 'easter_moday',
                   'easter_sunday', 'liberation_day', 'labour_day', 'republic_day', 'assumption', 'all_saint', 'immaculate', 'christmas', 'boxing', 'lock_down',
                   'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'FOURIER_S365-0', 'FOURIER_C365-0'])




# Define output schema
result_schema = StructType([
    StructField(get_param("key_col"), StringType()),
    StructField(get_param("date_col"), DateType()),
    StructField(get_param("fcst_col"), FloatType()),
    StructField(get_param("dt_creazione_col"), DateType()),
    StructField(get_param("dt_riferimento_col"), DateType())
])

test = forecast_udf(df)
