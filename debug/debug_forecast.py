import sys
# from pyspark.sql.functions import when, to_date, col
import datetime
from datetime import timedelta
import json
import kronos

import pandas as pd
import numpy as np

# set parameters
df_param = pd.read_csv("C:/data/hera/param.csv")
df_param = df_param[['chiave', 'valore']]

def get_param(param_name):
    return ((df_param[df_param['chiave'] == (param_name.upper())]['valore']).values[0])


# get_param("n_test")


from mlflow.tracking import MlflowClient
from kronos.forecast_udf import forecast_udf_gen
import mlflow

# Init mlflow client
client = MlflowClient()

import ast

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
    #complete model dict
    # models_config= {"prophet_1":{"model_flavor":"prophet","interval_width":0.95,"growth":"logistic","yearly_seasonality":True,"weekly_seasonality":True,"daily_seasonality":False,"seasonality_mode":"multiplicative","floor":0,"country_holidays":"IT"},"prophet_2":{"model_flavor":"prophet","interval_width":0.95,"growth":"linear","yearly_seasonality":True,"weekly_seasonality":True,"daily_seasonality":False,"seasonality_mode":"additive","floor":0,"country_holidays":"IT"},"pmdarima_1":{"model_flavor":"pmdarima","m":7,"seasonal":True},"tensorflow_1":{"model_flavor":"tensorflow","nn_type":"rnn","n_units":128,"activation":"relu","epochs":10,"n_inputs":30}},
    # models_config= {"prophet_1":{"model_flavor":"prophet","interval_width":0.95,"growth":"logistic","yearly_seasonality":True,"weekly_seasonality":True,"daily_seasonality":False,"seasonality_mode":"multiplicative","floor":0,"country_holidays":"IT"}},
    models_config= {"pmdarima_1":{"model_flavor":"pmdarima","m":7,"seasonal":True}},
    # models_config= {"tensorflow_1":{"model_flavor":"tensorflow","nn_type":"rnn","n_units":128,"activation":"relu","epochs":10,"n_inputs":30}},
    current_date=datetime.datetime.strptime('2022-10-27', '%Y-%m-%d').date(),
    fcst_first_date=datetime.datetime.strptime('2022-10-28', '%Y-%m-%d').date(),
    n_test=int(get_param("n_test")),
    n_unit_test=int(get_param("n_unit_test")),
    fcst_horizon=int(get_param("pdr_fcst_horizon")),
    dt_creation_col=get_param("dt_creazione_col"),
    dt_reference_col=get_param("dt_riferimento_col"),
    # fcst_competition_metrics=get_param("fcst_competition_metrics"),
    fcst_competition_metrics=['max_perc_diff_3_days'],
    # fcst_competition_metric_weights=get_param("fcst_default_competition_metric_weights"),
    fcst_competition_metric_weights=[1],
    future_only=True,
    x_reg_columns=['sabato','domenica', 'festivita']
)



from pyspark.sql.types import StructType, StructField, DateType, FloatType, StringType

# Define output schema
result_schema = StructType([
    StructField(get_param("key_col"), StringType()),
    StructField(get_param("date_col"), DateType()),
    StructField(get_param("fcst_col"), FloatType()),
    StructField(get_param("dt_creazione_col"), DateType()),
    StructField(get_param("dt_riferimento_col"), DateType())
])

# read sample data

df = pd.read_csv("C:/data/hera/df_03081000333586.csv")

df['giorno_gas'] = df['giorno_gas'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())

test = forecast_udf(df)


#
# actual = np.array([ 840,  977, 1133, 1120, 1086, 1327,  890], dtype="int64")
# pred   = np.array([ 830,  877, 1232, 1220, 1096, 1454,  901], dtype="int64")
#
