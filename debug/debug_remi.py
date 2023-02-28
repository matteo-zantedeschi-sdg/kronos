import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.append("/home/leon/Doc/programm/SDG/hera/kronos/")

from kronos.forecast_udf import forecast_udf_gen

pd_arera_valid = pd.read_csv(
    "/home/leon/Doc/programm/SDG/hera/fcst_local/df_arera_valid_es.csv"
    # "/home/leon/Doc/programm/SDG/hera/fcst_local/pdr.csv"
)
param = (
    pd.read_csv("/home/leon/Doc/programm/SDG/hera/fcst_local/param.csv")
    .set_index("chiave")
    .to_dict()["valore"]
)


param["KEY_COL"] = "arera"
param["DATE_COL"] = "ts_battuta"
param["METRIC_COL"] = "battuta"
param["FCST_COL"] = "portata_mezzoraria_fcst"

param["FCST_FIRST_DATE"] = "2022-01-12"
param["CURRENT_DATE"] = "2022-01-11"

param["PDR_FCST_HORIZON"] = 7
param["N_TEST"] = 1
param["N_UNIT_TEST"] = 1


param["FCST_MODELS_CONFIG"] = {
    #     "prophet_1": {
    #         "model_flavor": "prophet",
    #         "interval_width": 0.95,
    #         "growth": "logistic",
    #         "yearly_seasonality": True,
    #         "weekly_seasonality": True,
    #         "daily_seasonality": False,
    #         "seasonality_mode": "multiplicative",
    #         "floor": 0,
    #         "country_holidays": "IT"
    #     },
    #     "prophet_2": {
    #         "model_flavor": "prophet",
    #         "interval_width": 0.95,
    #         "growth": "linear",
    #         "yearly_seasonality": True,
    #         "weekly_seasonality": True,
    #         "daily_seasonality": False,
    #         "seasonality_mode": "additive",
    #         "floor": 0,
    #         "country_holidays": "IT"
    #     },
    #     "tensorflow_1": {
    #         "model_flavor": "tensorflow",
    #         "nn_type": "rnn",
    #         "n_units": 128,
    #         "activation": "relu",
    #         "epochs": 25,
    #         "n_inputs": 30
    #     },
    #     "tensorflow_2": {
    #         "model_flavor": "tensorflow",
    #         "nn_type": "lstm",
    #         "n_units": 128,
    #         "activation": "relu",
    #         "epochs": 25,
    #         "n_inputs": 30
    #     },
    # "pmdarima_1": {
    #     "model_flavor": "pmdarima",
    #     "m": 7,
    #     "seasonal": True,
    #     "select_variables": False,
    # },
    "pmdarima_2": {
        "model_flavor": "pmdarima",
        "m": 7,
        "seasonal": True,
        "select_variables": True,
        "prediction_method": "percentile_85",
    }
}


# Read all params from dictionary

# key to split data set in different series -> arera
key_col = param["KEY_COL"]
# key that identifies collumns -> ts_battuta
date_col = param["DATE_COL"]
# column used for prediction -> battuta
metric_col = param["METRIC_COL"]

# column to save forecast
fcst_col = param["FCST_COL"]

# additional columns that must be apended to the data set -> should be imported from controll panell?
# qualita_status, azione, modello
quality_col = param["QUALITY_COL"]
action_col = param["ACTION_COL"]
models_col = param["MODELS_COL"]

# prima data di forecast in principio deve essere il giorno dopo il curent date
fcst_first_date = datetime.strptime(param["FCST_FIRST_DATE"], "%Y-%m-%d").date()
current_date = datetime.strptime(param["CURRENT_DATE"], "%Y-%m-%d").date()

# TODO: Da scrivere nel modo corretto sulla tabella dei parametri
# dictionary with models parameters to train(run)
# models_config = json.loads(d_param['FCST_MODELS_CONFIG'].replace('""', '"')[1:-1])
models_config = param["FCST_MODELS_CONFIG"]

# number of observation to use in the test -> 7
n_test = int(param["N_TEST"])
# Number of points to forecast in the model deployment's unit test. -> 7
n_unit_test = int(param["N_UNIT_TEST"])

# a horizon for the forecast -> 5
fcst_horizon = int(param["PDR_FCST_HORIZON"])

dt_creation_col = param["DT_CREAZIONE_COL"]
dt_reference_col = param["DT_RIFERIMENTO_COL"]

# 'rmse | mape'
fcst_competition_metrics = (
    param["FCST_COMPETITION_METRICS"].lower().replace(" ", "").split("|")
)
# '0.5 | 0.5'
fcst_competition_metric_weights = [
    float(metric)
    for metric in param["FCST_DEFAULT_COMPETITION_METRIC_WEIGHTS"]
    .replace(" ", "")
    .split("|")
]


# arr = ["G_35524", "G_114065", "G_114114", "G_38138"]
# # i_r = 3
# i_r = 2

# print(arr[i_r])

# df_single_arera = pd_arera_valid[(pd_arera_valid["arera"] == arr[i_r])].copy(deep=True)
df_single_arera = pd_arera_valid.copy(deep=True)

df_single_arera[date_col] = [
    datetime.fromisoformat(i).date() for i in df_single_arera[date_col]
]

df_arera_valid_fcst = (
    df_single_arera[
        df_single_arera[date_col] <= current_date + timedelta(days=fcst_horizon)
    ]
    .sort_values(by=[date_col], ascending=False)
    .reset_index()
    .copy(deep=True)
)
# df_single_arera['date'] = [datetime.strptime(i[:-14], '%Y-%m-%d')
#                                 for i in df_single_arera['ts_battuta']]

# df_single_arera['date'] = pd.to_datetime(pd_arera_valid['ts_battuta'])


df_copy = df_arera_valid_fcst.copy(deep=True)
df_arera_valid_fcst.loc[
    df_arera_valid_fcst["ts_battuta"] > current_date, "battuta"
] = None
df_arera_valid_fcst.loc[
    df_arera_valid_fcst["ts_battuta"] > current_date, "missing_fraction"
] = 0
df_arera_valid_fcst.loc[
    df_arera_valid_fcst["ts_battuta"] > current_date, "remi_active"
] = 5
df_arera_valid_fcst.loc[
    df_arera_valid_fcst["ts_battuta"] > current_date, "flag_missing"
] = 0


df_arera_valid_fcst[key_col] = df_arera_valid_fcst[key_col].astype("string") + "_test"
df_arera_valid_fcst[action_col] = "prediction"
# df_arera_valid_fcst[action_col] = "training"
# df_arera_valid_fcst[action_col] = "competition"


current_date = current_date + timedelta(days=2)
fcst_horizon -= 2
df_arera_valid_fcst["classe_desc"] = "smooth"


import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Init mlflow client
client = MlflowClient()


forecast_udf = forecast_udf_gen(
    client=client,
    key_col=key_col,
    date_col=date_col,
    metric_col=metric_col,
    fcst_col=fcst_col,
    quality_col=quality_col,
    action_col=action_col,
    models_col=models_col,
    models_config=models_config,
    today_date=current_date,
    #     fcst_first_date=fcst_first_date,
    #     n_test=n_test,
    #     n_unit_test=n_unit_test,
    horizon=fcst_horizon,
    dt_creation_col=dt_creation_col,
    dt_reference_col=dt_reference_col,
    fcst_competition_metrics=fcst_competition_metrics,
    fcst_competition_metric_weights=fcst_competition_metric_weights,
    x_reg_columns=[
        "sabato",
        "domenica",
        "year",
        "mean_temperatura",
        "max_temperatura",
        "min_temperatura",
        "new_year",
        "epiphany",
        "local_holiday",
        "easter_moday",
        "easter_sunday",
        "liberation_day",
        "labour_day",
        "republic_day",
        "assumption",
        "all_saint",
        "immaculate",
        "christmas",
        "boxing",
        "lock_down",
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "flag_missing",
        "missing_fraction",
        "n_festivita_locale",
    ],
    future_only=True,
)


import logging

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.ERROR)


pd_res = forecast_udf(df_arera_valid_fcst)
