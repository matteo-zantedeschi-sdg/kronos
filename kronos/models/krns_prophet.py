# from fbprophet import Prophet
# import pandas as pd
#
#
# class KRNSProphet:
#     """
#     # TODO: Doc
#     """
#
#     def __init__(
#             self,
#             date_column: str,
#             key_column: str,
#             metric_column: str,
#             model: Prophet,
#             interval_width=0.95,
#             growth='linear',
#             daily_seasonality=False,
#             weekly_seasonality=True,
#             yearly_seasonality=True,
#             seasonality_mode='multiplicative'
#     ):
#         self.date_column = date_column
#         self.key_column = key_column
#         self.metric_column = metric_column
#         self.model = model
#         self.interval_width = interval_width
#         self.growth = growth
#         self.daily_seasonality = daily_seasonality
#         self.weekly_seasonality = weekly_seasonality
#         self.yearly_seasonality = yearly_seasonality
#         self.seasonality_mode = seasonality_mode
#
#     def fit(self, data: pd.DataFrame):
#         # Specific preprocessing
#         print("Renaming columns for prophet requirements")
#         data = data.rename(columns={self.date_column: "ds", self.metric_column: "y"})
#
#         # Define the model
#         self.model = Prophet(
#             interval_width=self.interval_width,
#             growth=self.growth,
#             daily_seasonality=self.daily_seasonality,
#             weekly_seasonality=self.weekly_seasonality,
#             yearly_seasonality=self.yearly_seasonality,
#             seasonality_mode=self.seasonality_mode
#         )
#
#         # Fit the model
#         print("Fitting the model")
#         self.model.fit(data)
#
#     def predict(self, data: pd.DataFrame, periods: int, freq: str, include_history: bool) -> pd.DataFrame:
#         # configure predictions
#         print("Configure prediction")
#         pred_conf = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
#
#         # make predictions
#         pred = self.model.predict(pred_conf)
#
#         # combine with input data
#         pred = pred[['ds', 'yhat']].set_index('ds')
#         data = data.rename(columns={self.date_column: 'ds', self.metric_column: 'y'})
#         data = data[['ds', self.key_column, 'y']].set_index('ds')
#         data = data.join(pred, how='left')
#         data.reset_index(level=0, inplace=True)
#
#         return data
