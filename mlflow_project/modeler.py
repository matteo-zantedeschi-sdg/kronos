import pandas as pd


class Modeler:

    def __init__(self):
        print("Modeler initialized")

    @staticmethod
    def train_test_split(data: pd.DataFrame, date_column: str, n_test: int) -> pd.DataFrame and pd.DataFrame:
        # TODO: Gestire il caso in cui non ci sono sufficienti record per il test

        train_data = data.sort_values(by=[date_column], ascending=False).iloc[n_test:, :]
        test_data = data.sort_values(by=[date_column], ascending=False).iloc[:n_test, :]

        return train_data, test_data

    @staticmethod
    def evaluate_model(data: pd.DataFrame, metric: str, predicted_col: str, true_value_col: str) -> int:

        # Transform metric in lower case and remove whitespaces
        metric = metric.lower().replace(" ", "")

        if metric not in ['rmse']:
            print(f"Requested metric {metric} is not supported.")
            print(f"Available metrics are: rmse")

        out = None
        if metric == 'rmse':
            out = ((data[true_value_col] - data[predicted_col]) ** 2).mean() ** .5

        return out
