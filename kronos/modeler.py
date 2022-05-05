import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Modeler:
    """
    TODO: Doc
    """

    def __init__(self):
        """
        TODO: Doc
        """
        print("Modeler initialized")

    @staticmethod
    def train_test_split(data: pd.DataFrame, date_col: str, n_test: int):
        """
        TODO: DOc
        :param data:
        :param date_col:
        :param n_test:
        :return:
        """

        logger.debug("Performing train/test split.")

        if data.shape[0] - n_test < n_test:
            logger.warning(f"Not enough records to perform train/test split: {data.shape[0]} rows, {n_test} for test")
            raise Exception(f"Not enough records to perform train/test split: {data.shape[0]} rows, {n_test} for test")
        else:
            train_data = data.sort_values(by=[date_col], ascending=False).iloc[n_test:, :]
            test_data = data.sort_values(by=[date_col], ascending=False).iloc[:n_test, :]
            logger.debug("Train/test split completed.")

            return train_data, test_data

    @staticmethod
    def evaluate_model(actual: pd.DataFrame, pred: pd.DataFrame, metric: str, pred_col: str, actual_col: str):
        """
        # TODO: Doc
        :param actual:
        :param pred:
        :param metric:
        :param actual_col:
        :param pred_col:
        :return:
        """

        logger.debug(f"Performing evaluation using {metric} metric.")

        supported_metrics = ['rmse']

        # Transform metric in lower case and remove whitespaces
        metric = metric.lower().replace(" ", "")

        if metric not in supported_metrics:
            logger.error(f"Requested metric {metric} is not supported. Available metrics are: {supported_metrics}")

        out = None
        if metric == 'rmse':
            out = ((actual[actual_col] - pred[pred_col]) ** 2).mean() ** .5
            logger.debug("Evaluation completed.")

        return out
