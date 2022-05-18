import pandas as pd
import datetime


class KRNSNaive:
    """
    # TODO: Doc
    """

    def __init__(self, date_column: str, key_column: str, metric_column: str):
        self.date_column = date_column
        self.key_column = key_column
        self.metric_column = metric_column

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:

        # Retrieve pdr
        pdr_code = data.iloc[0][self.key_column]

        # Retrieve last value
        try:
            last_value = data.sort_values(
                by=self.date_column, ascending=False, inplace=False
            ).iloc[0][self.metric_column]
            pred_values = [last_value for i in range(7)]
        except:
            last_value = 0
            pred_values = [last_value for i in range(7)]

        # Create prediction dates
        pred_dates = [
            datetime.date.today() - datetime.timedelta(days=x) for x in range(7)
        ]

        # Make future dataframe
        pred_df = pd.DataFrame(
            {
                "pdr": [pdr_code for i in range(7)],
                "date": pred_dates,
                "yhat": pred_values,
            }
        )

        return pred_df
