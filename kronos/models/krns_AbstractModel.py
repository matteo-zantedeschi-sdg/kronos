class AbstractModel:
    _PREDICTION_METHODS = [""]
    pred_method = None

    def __init__(self):
        pass

    @property
    def PREDICTION_METHODS(self):
        return self._PREDICTION_METHODS if not self.pred_method else [""]
