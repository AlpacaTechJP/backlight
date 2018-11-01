import pandas as pd


class Label(pd.DataFrame):
    def __init__(self, label_type, df):

        super(Label, self).__init__(df)

        self._label_type = label_type

    @property
    def label_type(self):
        return self._label_type


class Labelizer:
    def __init__(self, **kwargs):
        self._params = kwargs.copy()
        self.validate_params()

    def validate_params(self):
        pass

    def generate(self, mkt):
        raise NotImplementedError
