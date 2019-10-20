import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class OneHotTransformer(TransformerMixin):
    def __init__(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode
        self.onehotencoder = OneHotEncoder()
        self.le = LabelEncoder()

    def fit(self, df=None, y=None):
        return self

    def transform(self, df):
        try:
            result = pd.DataFrame()
            dfc = df.copy()
            for feature_name in self.columns_to_encode:
                result[feature_name] = self.le.fit_transform(dfc[feature_name])
            print(result)
            print(self.onehotencoder.fit_transform(result).toarray())
            result = pd.DataFrame(self.onehotencoder.fit_transform(result).toarray())
            return result

        except KeyError:
            cols_error = list(set(self.columns) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
