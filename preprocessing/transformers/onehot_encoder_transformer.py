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
                result = pd.concat([result, pd.get_dummies(dfc[feature_name])], axis=1)
            print(result)
            #result = pd.DataFrame(self.onehotencoder.fit_transform(result).toarray()).astype('int')
            result = result.astype('int64')
            return result

        except KeyError:
            cols_error = list(set(self.columns) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
