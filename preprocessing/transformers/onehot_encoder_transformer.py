import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing


class OneHotTransformer(TransformerMixin):
    def __init__(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode
        self.encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
        self.le = preprocessing.LabelEncoder()
        self.df_2 = {}

    def fit(self, df=None, y=None):
        self.df_2 = df.apply(self.le.fit_transform)
        self.encoder.fit(self.df_2)
        print(self.df_2)
        return self

    def transform(self, df):
        try:
            result = pd.DataFrame()
            dfc = df.copy()
            for feature_name in self.columns_to_encode:
                #result = pd.concat([result, self.encoder.transform(dfc[feature_name])], axis=1)
                result = pd.concat([result, self.df_2[feature_name]], axis=1)
            print(result)
            result = self.encoder.fit_transform(result)
            result = result.astype('int64')
            print(result)
            return result

        except KeyError:
            cols_error = list(set(self.columns) - set(df.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
