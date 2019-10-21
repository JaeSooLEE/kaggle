import unittest
from preprocessing.transformers.onehot_encoder_transformer import OneHotTransformer
import pandas as pd
import numpy as np
import collections


class TestOneHotTransformer(unittest.TestCase):
    def setUp(self):
        column_c1 = np.array(["male", "female", "female", "male", "male", "male", "female", "female", "male"])
        column_c2 = np.array(["France", "Spain", "Spain", "Germany", "Italy", "France", "Italy", "Germany", "Spain"])
        column_c3 = np.array([0, 1, 0, 0, 0, 0, 1, 0, np.nan])
        self.df = pd.DataFrame({
            "sex": column_c1,
            "country": column_c2,
            "column_c3": column_c3
        })
        self.columns_to_onehot = ["sex", "country"]
        self.onehotTransformer = OneHotTransformer(self.columns_to_onehot)

        male = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1])
        female = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0])
        france = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0])
        spain = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1])
        germany = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0])
        italy = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0])
        self.filled_df = pd.DataFrame(collections.OrderedDict([
            ("sex_female", female),
            ("sex_male", male),
            ("country_France", france),
            ("country_Germany", germany),
            ("country_Italy", italy),
            ("country_Spain", spain)
        ]))

    def test_transform(self):
        self.onehotTransformer.fit(self.df)
        transformed_df = self.onehotTransformer.transform(self.df)
        pd.testing.assert_frame_equal(transformed_df, self.filled_df)
