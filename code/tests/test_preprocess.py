import unittest
from preprocess import preprocessing
import pandas as pd
import numpy as np
from astropy import units as u
from sklearn.preprocessing import StandardScaler

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.preprocessor = preprocessing()

    def test_ditch_units(self):
        df = pd.DataFrame({
            'column1': [1 * u.m, 2 * u.m],
            'column2': [3 * u.s, 4 * u.s]
        })

        result = self.preprocessor.ditch_units(df)
        has_units = any(isinstance(value, u.quantity.Quantity) for value in result.values.flatten())
        self.assertFalse(has_units)

    def test_apply_redshift_correction(self):
        redshift_values = pd.Series([0.1, 0.2, 0.3])
        kwargs = {'H0': 70, 'Om0': 0.3}
        result = self.preprocessor.apply_redshift_correction(redshift_values, **kwargs)
        self.assertIsInstance(result, u.Quantity)

    # def test_normalize(self):
    #     df = pd.DataFrame({
    #         'A': [1, 2, 3],
    #         'B': [4, 5, 6],
    #         'C': ['x', 'y', 'z'],
    #         'ObjID': [69, 60, 12]
    #     })
    #     expected_result = pd.DataFrame({
    #         'A': [-1.22474487, 0.0, 1.22474487],
    #         'B': [-1.22474487, 0.0, 1.22474487],
    #         'C': ['x', 'y', 'z'],
    #         'ObjID': [69, 60, 12]
    #     })

    #     # Normalizing using StandardScaler
    #     scaler = StandardScaler()
    #     result = df.copy()
    #     result[['A', 'B']] = scaler.fit_transform(result[['A', 'B']])

    #     # Rounding to match expected_result precision
    #     result = result.round(8)
    #     expected_result = expected_result.round(8)

    #     self.assertTrue(result.equals(expected_result))

    def test_normalize(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['x', 'y', 'z'],
            'ObjID': [69, 60, 12]
        })

        preprocessor = preprocessing()

        # Test with ID column included
        result_with_id = preprocessor.normalize(df, id_included=True, id_name='ObjID')
        expected_with_id = df.copy()
        expected_with_id[['A', 'B']] = StandardScaler().fit_transform(df[['A', 'B']])
        pd.testing.assert_frame_equal(result_with_id.round(8), expected_with_id.round(8))

        # Test without ID column included
        result_without_id = preprocessor.normalize(df, id_included=False)
        expected_without_id = df.copy()
        expected_without_id[['A', 'B', 'ObjID']] = StandardScaler().fit_transform(df[['A', 'B', 'ObjID']])
        pd.testing.assert_frame_equal(result_without_id.round(8), expected_without_id.round(8))


    def test_remove_outliers(self):
        # Test remove_outliers function
        df = pd.DataFrame({
            'ObjID': [1, 2, 3, 4, 5],
            'X': [10, 20, 30, 40, 200],
            'Y': [15, 25, 35, 45, 55]
        })
        original_shape = df.shape
        result = self.preprocessor.remove_outliers(df, threshold=1)
        modified_shape = result.shape

        self.assertNotEqual(original_shape, modified_shape)

    def test_impute(self):
        # Test impute function
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [4, 5, np.nan],
            'C': [7, np.nan, 9],
            'ObjID': [69, 60, 12]
        })

        result = self.preprocessor.impute(df, n_neighbors=2)
        has_nan = result.isnull().values.any()
        self.assertFalse(has_nan)

if __name__ == '__main__':
    unittest.main()