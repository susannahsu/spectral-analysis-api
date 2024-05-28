# import os
# os.chdir('code')

import unittest
from fetch import SDSSDataFetcher
from preprocess import preprocessing
from spectral_analysis import SpecObj, spectral_analysis
from classification import spectral_classification
from interactive_visualization import SpectrumPlotter
from astropy.table import Table
from astropy import units as u
import pandas as pd

class TestIntegration(unittest.TestCase):
    def test_data_fetch_and_preprocess(self):
        # Create an instance of the data fetcher
        fetcher = SDSSDataFetcher()

        # Fetch some sample data
        sample_query = "SELECT TOP 10 * FROM SpecObj"
        fetched_data = fetcher.fetch_by_adql(sample_query)

        # Check if data is fetched
        self.assertIsNotNone(fetched_data)

        # Convert to pandas DataFrame if it's an Astropy Table
        if isinstance(fetched_data, Table):
            fetched_data = fetched_data.to_pandas()
        
        self.assertIsInstance(fetched_data, pd.DataFrame)

        # Create an instance of the preprocessor
        proc = preprocessing()

        # Preprocess the fetched data
        processed_data = proc.ditch_units(fetched_data)

        # Check if data is processed
        self.assertFalse(any(isinstance(value, u.quantity.Quantity) for value in processed_data.values.flatten()))


if __name__ == '__main__':
    unittest.main()