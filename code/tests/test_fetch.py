import unittest
from unittest.mock import patch, MagicMock
from fetch import SDSSDataFetcher 
import pandas as pd
import io

class TestSDSSDataFetcher(unittest.TestCase):
    def setUp(self):
        self.sdss_fetcher = SDSSDataFetcher()
        self.data = pd.read_csv('./code/tests/sample_data.csv')
        self.data_columns = list(self.data.columns)

    @patch('astroquery.sdss.SDSS.query_sql')
    def test_fetch_by_adql(self, mock_query):
        mock_query.return_value = self.data.head(10)
        result = self.sdss_fetcher.fetch_by_adql("SELECT TOP 10 * FROM SpecObj")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 10)

    @patch('fetch.SDSSDataFetcher.fetch_by_adql')
    def test_fetch_by_constraints(self, mock_fetch_by_adql):
        mock_fetch_by_adql.return_value = self.data
        constraints = {'ra': '<10', 'dec': '>0'}
        result = self.sdss_fetcher.fetch_by_constraints("SpecObj", 10, constraints)
        self.assertTrue(mock_fetch_by_adql.called)
        self.assertIsInstance(result, pd.DataFrame)

    def test_construct_query_from_constraints(self):
        constraints = {'ra': '<10', 'dec': '>0'}
        expected_query = "SELECT TOP 10 * FROM SpecObj WHERE ra <10 AND dec >0"
        query = self.sdss_fetcher._construct_query_from_constraints("SpecObj", 10, constraints)
        self.assertEqual(query, expected_query)

    def test_process_sdss_format_data(self):
        csv_data = self.data.to_csv(index=False)
        data = io.StringIO(csv_data)
        result = self.sdss_fetcher.process_sdss_format_data(data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data))
        self.assertEqual(list(result.columns), self.data_columns)

    @patch('fetch.SDSS.query_sql')
    def test_fetch_by_adql_exception(self, mock_query):
        # Setup the mock to raise an exception
        mock_query.side_effect = Exception("Test exception")

        # Create an instance of SDSSDataFetcher
        sdss_fetcher = SDSSDataFetcher()

        # Call fetch_by_adql and expect an exception to be handled
        result = sdss_fetcher.fetch_by_adql("SELECT * FROM TestTable")

        # Assert that the result is None due to the exception
        self.assertIsNone(result)
        
if __name__ == '__main__':
    unittest.main()
