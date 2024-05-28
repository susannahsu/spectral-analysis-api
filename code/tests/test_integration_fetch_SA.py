import unittest
from fetch import SDSSDataFetcher
from spectral_analysis import SpecObj, spectral_analysis
from preprocess import preprocessing
import pandas as pd
import astropy

class TestIntegration(unittest.TestCase):

    def test_fetch_process_integration(self):
        data_fetcher = SDSSDataFetcher()
        raw_data = data_fetcher.fetch_by_adql("SELECT TOP 5 * FROM SpecObj WHERE class = 'GALAXY'")

        self.assertIsNotNone(raw_data)
        self.assertIsInstance(raw_data, astropy.table.Table)

        sa = spectral_analysis(query="SELECT TOP 5 * FROM SpecObj WHERE class = 'GALAXY'")

        self.assertGreater(len(sa.SpecObjs), 0)
        for spec_obj in sa.SpecObjs:
            self.assertIsInstance(spec_obj, SpecObj)

    def test_specobj_functions_integration(self):
        df = pd.read_csv('./code/tests/sample_data.csv')
        spec_obj = SpecObj(df.iloc[1])
        
        metadata = spec_obj.metadatas()
        self.assertIsInstance(metadata, pd.DataFrame)

        wavelength = spec_obj.wavelength()
        flux = spec_obj.flux()
        self.assertIsNotNone(wavelength)
        self.assertIsNotNone(flux)

        target_wavelengths = [5000, 6000, 7000]
        aligned_spectra = spec_obj.align_spectra(target_wavelengths)
        self.assertIn('wavelength', aligned_spectra)
        self.assertIn('flux', aligned_spectra)


if __name__ == '__main__':
    unittest.main()
