import unittest
import pandas as pd
import numpy as np
from spectral_analysis import SpecObj, spectral_analysis

class TestSpecObj(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('./code/tests/sample_data.csv')
        self.spec_obj = SpecObj(self.df.iloc[1])

    def test_metadatas(self):
        metadata = self.spec_obj.metadatas()
        self.assertIsInstance(metadata, pd.DataFrame)

    def test_spectra(self):
        spectra = self.spec_obj.spectra()
        self.assertIsNotNone(spectra)

    def test_wavelength(self):
        wavelength = self.spec_obj.wavelength()
        self.assertIsNotNone(wavelength)  

    def test_flux(self):
        flux = self.spec_obj.flux()
        self.assertIsNotNone(flux)  

    def test_peak(self):
        peak = self.spec_obj.peak()
        self.assertIsNotNone(peak)  

    def test_equivalent_width(self):
        eq_width = self.spec_obj.equivalent_width()
        self.assertIsNotNone(eq_width) 

    def test_redshifts(self):
        redshifts = self.spec_obj.redshifts()
        self.assertIsNotNone(redshifts)

    def test_classifier_features(self):
        classifier_features = self.spec_obj.classifier_features()
        self.assertIsInstance(classifier_features, pd.Series)

    def test_peak_values(self):
        peak = self.spec_obj.peak()
        self.assertIsInstance(peak, tuple)

    def test_interpolate_flux(self):
        known_wavelengths = [4000, 5000, 6000]  
        interpolated_flux = self.spec_obj.interpolate_flux(known_wavelengths)
        self.assertEqual(len(interpolated_flux), len(known_wavelengths))

    def test_align_spectra(self):
        target_wavelengths = [4000, 5000, 6000] 
        aligned_spectra = self.spec_obj.align_spectra(target_wavelengths)
        self.assertIn('wavelength', aligned_spectra)
        self.assertIn('flux', aligned_spectra)
        self.assertEqual(len(aligned_spectra['wavelength']), len(target_wavelengths))


class TestSpectralAnalysis(unittest.TestCase):
    def setUp(self):
        self.sa = spectral_analysis(query="SELECT TOP 1 * FROM SpecObj WHERE class = 'GALAXY'")

    def test_multiple_processing(self):
        self.sa.multiple_processing()
        self.assertGreater(len(self.sa.SpecObjs), 0)
        
    def test_spectral_data_structure(self):
        for spec_obj in self.sa.SpecObjs:
            self.assertIsInstance(spec_obj, SpecObj)
            self.assertIsNotNone(spec_obj.metadata)
            self.assertIsNotNone(spec_obj.spectra_data)
            
if __name__ == '__main__':
    unittest.main()