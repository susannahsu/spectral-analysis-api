import unittest
import pandas as pd
import numpy as np
from spectral_analysis import SpecObj, spectral_analysis
from classification import spectral_classification
from interactive_visualization import SpectrumPlotter, FeatureVisualizer
import plotly.graph_objs as go
from unittest.mock import patch

class TestInteractiveVisualization(unittest.TestCase):
    
    def setUp(self):
        # query = "SELECT TOP 10 * FROM SpecObj"
        # sa = spectral_analysis(query)
        # self.spec_obj = sa.data.SpecObjs[0]

        self.df = pd.read_csv('./code/tests/sample_data.csv')
        self.spec_obj = SpecObj(self.df.iloc[1])

    def testPlotRawSpectrum(self):
        wavelengths = self.spec_obj.wavelength()
        fluxes = self.spec_obj.flux()

        spectrum_plotter = SpectrumPlotter(wavelengths, fluxes)
        try:
            spectrum_plotter.plot_raw_spectrum()
            plot_created = True
        except Exception as e:
            plot_created = False
            print(f"Error: {e}")

        # Verify that the plot creation was successful
        self.assertTrue(plot_created)
    
    def testPlotCustomSpectrum(self):
        wavelengths = self.spec_obj.wavelength()
        fluxes = self.spec_obj.flux()

        spectrum_plotter = SpectrumPlotter(wavelengths, fluxes)
        try:
            spectrum_plotter.plot_custom_spectrum(plot_type='line', line_style='dashed')
            plot_created = True
        except Exception as e:
            plot_created = False
            print(f"Error: {e}")

        self.assertTrue(plot_created)

    def testFeatureVisualizer_basic(self):
        feature_name = 'spectroSynFlux_z'
        feature_visualizer = FeatureVisualizer(self.df)
        try:
            feature_visualizer.plot_feature_distribution(feature_name)
            plot_created = True
        except Exception as e:
            plot_created = False
            print(f"Error creating plot: {e}")

        # Verify that show was called
        self.assertTrue(plot_created)
    
    def testFeatureVisualizer_with_options(self):
        feature_name = 'spectroSynFlux_z'
        feature_visualizer = FeatureVisualizer(self.df)
        try:
            feature_visualizer.plot_feature_distribution(feature_name, bins=50, show_stats=True)
            plot_created = True
        except Exception as e:
            plot_created = False
            print(f"Error creating plot: {e}")

        self.assertTrue(plot_created)

    def testPlotRawSpectrum_with_invalid_data(self):
        # Test with invalid data (None)
        spectrum_plotter = SpectrumPlotter(None, None)
        try:
            spectrum_plotter.plot_raw_spectrum()
            plot_created = True
        except Exception as e:
            plot_created = False
            error_message = str(e)

        self.assertFalse(plot_created)
        self.assertIn("No data available for plotting", error_message)

    def testPlotCustomSpectrum_with_invalid_data(self):
        # Test with invalid data (None)
        spectrum_plotter = SpectrumPlotter(None, None)
        try:
            spectrum_plotter.plot_custom_spectrum()
            plot_created = True
        except Exception as e:
            plot_created = False
            error_message = str(e)

        self.assertFalse(plot_created)
        self.assertIn("No data available for plotting", error_message)

    def testFeatureVisualizer_with_invalid_feature(self):
        # Test with an invalid feature name
        feature_name = 'invalid_feature'
        feature_visualizer = FeatureVisualizer(self.df)
        try:
            feature_visualizer.plot_feature_distribution(feature_name)
            plot_created = True
        except Exception as e:
            plot_created = False
            error_message = str(e)

        # Verify that the plot creation was not successful
        self.assertFalse(plot_created)
        self.assertIn("Feature 'invalid_feature' not found", error_message)


if __name__ == '__main__':
    unittest.main()
