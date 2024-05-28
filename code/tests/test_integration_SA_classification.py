import unittest
from unittest.mock import patch
from classification import spectral_classification
from spectral_analysis import SpecObj, spectral_analysis

class TestSpectralClassificationIntegration(unittest.TestCase):
    """
    Integration test class for spectral_analysis and classification modules.
    """
    def test_integration_model_generation(self):         
        self.spec_objs = spectral_analysis(directory='./code/tests/sample_data.csv')         
        self.spectral_classifier = spectral_classification(self.spec_objs)         
        self.assertIsNotNone(self.spectral_classifier.classifier_model)         
        self.assertGreater(self.spectral_classifier.model_score, 0 )