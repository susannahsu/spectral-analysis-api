import unittest
from unittest.mock import patch
from classification import spectral_classification
from spectral_analysis import SpecObj, spectral_analysis
import pandas as pd
import joblib
import os

class TestSpectralClassification(unittest.TestCase):

    def setUp(self):
        self.spec_objs = spectral_analysis(directory='./code/tests/sample_data.csv')
        self.spectral_classifier = spectral_classification(self.spec_objs, predict=True)
        print(self.spectral_classifier.data)

    def test_data_collection(self):
        self.spectral_classifier.data_collection()
        self.assertNotEqual(len(self.spectral_classifier.data), 0)

    def test_data_preprocessing(self):
        self.assertFalse(self.spectral_classifier.data.isnull().values.any())

    def test_encode_target(self):
        self.spectral_classifier.encode_target()
        self.assertIn('classification_encoded', self.spectral_classifier.data.columns)

    def test_setup_data(self):
        self.spectral_classifier.setup_data()
        self.assertIsNotNone(self.spectral_classifier.X)
        self.assertIsNotNone(self.spectral_classifier.y)

    def test_setup_model(self):         
        self.spec_objs = spectral_analysis(directory='./code/tests/sample_data.csv')         
        self.spectral_classifier = spectral_classification(self.spec_objs)         
        self.assertIsNotNone(self.spectral_classifier.classifier_model)         
        self.assertGreater(self.spectral_classifier.model_score, 0 )
        
    def test_predict(self):
        test_data = self.spectral_classifier.X
        predictions = self.spectral_classifier.predict(test_data)
        self.assertIsNone(predictions)

    def test_save_and_load_model(self):
        filename = 'test_model.joblib'
        self.spectral_classifier.save_model(filename)
        self.spectral_classifier.load_model(filename)
        self.assertIsNotNone(self.spectral_classifier.classifier_model)
        os.remove(filename) 

if __name__ == '__main__':
    unittest.main()
