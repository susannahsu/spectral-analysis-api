import unittest
import matplotlib
# Use a non-interactive backend suitable for testing
matplotlib.use('Agg')  # Set the backend to 'Agg'
from spectral_analysis import spectral_analysis, SpecObj
from visualization import Visualization

class TestIntegration(unittest.TestCase):
    """
    Integration test class for spectral_analysis and visualization modules.
    """

    def test_integration(self):
        """
        Test the integration between spectral_analysis and visualization.

        This test verifies that the spec_obj attribute of the Visualization
        object matches the SpecObj instance obtained from spectral_analysis.
        """
        # Simulate spectral analysis with query
        sa = spectral_analysis(query="SELECT TOP 1 * FROM SpecObj")
        spec_obj = sa.SpecObjs[0]

        vis = Visualization(0.25, spec_obj)
        self.assertEqual(vis.spec_obj, spec_obj)

if __name__ == '__main__':
    unittest.main()
