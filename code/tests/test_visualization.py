import unittest
import matplotlib
# Use a non-interactive backend suitable for testing
matplotlib.use('Agg')  # Set the backend to 'Agg'
from unittest.mock import patch
from spectral_analysis import *
from visualization import *


class TestVisualization(unittest.TestCase):
    """A test class for the Visualization class."""

    def setUp(self):
        """
        Set up the test environment.

        This method creates an instance of the Visualization class
        for testing purposes.
        """
        sa = spectral_analysis(query="SELECT TOP 1 * FROM SpecObj")
        spec_obj = sa.SpecObjs[0]
        self.vis = Visualization(0.25, spec_obj)

    def test_calc_inferred_cont(self):
        """
        Test the calculation of inferred continuum.

        Checks if the inferred continuum is calculated properly.
        """
        self.vis.calc_inferred_cont()
        self.assertIsNotNone(self.vis.inferred_cont)
        self.assertTrue(len(self.vis.inferred_cont) > 0)

    @patch("matplotlib.pyplot.show")
    def test_plot_spec(self, mock_show):
        """
        Test the plotting of the spectrum.

        Verifies if the spectrum plotting function is called properly.
        """
        self.vis.calc_inferred_cont()
        self.vis.plot_spec()
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
