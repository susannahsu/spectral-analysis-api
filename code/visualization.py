import matplotlib.pyplot as plt
from spectral_analysis import *
from statsmodels.nonparametric.smoothers_lowess import lowess

class Visualization:
    """A class to visualize spectral data and infer the continuum."""

    def __init__(self, span, spec_obj):
        """
        Initialize the Visualization object.

        Parameters:
        - span (float): The span for lowess smoothing.
        - spec_obj (object): The spectral object containing metadata.

        Attributes:
        - span (float): The span for lowess smoothing.
        - spec_obj (object): The spectral object containing metadata.
        - wavelength (numpy.ndarray): Array of wavelengths.
        - flux (numpy.ndarray): Array of flux values.
        - inferred_cont (numpy.ndarray): Inferred continuum using lowess.
        """
        self.span = span
        self.spec_obj = spec_obj
        self.wavelength = np.concatenate(spec_obj.metadata['wavelength']).astype(float)
        self.flux = np.concatenate(spec_obj.metadata['flux']).astype(float)
        self.inferred_cont = None  # Placeholder for inferred continuum

    def calc_inferred_cont(self):
        """
        Calculate the inferred continuum using lowess smoothing.
        """
        self.inferred_cont = lowess(self.flux, self.wavelength, frac=self.span, return_sorted=False)

    def plot_spec(self):
        """
        Plot the original spectrum and inferred continuum.
        """
        if self.inferred_cont is None or len(self.inferred_cont) == 0:
            raise ValueError("Inferred continuum is None or 0")
        else:
            plt.figure(figsize=(8, 6))
            plt.plot(self.wavelength, self.flux, label='Original Spectrum')
            plt.plot(self.wavelength, self.inferred_cont, label='Inferred Continuum', color='red')
            plt.xlabel('Wavelength')
            plt.ylabel('Flux')
            plt.title("Spectral Visualization")
            plt.legend()
            plt.show()


sa = spectral_analysis(query="SELECT TOP 1 * FROM SpecObj")
spec_obj = sa.SpecObjs[0]
vis = Visualization(0.25,spec_obj)
vis.calc_inferred_cont()
vis.plot_spec()
