from preprocess import preprocessing
from fetch import SDSSDataFetcher
import numpy as np
from astroquery.sdss import SDSS
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import pandas as pd
import os


class SpecObj:
    """
    Class representing a Spectral Object.

    Args:
        raw_data: DataFrame containing raw spectral data.

    Attributes:
        sds: SDSSDataFetcher instance.
        pr: preprocessing instance.
        raw_data: Raw spectral data DataFrame.
        identifier: DataFrame containing identifier information.
        coordinates: DataFrame containing coordinates (ra, dec).
        spectra_data: Spectra data from SDSS.
        data: Processed spectral data.
        metadata: Metadata including identifier, redshifts, wavelength, flux, peak, equivalent width, and classifier features.

    Methods:
        - metadatas(): Returns metadata as a DataFrame.
        - spectra(): Fetches spectra data from SDSS.
        - wavelength(): Returns the wavelength data.
        - flux(): Returns the flux data.
        - peak(): Returns the peak wavelength and flux.
        - equivalent_width(): Returns the equivalent width data.
        - interpolate_flux(target_wavelengths): Interpolates flux at target wavelengths.
        - align_spectra(target_wavelengths): Aligns spectra to target wavelengths.
        - redshifts(): Calculates and returns redshifts.
        - classifier_features(): Returns classifier features from raw data.
    """
    def __init__(self, raw_data) -> None:
        self.sds = SDSSDataFetcher()
        self.pr = preprocessing()
        self.raw_data = raw_data
        self.identifier = self.raw_data[['specObjID', 'class', 'plate', 'fiberID', 'mjd']]
        self.coordinates = self.raw_data[['ra', 'dec']] 
        self.spectra_data = self.spectra()
        if self.spectra_data is not None:
            self.data = self.spectra_data[1].data
        else:
            self.data = None
        self.metadata = self.metadatas()
        
    def metadatas(self):
        """
        Constructs and returns the metadata as a DataFrame.

        This method combines various spectral properties like redshifts, wavelength, flux, peak, 
        equivalent width, and classifier features into a single DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the combined metadata.
        """
        metadata ={}
        metadata['identifier'] = self.identifier
        metadata['redshifts'] = self.redshifts()
        metadata['wavelength'] = self.wavelength()
        metadata['flux'] = self.flux()
        metadata['peak'] = self.peak()
        metadata['equivalent_width'] = self.equivalent_width()
        metadata['classifier_features'] = self.classifier_features()
        metadata = pd.DataFrame([metadata])
        return metadata
        
    def spectra(self):
        """
        Fetches and returns the spectra data from the Sloan Digital Sky Survey (SDSS).

        This method uses the plate, mjd, and fiberID from the identifier to fetch the spectra.

        Returns:
            The spectra data from SDSS if available, otherwise None.
        """
        plate, mjd, fiberID = self.identifier['plate'], self.identifier['mjd'], self.identifier['fiberID']
        spec = SDSS.get_spectra(plate=plate, fiberID=fiberID, mjd=mjd)
        if spec is not None and len(spec) > 0:
            return spec[0]
        else:
            print(f"No spectral data found for Plate={plate}, FiberID={fiberID}, MJD={mjd}")
            return None    
        
    def wavelength(self):
        """
        Extracts and returns the wavelength data from the spectra.

        The method converts the logarithmic wavelengths to linear scale.

        Returns:
            np.array: Array of wavelengths, or None if data is not available.
        """
        if self.data is not None:
            return 10**self.data['loglam']
        return None
    
    def flux(self):
        """
        Extracts and returns the flux data from the spectra.

        Returns:
            np.array: Array of flux values, or None if data is not available.
        """
        if self.data is not None:
            return self.data['flux']
        return None
    
    def peak(self):
        """
        Determines and returns the peak wavelength and flux from the spectra.

        This method finds the maximum flux and its corresponding wavelength.

        Returns:
            tuple: (peak wavelength, peak flux), or (None, None) if data is not available.
        """
        try:
            flux = self.flux()
            wavelength = self.wavelength()

            if flux is None or wavelength is None or len(flux) == 0 or len(wavelength) == 0:
                raise ValueError("Spectra data is missing or empty.")

            max_flux_index = np.argmax(flux)
            return wavelength[max_flux_index], flux[max_flux_index]

        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None 
    
    def equivalent_width(self):
        """
        Extracts and returns the equivalent width data from the spectra.

        Returns:
            np.array: Array of equivalent widths, or None if data is not available.
        """
        if self.spectra_data is not None:
            return self.spectra_data[3].data['LINEEW']
        return None
    
    def interpolate_flux(self, target_wavelengths):
        """
        Interpolates the flux values at given target wavelengths.
        
        :param target_wavelengths: List or array of target wavelengths.
        :return: Interpolated flux values at the target wavelengths.
        """
        try:
            if self.metadata['wavelength'][0] is None or self.metadata['flux'][0] is None:
                raise ValueError("Wavelength or flux data is missing.")

            interp_func = interp1d(self.metadata['wavelength'][0], self.metadata['flux'][0], kind='linear', bounds_error=False, fill_value="extrapolate")
            return interp_func(target_wavelengths)

        except Exception as e:
            print(f"Error in interpolation: {e}")
            return None

    def align_spectra(self, target_wavelengths):
        """
        Aligns the spectra to the same set of target wavelengths.
        
        :param target_wavelengths: List or array of target wavelengths.
        :return: Aligned spectra as a dictionary with wavelengths and interpolated fluxes.
        """
        try:
            aligned_spectra = {}
            aligned_spectra['wavelength'] = target_wavelengths
            aligned_spectra['flux'] = self.interpolate_flux(target_wavelengths)
            return aligned_spectra

        except Exception as e:
            print(f"Error in aligning spectra: {e}")
            return None
    
    def redshifts(self):
        """
        Calculates and returns the redshifts from the raw data.

        This method applies redshift correction to the raw data.

        Returns:
            np.array: Array of redshift values, or None if redshift data is not available.
        """
        if self.raw_data['z'] is not None:
            redshift = self.pr.apply_redshift_correction(self.raw_data['z'], H0=72, Om0=0.28)
            return redshift
        else:
            return None
        
    def classifier_features(self):
        """
        Extracts and returns classifier features from the raw data.

        Returns:
            pd.DataFrame: DataFrame containing classifier features.
        """
        return self.raw_data[['spectroFlux_u', 'spectroFlux_g', 'spectroFlux_r', 'spectroFlux_i', 'spectroFlux_z']]
    
    
class spectral_analysis:
    """
    Class for performing spectral analysis based on SDSS data.

    This class is designed to handle multiple spectral objects and perform batch processing.

    Args:
        **kwargs: Keyword arguments specifying the source and type of data to be analyzed.
                  Valid keywords include 'directory', 'table_name', 'num', 'constraints', and 'query'.

    Attributes:
        sds (SDSSDataFetcher): Instance of SDSSDataFetcher for data fetching.
        raw_data (pd.DataFrame): DataFrame containing raw spectral data fetched based on input criteria.
        ditched_units (pd.DataFrame): Processed data with units removed.
        SpecObjs (list[SpecObj]): List of SpecObj instances for batch processing.

    Methods:
        multiple_processing(): Processes multiple spectral objects and populates the SpecObjs list.
    """
    def __init__(self, **kwargs):
        self.sds = SDSSDataFetcher()
        if 'directory' in kwargs and os.path.isfile(kwargs['directory']):
            self.raw_data = self.sds.process_sdss_format_data(kwargs['directory'])
        elif 'table_name' in kwargs and 'num' in kwargs and 'constraints' in kwargs:
            self.raw_data = self.sds.fetch_by_constraints(kwargs['table_name'], kwargs['num'], kwargs['constraints'])
        elif 'query' in kwargs and isinstance(kwargs['query'], str):
            self.raw_data = self.sds.fetch_by_adql(kwargs['query'])  
        else:
            raise ValueError("Invalid input.")
        
        pr = preprocessing() 
        self.ditched_units = pr.ditch_units(self.raw_data)
        self.SpecObjs = []
        self.multiple_processing()
        
    def multiple_processing(self):
        """
        Processes multiple spectral objects based on the raw_data attribute.

        This method iterates over each row in the raw_data DataFrame, instantiates a SpecObj object
        for each row, and appends it to the SpecObjs list.
        """
        for _, row in self.ditched_units.iterrows():
            so = SpecObj(row)
            self.SpecObjs.append(so)


# sa = spectral_analysis(query="SELECT TOP 1 * FROM SpecObj")
# #print(sa.SpecObjs[0].metadata['identifier'])

# target_wavelengths = np.linspace(3500, 10000, 100)
# aligned_spectra = sa.SpecObjs[0].align_spectra(target_wavelengths)

