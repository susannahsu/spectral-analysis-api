# APCOMP 207 Final Project: A Spectral Analysis Python Package

## Description
This library provides tools for performing spectral analysis on astronomical data. Users will be able to process, analyze, and visualize spectral data from public astronomy database or local files.

## Background on Spectral Analysis
Spectral analysis is used to examine the distribution of energy or power across different wavelengths or frequencies. In astronomy specifically, it is the analysis of the light spectra emanating from celestial objects. The spectrum of an object provides information about the object's physical properties (e.g. velocity, distance, temperature, and composition).

## Features
This API provides features such as data fetching from the SDSS database, data preprocessing, simple visualization, spectral analysis, classification ML module and an interactive visualization.  

## Documentation
This document contains information about the modules and how to utilize this API.

## Required Installations
Please see the "requirements.txt" document in the tests folder for a list of required modules that the user must install in Python before beginning. To install the required dependencies, run "pip install -r code/requirements.txt" in your Python environment.

## Contribution
Interested in contributing? We welcome pull requests. Please read through `CONTRIBUTING.md` for guidelines on how to make a contribution.

## Credits
(Alphabetic order of last name)
* Contributor: James Cao
* Contributor: Joon Kang
* Contributor: Renzo Silva
* Contributor: Susannah Su
* Contributor: Nathan Zhao
* Third-party library name (if we end up using)

## Core Libary
### Module: SDSS Data Fetching
#### Class `SDSSDataFetcher`: 
Handles fetching and processing data from the Sloan Digital Sky Survey (SDSS) database. It provides functionalities for querying the database using either ADQL (Astronomical Data Query Language) queries or specific constraints.

- `fetch_by_adql(adql_query)`: Fetches data from the SDSS database using an ADQL query. 
  - **input**: `adql_query` (string) - An ADQL query string.
  - **output**: Query results or `None` in case of an error.

- `fetch_by_constraints(table_name, num, constraints)`: Fetches data from the SDSS database based on specified constraints.
  - **input**: 
    - `table_name` (string) - Name of the table to query.
    - `num` (int) - Number of records to fetch.
    - `constraints` (dict) - A dictionary of constraints (e.g., `{'ra': '<10', 'dec': '>0'}`).
  - **output**: Query results or `None` in case of an error.

- `_construct_query_from_constraints(table_name, num, constraints)`: Constructs an ADQL query string based on provided constraints.
  - **input**: Same as `fetch_by_constraints`.
  - **output**: Constructed ADQL query string.

- `process_sdss_format_data(data)`: Processes data already in SDSS format.
  - **input**: `data` (string) - Path to a CSV file containing SDSS format data.
  - **output**: Pandas DataFrame containing the processed data.

#### Example Usage:
```python
# Create an instance of the SDSSDataFetcher class
data_fetcher = SDSSDataFetcher()

# Example ADQL query
adql_query = "SELECT TOP 10 * FROM PhotoObj WHERE ra < 10 AND dec > 0"
fetched_data = data_fetcher.fetch_by_adql(adql_query)

# Example constraints for fetching data
constraints = {'ra': '<10', 'dec': '>0'}
fetched_data_with_constraints = data_fetcher.fetch_by_constraints('PhotoObj', 10, constraints)

# Example for processing data in SDSS format
processed_data = data_fetcher.process_sdss_format_data('path/to/sdss_data.csv')
```

### Module: Data Preprocessing
#### Class `preprocessing`: 
Provides a set of tools for preprocessing data, especially designed for handling astronomical datasets. It includes methods for unit removal, redshift correction, normalization, outlier removal, and imputation.

- `ditch_units(df)`: Removes units from columns in a pandas DataFrame.
  - **Args**:
    - `df` (pandas.DataFrame): Input DataFrame containing columns with units.
  - **Returns**:
    - pandas.DataFrame: DataFrame with units removed.

- `apply_redshift_correction(redshift_values, **kwargs)`: Applies redshift correction using a cosmological model.
  - **Args**:
    - `redshift_values` (pandas.Series or array-like): Redshift values.
    - `**kwargs`: Additional keyword arguments for FlatLambdaCDM.
  - **Returns**:
    - astropy.units.quantity.Quantity: Co-moving distances after correction.

- `normalize(df, id_included=True, id_name='ObjID')`: Normalizes numerical columns in a DataFrame.
  - **Args**:
    - `df` (pandas.DataFrame): Input DataFrame.
    - `id_included` (bool): Flag to indicate if Object ID should be excluded from normalization.
    - `id_name` (str): Name of the Object ID column.
  - **Returns**:
    - pandas.DataFrame: Normalized DataFrame.

- `remove_outliers(df, threshold=3, id_included=True, id_name='ObjID')`: Removes outliers using Z-score approach.
  - **Args**:
    - `df` (pandas.DataFrame): Input DataFrame.
    - `threshold` (int): Z-score threshold for identifying outliers.
    - `id_included` (bool): Flag to indicate if Object ID should be excluded from outlier removal.
    - `id_name` (str): Name of the Object ID column.
  - **Returns**:
    - pandas.DataFrame: DataFrame with outliers removed.

- `impute(df, n_neighbors=5, id_included=True, id_name='ObjID')`: Fills missing numerical values using KNN imputation.
  - **Args**:
    - `df` (pandas.DataFrame): Input DataFrame.
    - `n_neighbors` (int): Number of neighbors for KNN imputation.
    - `id_included` (bool): Flag to indicate if Object ID should be excluded from imputation.
    - `id_name` (str): Name of the Object ID column.
  - **Returns**:
    - pandas.DataFrame: DataFrame with missing values imputed.

#### Example Usage:
```python
from preprocessing import preprocessing

# Create an instance of the preprocessing class
preprocessor = preprocessing()

# Example DataFrame
df = pd.DataFrame(...)

# Remove units from DataFrame
df_no_units = preprocessor.ditch_units(df)

# Apply redshift correction
redshift_values = pd.Series(...)
corrected_distances = preprocessor.apply_redshift_correction(redshift_values)

# Normalize the DataFrame
normalized_df = preprocessor.normalize(df)

# Remove outliers from DataFrame
df_no_outliers = preprocessor.remove_outliers(df)

# Impute missing values in DataFrame
imputed_df = preprocessor.impute(df)
```


### Module: Spectral Data Analysis
#### Class `SpecObj`: 
Represents a Spectral Object, encapsulating various functionalities for handling spectral data, including fetching spectra from SDSS, data processing, and feature extraction.

- **Attributes**:
  - `sds`: Instance of `SDSSDataFetcher`.
  - `pr`: Instance of `preprocessing`.
  - `raw_data`: Raw spectral data DataFrame.
  - `identifier`: DataFrame containing identifier information.
  - `coordinates`: DataFrame containing coordinates (ra, dec).
  - `spectra_data`: Spectra data fetched from SDSS.
  - `data`: Processed spectral data.
  - `metadata`: Metadata including various spectral features.

- **Methods**:
  - `metadatas()`: Returns the metadata as a DataFrame.
  - `spectra()`: Fetches and returns spectra data from SDSS.
  - `wavelength()`: Extracts and returns the wavelength data.
  - `flux()`: Extracts and returns the flux data.
  - `peak()`: Calculates and returns the peak wavelength and flux.
  - `equivalent_width()`: Calculates and returns the equivalent width data.
  - `interpolate_flux(target_wavelengths)`: Interpolates flux at given target wavelengths.
  - `align_spectra(target_wavelengths)`: Aligns the spectra to given target wavelengths.
  - `redshifts()`: Calculates and returns redshifts.
  - `classifier_features()`: Extracts and returns classifier features from raw data.

#### Example Usage:
```python
# Example raw spectral data DataFrame
raw_data = pd.DataFrame(...)

# Create an instance of SpecObj
spec_obj = SpecObj(raw_data)

# Access metadata
metadata = spec_obj.metadatas()

# Interpolate flux at target wavelengths
target_wavelengths = [4000, 5000, 6000]
interpolated_flux = spec_obj.interpolate_flux(target_wavelengths)

# Align spectra to target wavelengths
aligned_spectra = spec_obj.align_spectra(target_wavelengths)
```

### Module: Spectral Data Analysis
#### Class `spectral_analysis`: 
Facilitates the analysis of multiple spectral objects, providing functionalities for data fetching, preprocessing, and organization of spectral data into `SpecObj` instances.

- **Methods**:
  - `__init__(**kwargs)`: Constructor for initializing the class with different data sources. It can process raw data from a specified directory, fetch data by constraints, or execute a provided ADQL query.
  - `multiple_processing()`: Iterates over the processed data, creates `SpecObj` instances for each row, and stores them in the `SpecObjs` list.

- **Attributes**:
  - `sds`: Instance of `SDSSDataFetcher` for fetching data.
  - `raw_data`: Raw spectral data, either processed or fetched.
  - `ditched_units`: DataFrame with units removed from the raw data.
  - `SpecObjs`: List of `SpecObj` instances created from the processed data.

#### Example Usage:
```python
# Example usage with an ADQL query
params = {'query': "SELECT TOP 10 * FROM PhotoObj WHERE ra < 10 AND dec > 0"}

# Initialize the spectral_analysis class with the query
spectral_analysis_obj = spectral_analysis(**params)

# Process multiple spectral objects
spectral_analysis_obj.multiple_processing()

# Access the processed SpecObj instances
spec_objs = spectral_analysis_obj.SpecObjs
```


### Module: Spectral Visualization
#### Class `Visualization`: 
Provides tools for visualizing spectral data, including the ability to plot the original spectrum and infer the continuum using lowess smoothing.

- `__init__(span, spec_obj)`: Initializes the Visualization object.
  - **Parameters**:
    - `span` (float): The span for lowess smoothing.
    - `spec_obj` (object): The spectral object containing metadata.
  - **Attributes**:
    - `span` (float): The span for lowess smoothing.
    - `spec_obj` (object): The spectral object containing metadata.
    - `wavelength` (numpy.ndarray): Array of wavelengths.
    - `flux` (numpy.ndarray): Array of flux values.
    - `inferred_cont` (numpy.ndarray): Inferred continuum using lowess.

- `calc_inferred_cont()`: Calculates the inferred continuum using lowess smoothing.
  - **Args**:
    - None.
  - **Returns**:
    - None, updates `inferred_cont` with the calculated continuum.

- `plot_spec()`: Plots the original spectrum and inferred continuum.
  - **Args**:
    - None.
  - **Returns**:
    - None, displays a plot of the original spectrum and inferred continuum.

#### Example Usage:
```python
# Assuming spec_obj is a spectral object with metadata
visualization = Visualization(span=0.05, spec_obj=spec_obj)

# Calculate the inferred continuum
visualization.calc_inferred_cont()

# Plot the spectrum and inferred continuum
visualization.plot_spec()
```




## Additional Modules
### Module: Spectral Classification
#### Class `spectral_classification`: 
Provides functionalities for the classification of spectral data. It supports data collection from multiple `spectral_analysis` objects, preprocessing, and model training for classification purposes.

- **Attributes**:
  - `spectral_data`: List of `spectral_analysis` objects.
  - `data`: DataFrame containing consolidated spectral data.
  - `model_score`: Score of the trained classifier model.
  - `classifier_model`: Classifier model, typically an XGBoost model.
  - `target_names`: Names of the target classes for classification.
  - `encoder`: `LabelEncoder` instance for encoding target classes.
  - `scaler`: `StandardScaler` instance for normalizing data.

- **Methods**:
  - `data_collection()`: Collects and consolidates data from `spectral_analysis` objects.
  - `data_preprocessing()`: Preprocesses the collected data, including averaging, NaN removal, normalization, and outlier removal.
  - `average_lists()`: Computes average values for wavelength and flux from the spectral data.
  - `drop_nans()`: Removes rows with NaN values from the data.
  - `normalize_data()`: Normalizes numeric data using standard scaling.
  - `remove_outliers()`: Removes outliers from the numeric data using IQR.
  - `encode_target()`: Encodes target classes for classification.
  - `setup_data()`: Prepares the features (`X`) and target (`y`) for model training.
  - `setup_model()`: Sets up and trains the XGBoost classifier model.
  - `save_model()`: Saves the trained classifier model to a file.
  - `load_model()`: Loads a trained classifier model from a file.
  - `predict(data)`: Makes predictions using the trained classifier model.

#### Example Usage:
```python
# Assuming spectral_data is a list of spectral_analysis objects
spectral_cls = spectral_classification(*spectral_data)

# Data collection and preprocessing
spectral_cls.data_collection()
spectral_cls.data_preprocessing()

# Model setup and training
spectral_cls.setup_model()

# Save the trained model
spectral_cls.save_model('path/to/save/model')

# Load a trained model
spectral_cls.load_model('path/to/load/model')

# Make predictions
predictions = spectral_cls.predict(new_data)
```

### Module: Dynamic Spectrum Visualization
#### Class `SpectrumPlotter`: 
Enables interactive plotting of spectral data with customizable visual features.

- `__init__(wavelengths, fluxes)`: Initializes the SpectrumPlotter object.
  - **Args**:
    - `wavelengths` (list): List of wavelengths.
    - `fluxes` (list): List of corresponding flux values.

- `plot_raw_spectrum(color='blue')`: Plots the raw spectral data.
  - **Args**:
    - `color` (str): Color of the plot line. Default is 'blue'.
  - **Returns**:
    - None, displays an interactive plot of the raw spectral data.

- `plot_custom_spectrum(plot_type='line', line_style='solid')`: Plots the spectral data with custom options.
  - **Args**:
    - `plot_type` (str): Type of plot ('line' or 'scatter'). Default is 'line'.
    - `line_style` (str): Line style for line plot ('solid', 'dashed', or 'dotted'). Default is 'solid'.
  - **Returns**:
    - None, displays an interactive plot with custom settings.

#### Example Usage:
```python
# Initialize SpectrumPlotter with wavelengths and fluxes
plotter = SpectrumPlotter(wavelengths, fluxes)

# Plot raw spectral data
plotter.plot_raw_spectrum(color='blue')

# Plot custom spectral data
plotter.plot_custom_spectrum(plot_type='line', line_style='dashed')
```
