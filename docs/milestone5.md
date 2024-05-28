# Milestone 5

### API Description 

The overall API has not changed since the last implementation. The modules stay the same and its functionalities are also identical - however, there has been some updates in terms of how the data is processed throughout the document (with a class object called SpecObj), but in terms of the larger scale of things, there are no changes in the structure. 

#### Changes in this milestone (Functions)

There have been changes in the modules themselves.

1. Data Pipeline Updates:
    - Like mentioned above, the structure of how data is passed between modules and within classes are not standardized with the class SpecObj, where it contains the data and metadata of a spectral object queried from the SDSS database. 

2. Updates on the Fetch Function:
    - The fetch function now uses 3 different methods of query - Using a normal query, Using a contrainted query, and Using a csv file. 

3. Updates on the Spectral Analysis:
    - The spectral analysis module now has more functions that allows getting further metadata from the spectral data, starting with aligned wavelength and other values needed for classification. 

4. Addition of Annex B modules:
    - We have added the Classifiaction module along with the correlation matrix. 
    - We have also added the interactive visualization module. 

5. Addition of Wavelength Visualization:
    - We now have a module to visualize the wavelength and flux of a spectral object as well as plotting its inferred continuity  

6. Tests and Interaction Tests
    - All modules have corresponding unit tests that pass. 
    - More than 2 modules have interaction tests to verify their successful interactions. 

Some of these changes (notablely 1, 2 and 3) improves functionality, usability, or adaptability to project requirements because it now contains a object class that has a divison of data and metadata, and makes it easier for further implementation of modules at a later date. It is extremely usable due to its intuative docstrings and variable/structure, and the README docs contain all the information about how to utilize them. This simplified making the integration tests as well, as it is now streamlined/standardized to receive required data into a new module. 

### Notes for the Teaching Staff

- The required Correlation Matrix for the classification module exists in the API_draft folder. 
- The integration tests have been performed on the following modules:
    - module fetch -> module Spectral Analysis 
    - module Spectral Analysis -> module Visualization 
    - module Spectral Analysis -> module Interactive Visualization 
- 
