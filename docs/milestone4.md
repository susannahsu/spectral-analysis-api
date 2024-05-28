# Milestone 4

### Directory Structure
1. Root directory
* `README`
* `LICENSE`
* `pyproject.toml` for `PEP 517/518`
* and other configuration files

2. `src/` (where we store our primary pacakge modules)
* `data_processing/`
    - `data_preprocessor.py`: For data nornalization and noise filtering.
* `spectral_analysis/`
    - `spectral_analysis.py`: Core module for spectral computations.
* `cross_matching/`
    - `cross_matcher.py`: For comparing observed data against databases.
* `machine_learning/` (tbd)
    - `mla.py`: For machine learning algorithms for classification, regression, or clustering on spectral data.
* `visualization/` (tbd)
    - `visualization.py`: For dynamic and interactive visual representations of spectral data for easier interpretation.
* `feature_extraction/` (tbd)
    - `feature_extractor.py`: For identifying and extracting characteristic features from spectral data.
* `utils/`
    - `helpers.py`, `constants.py`: Utility scripts and constants used across the package.

3. `tests/`
* This directory contains our test suite.

4. `docs/`
* For Sphinx documentation.

5. `examples/`
* Sample scripts demonstrating the usage of our package.


### Test Suite
* The test suite will cover unit tests, integration tests, and functional tests.
* It will live in its own directory at the root level of our project, as demonstrated in the above structure. 
* The structure of our `tests/` directory will mirror the structure of our `src/` directory. So for every module in our `src/` directory, there will be a corresponding test module in the `tests/` directory.


### Distribution
* We will use `PyPI` with `PEP 517/518` compatibility. This approach uses `pyproject.toml` for project specifications and dependencies.


### Other Considerations
* Dependency management