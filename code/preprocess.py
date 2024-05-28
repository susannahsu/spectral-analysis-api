from astroquery.sdss import SDSS
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

import numpy as np
import pandas as pd
from astroquery.sdss import SDSS
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


class preprocessing:
    def __init__(self):
        pass

    def ditch_units(self, df):
        """
        Remove units from columns in a pandas DataFrame.

        Args:
        - df (pandas.DataFrame): Input DataFrame containing columns with units.

        Returns:
        - pandas.DataFrame: DataFrame with units removed.
        """
        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()
        for column in df.columns:
            if isinstance(df[column][0], u.quantity.Quantity):
                df[column] = df[column].apply(lambda x: x.value)
        return df

    def apply_redshift_correction(self, redshift_values, **kwargs):
        """
        Apply redshift correction using a cosmological model.

        Args:
        - redshift_values (pandas.Series or array-like): Redshift values.
        - **kwargs: Additional keyword arguments for FlatLambdaCDM.

        Returns:
        - astropy.units.quantity.Quantity: Co-moving distances after correction.
        """
        cosmo = FlatLambdaCDM(**kwargs)
        distances = cosmo.comoving_distance(redshift_values).to(u.Mpc)
        return distances

    def normalize(self, df, id_included=True, id_name='ObjID'):
        """
        Normalize numerical columns in a DataFrame.

        Args:
        - df (pandas.DataFrame): Input DataFrame.
        - id_included (bool): Flag to indicate if Object ID should be excluded from normalization.
        - id_name (str): Name of the Object ID column.

        Returns:
        - pandas.DataFrame: Normalized DataFrame.
        """
        numerical_cols = df.select_dtypes(include=['number']).columns

        if id_included:
            col_norm = [col for col in numerical_cols if (id_name.casefold() not in col.casefold())]
        else:
            col_norm = numerical_cols

        scaler = StandardScaler()
        df[col_norm] = scaler.fit_transform(df[col_norm])
        return df

    def remove_outliers(self, df, threshold=3, id_included=True, id_name='ObjID'):
        """
        Remove outliers using Z-score approach.

        Args:
        - df (pandas.DataFrame): Input DataFrame.
        - threshold (int): Z-score threshold for identifying outliers.
        - id_included (bool): Flag to indicate if Object ID should be excluded from outlier removal.
        - id_name (str): Name of the Object ID column.

        Returns:
        - pandas.DataFrame: DataFrame with outliers removed.
        """
        numerical_cols = df.select_dtypes(include=['number']).columns

        if id_included:
            col_rm = [col for col in numerical_cols if (id_name.casefold() not in col.casefold())]
        else:
            col_rm = numerical_cols

        z_scores = (df[col_rm] - df[col_rm].mean()) / df[col_rm].std()
        outliers = (z_scores > threshold).any(axis=1)
        df = df[~outliers]
        return df

    def impute(self, df, n_neighbors=5, id_included=True, id_name='ObjID'):
        """
        Fill missing numerical values using KNN imputation.

        Args:
        - df (pandas.DataFrame): Input DataFrame.
        - n_neighbors (int): Number of neighbors for KNN imputation.
        - id_included (bool): Flag to indicate if Object ID should be excluded from imputation.
        - id_name (str): Name of the Object ID column.

        Returns:
        - pandas.DataFrame: DataFrame with missing values imputed.
        """
        numerical_cols = df.select_dtypes(include=['number']).columns

        if id_included:
            numerical_cols = [col for col in numerical_cols if (id_name.casefold() not in col.casefold())]
        else:
            numerical_cols = numerical_cols

        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        return df
