from spectral_analysis import * 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from preprocess import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

class spectral_classification:
    """
    Spectral Classification Class.

    Args:
        *spectral_data: Variable number of spectral_analysis objects.
        predict (bool): Set to True if the class is used for prediction.

    Attributes:
        spectral_data: List of spectral_analysis objects.
        data: DataFrame containing spectral data.
        model_score: Score of the classifier model.
        classifier_model: Classifier model (XGBoost).
        target_names: Names of the target classes.
        encoder: LabelEncoder for encoding target classes.
        scaler: StandardScaler for data normalization.

    Methods:
        - data_collection(): Collects data from spectral_analysis objects.
        - data_preprocessing(): Preprocesses the collected data.
        - average_lists(): Calculates the average values for wavelength and flux.
        - drop_nans(): Drops rows with NaN values.
        - normalize_data(): Normalizes numeric data.
        - remove_outliers(): Removes outliers from numeric data.
        - encode_target(): Encodes target classes.
        - setup_data(): Sets up X and y for model training.
        - setup_model(): Sets up and trains the XGBoost classifier.
        - save_model(): Saves the trained model to a file.
        - load_model(): Loads a trained model from a file.
        - predict(data): Makes predictions using the trained model.
    """
    def __init__(self, *spectral_data, predict = False) -> None:
        self.spectral_data = spectral_data
        self.data = pd.DataFrame()
        self.model_score = 0
        if predict == False:
            self.classifier_model = None
        else:
            self.classifier_model = self.load_model()
        self.target_names = None
        self.encoder = None
        self.scaler = None
        self.data_collection()
        self.data_preprocessing()
        self.encode_target()
        self.setup_data()
        
        if predict == False:
            self.setup_model()
        self.load_model()
        
    def data_collection(self):
        """
        Collects and aggregates data from multiple spectral_analysis objects into a single DataFrame.

        This method iterates through each spectral_analysis object, extracts relevant data from each SpecObj, 
        and concatenates them into a unified DataFrame.
        """
        all_dfs = []
        for query in self.spectral_data:
            for specObj in query.SpecObjs:
                specObj_data = {
                    'classification': [specObj.identifier['class']],
                    'spectroFlux_u': [specObj.metadata['classifier_features'][0]['spectroFlux_u']],
                    'spectroFlux_g': [specObj.metadata['classifier_features'][0]['spectroFlux_g']],
                    'spectroFlux_r': [specObj.metadata['classifier_features'][0]['spectroFlux_r']],
                    'spectroFlux_i': [specObj.metadata['classifier_features'][0]['spectroFlux_i']],
                    'spectroFlux_z': [specObj.metadata['classifier_features'][0]['spectroFlux_z']],
                    'wavelength': [specObj.metadata['wavelength'][0]],
                    'flux': [specObj.metadata['flux'][0]],
                    #'peak': [specObj.metadata['peak'][0]],
                }
                df = pd.DataFrame(specObj_data)
                all_dfs.append(df)

        self.data = pd.concat(all_dfs, ignore_index=True)
                
    def data_preprocessing(self):
        """
        Performs preprocessing steps on the collected data.

        This method sequentially calls other preprocessing methods like average_lists, drop_nans, normalize_data, 
        and remove_outliers to prepare the data for classification.
        """
        self.average_lists()
        self.drop_nans()
        self.normalize_data()
        self.remove_outliers()

    def average_lists(self):
        """
        Calculates and stores the average values of wavelength and flux in the data DataFrame.

        Replaces the list of wavelengths and fluxes with their average values for each spectral object.
        """
        self.data['average_wavelength'] = self.data['wavelength'].apply(lambda x: np.mean(x) if isinstance(x, np.ndarray) and x.size > 0 else np.nan)
        self.data['average_flux'] = self.data['flux'].apply(lambda x: np.mean(x) if isinstance(x, np.ndarray) and x.size > 0 else np.nan)
        self.data = self.data.drop(columns=['wavelength', 'flux'])

    def drop_nans(self):
        """
        Removes rows with NaN values from the data DataFrame.

        This method ensures that the dataset is free of missing values which might affect model training and predictions.
        """
        self.data.dropna(inplace=True)

    def normalize_data(self):
        """
        Normalizes numeric data in the data DataFrame using StandardScaler.

        This step is crucial for ensuring that all numerical features contribute equally to the analysis.
        """
        self.scaler = StandardScaler()
        numeric_cols = ['spectroFlux_u', 'spectroFlux_g', 'spectroFlux_r', 'spectroFlux_i', 'spectroFlux_z', 'average_wavelength', 'average_flux']
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])

    def remove_outliers(self):
        """
        Removes outliers from the numeric data in the data DataFrame.

        Outliers are detected using the Interquartile Range (IQR) method and are removed to improve model performance.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        Q1 = self.data[numeric_cols].quantile(0.25)
        Q3 = self.data[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        outlier_filter = ((self.data[numeric_cols] < (Q1 - 1.5 * IQR)) | (self.data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        self.data = self.data[~outlier_filter]
        
    def encode_target(self):
        """
        Encodes the target class labels using LabelEncoder.

        This is a necessary step to convert categorical class labels into a format suitable for model training.
        """
        self.data['classification'] = self.data['classification'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        self.encoder = LabelEncoder()
        self.data['classification_encoded'] = self.encoder.fit_transform(self.data['classification'])
        self.target_names = self.encoder.classes_ 

    def setup_data(self):
        """
        Prepares the feature matrix (X) and target vector (y) for model training.

        This method separates the classification labels from the feature data and encodes the labels.
        """
        self.X = self.data.drop(['classification', 'classification_encoded'], axis=1)
        self.y = self.data['classification_encoded']

    def setup_model(self):
        """
        Sets up, configures, and trains the XGBoost classifier model.

        Utilizes GridSearchCV for hyperparameter tuning and cross-validation to ensure the best model performance.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        }
        self.classifier_model = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), param_grid, cv=5, verbose=2, n_jobs=-1)

        self.classifier_model.fit(X_train, y_train)
        print(f"Best parameters: {self.classifier_model.best_params_}")
        self.model_score = self.classifier_model.best_score_
        
        predictions = self.classifier_model.predict(X_test)
        print(classification_report(y_test, predictions, target_names=self.target_names))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.target_names, yticklabels=self.target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        
        self.save_model()
        
    def save_model(self, filename='./spectral_classifier.joblib'):
        """
        Saves the trained classifier model to a file.

        Args:
            filename (str): Path to save the model file.
        """
        joblib.dump(self.classifier_model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='./spectral_classifier.joblib'):
        """
        Loads a trained classifier model from a file.

        Args:
            filename (str): Path to the model file to be loaded.
        """
        self.classifier_model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        
    def predict(self, data):
        """
        Makes predictions using the trained classifier model.

        Args:
            data (pd.DataFrame or np.ndarray): Data on which predictions are to be made.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predictions = self.classifier_model.predict(data)
        print(predictions)
    
#Training Data 
# galaxy_sa = spectral_analysis(query="SELECT TOP 20 * FROM SpecObj WHERE class = 'GALAXY'")
# qso_sa = spectral_analysis(query="SELECT TOP 20 * FROM SpecObj WHERE class = 'QSO'")
# star_sa = spectral_analysis(query="SELECT TOP 20 * FROM SpecObj WHERE class = 'STAR'")
# class_sa = spectral_classification(galaxy_sa, qso_sa, star_sa)


#Prediction
# sa = spectral_analysis(query="SELECT TOP 10 * FROM SpecObj WHERE class = 'QSO'")
# sample_data = spectral_classification(sa, predict=True)
# print(sample_data.classifier_model.best_score_)
# sample_data.predict(sample_data.X)

