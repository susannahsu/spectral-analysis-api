# import os
# os.chdir('code')

from fetch import SDSSDataFetcher
from preprocess import preprocessing
from spectral_analysis import spectral_analysis, SpecObj
from classification import spectral_classification

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

# ------------------- Class Definitions -------------------

# Dynamic Spectrum Plotter
class SpectrumPlotter:
    def __init__(self, wavelengths, fluxes):
        """
        Initialize a SpectrumPlotter object.

        Args:
            wavelengths (list): List of wavelengths.
            fluxes (list): List of corresponding flux values.
        """
        self.wavelengths = wavelengths
        self.fluxes = fluxes
    
    def plot_raw_spectrum(self, color='blue'):
        """
        Plot the raw spectral data with interactivity.

        Args:
            color (str): Color of the plot line. Default is 'blue'.
        """
        if self.wavelengths is None or self.fluxes is None:
            raise ValueError("No data available for plotting")
    
        df = pd.DataFrame({
            'Wavelength': self.wavelengths,
            'Flux': self.fluxes
        })

        # Creating the plot
        fig = px.line(df, x='Wavelength', y='Flux',
            title='Spectral Data - Flux vs Wavelength',
            hover_data=['Wavelength', 'Flux'],
            line_shape='spline',
            render_mode='SVG')
        fig.update_traces(line=dict(color=color))
        fig.update_layout(xaxis_title='Wavelength', 
                          yaxis_title='Flux', 
                          hovermode='x',
                          xaxis=dict(
                            rangeslider=dict(visible=True),
                            type='linear'
                          ))
        fig.show()

    def plot_custom_spectrum(self, plot_type='line', line_style='solid'):
        """
        Plot the spectral data with custom options and interactivity.

        Args:
            plot_type (str): Type of plot ('line' or 'scatter'). Default is 'line'.
            line_style (str): Line style for line plot ('solid', 'dashed', or 'dotted'). Default is 'solid'.
        """
        if self.wavelengths is None or self.fluxes is None:
            raise ValueError("No data available for plotting")
        
        # Data preparation
        df = pd.DataFrame({'Wavelength': self.wavelengths, 'Flux': self.fluxes})
        
        # Plot type selection
        if plot_type == 'line':
            fig = px.line(df, x='Wavelength', y='Flux', 
            title='Spectral Data - Flux vs Wavelength',
            hover_data=['Wavelength', 'Flux'],
            line_shape='spline',
            render_mode='SVG')
            # Update line style
            if line_style == 'solid':
                fig.update_traces(line=dict(dash='solid'))
            elif line_style == 'dashed':
                fig.update_traces(line=dict(dash='dash'))
            elif line_style == 'dotted':
                fig.update_traces(line=dict(dash='dot'))
        elif plot_type == 'scatter':
            fig = px.scatter(df, x='Wavelength', y='Flux')

        # Customization
        fig.update_layout(xaxis_title='Wavelength',
                          yaxis_title='Flux', 
                          hovermode='closest',
                          xaxis=dict(
                            rangeslider=dict(visible=True),
                            type='linear'
                          ))

        fig.show()
    

# Feature Visualizer
class FeatureVisualizer:
    def __init__(self, df):
        self.df = df
    
    def plot_feature_distribution(self, feature, bins=None, show_stats=False):
        """
        Plot the distribution of a feature with interactivity.

        Args:
            feature (str): Name of the feature to visualize.
            bins (int): Number of bins for the histogram. Default is None.
            show_stats (bool): Whether to display mean, median, and std statistics on the plot. Default is False.
        """
        if feature not in self.df.columns:
            raise ValueError(f"Feature '{feature}' not found in the DataFrame")

        fig = px.histogram(self.df, x=feature, nbins=bins, 
                           marginal="box", 
                           title=f'Distribution of {feature}',
                           labels={feature: feature})
        
        # Update layout and axis ranges
        data_min = self.df[feature].min()
        data_max = self.df[feature].max()
        margin_factor = 0.05 * (data_max - data_min)

        fig.update_layout(xaxis_title=feature, 
                          yaxis_title='Frequency',
                          xaxis=dict(
                            rangeslider=dict(visible=True),
                            range=[data_min - margin_factor, data_max + margin_factor],
                            type='linear'
                          ),
                          margin=dict(l=50, r=50, t=50, b=50))  # Adjust margins

        if show_stats:
            mean_val = self.df[feature].mean()
            median_val = self.df[feature].median()
            std_val = self.df[feature].std()
            
            fig.add_annotation(x=mean_val, y=0.85, text=f'Mean: {mean_val:.2f}', showarrow=False, yref="paper")
            fig.add_annotation(x=median_val, y=0.75, text=f'Median: {median_val:.2f}', showarrow=False, yref="paper")
            fig.add_annotation(x=std_val, y=0.85, text=f'STD: {std_val:.2f}', showarrow=False, yref="paper")
        
        fig.show()

# Machine Learning Model Insights
# class ModelInsights:
#     def __init__(self, spectral_classifier):
#         self.classifier = spectral_classifier

#     def plot_confusion_matrix(self):
#         # Use the test data from the classifier to plot the confusion matrix
#         X_test, y_test = self.classifier.X_test, self.classifier.y_test
#         predictions = self.classifier.classifier_model.predict(X_test)

#         cm = confusion_matrix(y_test, predictions)
#         fig = ff.create_annotated_heatmap(cm,
#                           x=self.classifier.target_names, 
#                           y=self.classifier.target_names)
#         fig.update_layout(title='Confusion Matrix', 
#         xaxis_title='Predicted', yaxis_title='Actual')
#         fig.show()

#     def plot_roc_curve(self):
#         # Binarize the labels for multi-class ROC
#         y_bin = label_binarize(self.y_test, classes=self.classes)
#         n_classes = y_bin.shape[1]

#         # Compute ROC curve and ROC area for each class
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#         fig = go.Figure()
#         for i in range(n_classes):
#             fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], 
#                 self.model.predict_proba(self.X_test)[:, i])
#             roc_auc[i] = auc(fpr[i], tpr[i])

#             fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines', 
#                                      name=f'Class {self.classes[i]} (AUC = {roc_auc[i]:.2f})'))

#         fig.update_layout(title='Multi-class ROC Curve', 
#             xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
#         fig.show()

# ------------------- Example Workflow -------------------
# def example_workflow():
#     # Spectral Analysis
#     sa = spectral_analysis(query="SELECT TOP 10 * FROM SpecObj")
    
#     # Assuming SpecObj has wavelength and flux methods
#     wavelengths, fluxes = sa.SpecObjs[0].wavelength(), sa.SpecObjs[0].flux()

#     # Visualization of Spectral Data
#     spectrum_plotter = SpectrumPlotter(wavelengths, fluxes)
#     spectrum_plotter.plot_raw_spectrum()

# ------------------- Utility Functions -------------------

# ------------------- Main Execution -------------------

# if __name__ == "__main__":
#     example_workflow()
    # # Test data
    # wavelengths = np.linspace(400, 700, 100)  # Example wavelengths
    # fluxes = np.random.rand(100)  # Example random flux values

    # # Instantiate and use SpectrumPlotter
    # spectrum_plotter = SpectrumPlotter(wavelengths, fluxes)
    # spectrum_plotter.plot_raw_spectrum()

    # # Sample data for FeatureVisualizer
    # feature_data = pd.DataFrame({
    #     'peak_flux': np.random.rand(100) * 100,  # Example feature data
    #     'mean_flux': np.random.rand(100) * 50
    # })

    # # Testing FeatureVisualizer
    # feature_visualizer = FeatureVisualizer(feature_data)
    # feature_visualizer.plot_feature_distribution('peak_flux')

    # # Sample data for ClassificationVisualizer
    # classification_data = pd.DataFrame({
    #     'feature_1': np.random.rand(100),
    #     'feature_2': np.random.rand(100)
    # })
    # classification_labels = np.random.choice(['Class A', 'Class B', 'Class C'], size=100)

    # # Testing ClassificationVisualizer
    # classification_visualizer = ClassificationVisualizer(classification_data, classification_labels)
    # classification_visualizer.plot_classification_results()

    # from sklearn.datasets import make_classification
    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestClassifier

    # # Sample data for ModelInsights
    # X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # # Train a sample model
    # model = RandomForestClassifier()
    # model.fit(X_train, y_train)

    # # Testing ModelInsights
    # model_insights = ModelInsights(model, X_test, y_test)
    # model_insights.plot_confusion_matrix()

    # # Create a multi-class dataset
    # X, y = make_classification(n_samples=300, n_features=4, n_classes=3, n_informative=3, n_redundant=0, random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # # Train an XGBClassifier
    # model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    # model.fit(X_train, y_train)

    # # Testing ModelInsights with multi-class ROC Curve
    # classes = np.unique(y)
    # model_insights = ModelInsights(model, X_test, y_test, classes)
    # model_insights.plot_roc_curve()
