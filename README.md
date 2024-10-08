# Fingerprint Spoofing Detection Project

## Project Overview
The Fingerprint Spoofing Detection Project utilizes a variety of machine learning techniques and statistical models to differentiate between genuine and spoofed fingerprint images. This project includes data visualization, linear and nonlinear classifiers, dimensionality reduction techniques, and model evaluations to determine the best approaches for fingerprint authentication.

## System Requirements
- **Python Version**: Python 3.8 or higher
- **Required Libraries**: numpy, pandas, matplotlib, scipy, seaborn, os

## Project Structure
The project consists of several Python scripts, each addressing different aspects of machine learning processes:

### Data Visualization and Analysis
- **vizualization_data.py**: Plots histograms and scatter plots of features and applies PCA and LDA for visualization.
  - Usage: `python vizualization_data.py`

### Classifiers
- **LDAclassification.py**: Implements LDA for dimensionality reduction and classification.
- **LogisticRegression.py**: Executes logistic regression models, including standard and weighted variants.
- **SupportVectorMachines.py**: Trains SVM models with linear, polynomial, and RBF kernels.
- **GMM.py**: Applies Gaussian Mixture Models using the EM algorithm for density estimation and classification.
- **OptimalBayesMVG.py**: Implements various Gaussian classifiers including MVG, Naive Bayes, and Tied MVG.
  - Usage for classifier scripts: `python [script_name].py`

### Dimensionality Reduction
- **pca.py**: Provides PCA implementation to reduce dimensions and project data.
- **lda.py**: Contains functions for performing Linear Discriminant Analysis.

### Model Evaluation and Comparison
- **Evaluation.py**: General script for evaluating classifier performance.
- **ComparisonBestModels.py**: Compares different models using Bayesian error analysis.
- **CalibrationAndFusion.py**: Adjusts model predictions to improve decision-making.
- **ExtendedGaussianModels.py**: Extends Gaussian models for better fit and performance.
- **bayesRisk.py**: Computes Bayesian risks and decision functions for evaluating classifiers.

## Installation
Ensure all dependencies are installed using:

    pip install numpy pandas matplotlib scipy seaborn


## Running the Scripts
To run any script, use the following command in the terminal:

    python [script_name].py

Replace `[script_name]` with the name of the script you wish to run, located in the project directory.


