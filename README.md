# MNIST-Image-Classification-using-Various-Machine-Learning-Models
This repository contains the code and documentation for classifying MNIST digit images using feature extraction with VGG16 and various machine learning models. 

**Data Processing:**

Loaded the MNIST dataset and preprocessed images by expanding dimensions, converting grayscale images to RGB, resizing to 48x48, and scaling pixel values.
Visualized data distributions before and after preprocessing using histograms.
Feature Extraction:

Employed the VGG16 pretrained model for feature extraction from RGB images.
Flattened the extracted features for use in machine learning models.

**Model Development:**

Implemented and trained the following machine learning models:
Neural Network
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Logistic Regression
Random Forest
Decision Tree
Support Vector Classifier (SVC)
Linear Regression

**Model Evaluation:**

Evaluated model performance using confusion matrices, classification reports, and accuracy scores.
Visualized the performance of each model through confusion matrix plots and a comparative accuracy bar chart.

**Prediction:**

Developed a function to preprocess new test samples, extract features using VGG16, and predict classes using trained models.
Evaluated the prediction accuracy for new test samples.

**Results:**

Achieved high accuracy across various models with Random Forest and Support Vector Classifier performing exceptionally well.
Successfully predicted classes for new test samples.

**Key Insights:**

Importance of data preprocessing and feature extraction in improving model performance.
Comparison of different machine learning models and their effectiveness in image classification.
Visualization of model performance using confusion matrices and accuracy charts.

**Usage:**

Clone the repository.
Install the required packages using pip install -r requirements.txt.
Run the Jupyter notebook to see the implementation and results.
