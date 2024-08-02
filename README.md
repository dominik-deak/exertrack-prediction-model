# ExerTrack Machine Learning Model

## Overview

The ExerTrack machine learning model is developed to provide personalized exercise recommendations based on historical workout data. Implemented in a Jupyter Notebook, the model leverages TensorFlow and Keras to analyze user data and suggest optimal workout adjustments. The trained model is exported to TensorflowJS for integration with the ExerTrack server.

## Features

- **Data Preprocessing:** Cleans and preprocesses workout data, handling missing values and standardizing input features for model training.
- **Feature Engineering:** Generates new features from raw data, including moving averages and percentage changes, to enhance model performance.
- **Model Training:** Utilizes a TensorFlow Keras model to learn patterns in workout data and make predictions.
- **Evaluation Metrics:** Analyzes model performance using metrics such as accuracy, precision, recall, and F1-score.
- **Model Export:** Converts the trained model to TensorflowJS format for deployment in a Node.js environment.

## Technology Stack

- **Python:** Programming language for data analysis and model development.
- **Jupyter Notebook:** Interactive environment for exploratory data analysis and model training.
- **TensorFlow & Keras:** Libraries for building and training deep learning models.
- **Pandas:** Handles data manipulation and preprocessing.
- **Seaborn & Matplotlib:** Visualizes data and evaluation metrics with plots and charts.

## Data Preprocessing

The notebook begins by loading and preprocessing the [721 Weight Training Workouts](https://www.kaggle.com/datasets/joep89/weightlifting) dataset from Kaggle. Key steps include:

- **Data Cleaning:** Removes unnecessary columns and converts data types as needed.
- **Normalization:** Scales features to ensure consistent input for the model.
- **Feature Engineering:** Introduces new features like moving averages and percentage changes to capture trends in the data.

## Model Development

The model is built using TensorFlow and Keras, employing a neural network architecture optimized for time-series data. Steps include:

- **Model Architecture:** Defines a sequential model with dense layers and dropout for regularization.
- **Hyperparameter Tuning:** Utilizes K-Fold Cross Validation and Hyperband for optimal hyperparameter selection.
- **Training and Evaluation:** Trains the model on preprocessed data and evaluates performance on a test set.

## Evaluation

The model is evaluated using a variety of metrics, with results visualized through plots and confusion matrices. Key insights include:

- **Accuracy:** Measures overall performance, with the model achieving high accuracy across classes.
- **Confusion Matrix:** Identifies misclassification patterns and areas for improvement.
- **Classification Report:** Provides detailed metrics like precision, recall, and F1-score for each class.

## Model Export

Upon successful training and evaluation, the model is exported to a TensorflowJS format, enabling integration with the ExerTrack web server. The export process includes:

- **Model Conversion:** Converts the trained model to a format suitable for browser-based inference.
- **Deployment:** Prepares the model for use in a Node.js environment, facilitating real-time predictions.

## Misc info

This notebook was developed and tested with Python 3.10.12 on a Linux system
