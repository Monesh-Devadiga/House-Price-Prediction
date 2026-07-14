# House-Price-Prediction
 
## Project Overview

This project implements an end-to-end machine learning pipeline for house price prediction using the California Housing Dataset. It performs data preprocessing, exploratory data analysis (EDA), feature scaling, model training, hyperparameter tuning and model evaluation to predict housing prices accurately. Multiple regression algorithms, including Linear Regression, Ridge Regression, Lasso Regression, Decision Tree, Random Forest and Gradient Boosting, are trained and compared using R² Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) and 5-fold Cross Validation. The project also generates visualizations such as correlation heatmaps, feature distributions, feature importance, actual vs predicted plots, residual analysis and model comparison charts to provide insights into model performance and identify the most effective algorithm for house price prediction.
        
## Features

- Loads and preprocesses the California Housing Dataset.
- Performs comprehensive Exploratory Data Analysis (EDA).
- Generates correlation matrices, feature distributions and pair plots.
- Splits and standardizes data for model training.
- Trains multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluates models using:
  - R² Score
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
- Performs 5-fold Cross Validation.
- Optimizes ensemble models using GridSearchCV.
- Visualizes:
  - Correlation Heatmap
  - Feature Importance
  - Actual vs Predicted Values
  - Residual Analysis
  - Model Performance Comparison
- Identifies the best-performing regression model for house price prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Workflow

1. Data Loading
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Scaling
5. Model Training
6. Model Evaluation
7. Hyperparameter Tuning
8. Performance Visualization
9. Best Model Selection

## Results

The project compares multiple machine learning regression algorithms and selects the best-performing model based on R² Score, MSE, MAE and cross-validation performance. Ensemble methods such as Random Forest and Gradient Boosting generally achieve higher predictive accuracy than traditional linear regression models.


---------------------------------------------------------------------------------------------------------------------
Created By: 
  [@Monesh Devadiga](https://github.com/Monesh-Devadiga)
