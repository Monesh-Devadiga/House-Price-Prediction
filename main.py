import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing  # Add this import
import warnings
warnings.filterwarnings('ignore')
   
# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Step 1: Load and preprocess the data
def load_and_preprocess_data():
    """
    Load the California housing dataset
    """
    print("=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    
    # Load California housing dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MEDV'] = housing.target * 100  # Multiply by 100 to get prices in $1000s (approx)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Features: {list(housing.feature_names)}")
    print(f"Target: MEDV (house prices in $1000s)")
    print(f"First few rows:")
    print(df.head())
    print()
    
    return df

# Step 2: Exploratory Data Analysis
def exploratory_data_analysis(df):
    """
    Perform EDA on the dataset
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Basic info
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Info:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title('Correlation Matrix of Housing Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['MEDV'], bins=30, kde=True)
    plt.title('Distribution of House Prices (MEDV)', fontsize=16)
    plt.xlabel('Median Value of Homes ($1000s)')
    plt.ylabel('Frequency')
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Pairplot of important features
    # For California housing, use different important features
    important_features = ['MedInc', 'AveRooms', 'AveOccup', 'Population', 'MEDV']
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[important_features])
    plt.suptitle('Pairplot of Important Features', y=1.02, fontsize=16)
    plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

# Step 3: Prepare data for modeling
def prepare_data(df):
    """
    Split data into features and target, then into train/test sets
    """
    # Features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=" * 60)
    print("DATA SPLIT")
    print("=" * 60)
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Step 4: Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple regression models and compare their performance
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'Model': model,
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train R2': train_r2,
            'Test R2': test_r2,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'CV R2 Mean': cv_scores.mean(),
            'CV R2 Std': cv_scores.std()
        }
        
        print(f"\n{name}:")
        print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        print(f"  Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
        print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return results

# Step 5: Hyperparameter tuning for best model
def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest and Gradient Boosting
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Random Forest tuning
    print("\nTuning Random Forest...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    
    print(f"Best Random Forest Parameters: {rf_grid.best_params_}")
    print(f"Best Random Forest CV R2: {rf_grid.best_score_:.4f}")
    
    # Gradient Boosting tuning
    print("\nTuning Gradient Boosting...")
    gb_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_samples_split': [2, 5, 10]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    gb_grid.fit(X_train, y_train)
    
    print(f"Best Gradient Boosting Parameters: {gb_grid.best_params_}")
    print(f"Best Gradient Boosting CV R2: {gb_grid.best_score_:.4f}")
    
    return rf_grid.best_estimator_, gb_grid.best_estimator_

# Step 6: Visualize results
def visualize_results(results, X_test, y_test, best_rf, best_gb):
    """
    Create visualizations to compare model performance
    """
    print("\n" + "=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Model Comparison Bar Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Prepare data for plotting
    model_names = list(results.keys())
    test_r2_scores = [results[name]['Test R2'] for name in model_names]
    test_mse_scores = [results[name]['Test MSE'] for name in model_names]
    test_mae_scores = [results[name]['Test MAE'] for name in model_names]
    cv_r2_scores = [results[name]['CV R2 Mean'] for name in model_names]
    
    # R2 Score comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(model_names, test_r2_scores, color='skyblue', edgecolor='navy')
    ax1.set_title('Test R2 Score by Model', fontsize=14)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R2 Score')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, test_r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # MSE comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(model_names, test_mse_scores, color='lightcoral', edgecolor='darkred')
    ax2.set_title('Test MSE by Model', fontsize=14)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('MSE')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, test_mse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # MAE comparison
    ax3 = axes[0, 2]
    bars3 = ax3.bar(model_names, test_mae_scores, color='lightgreen', edgecolor='darkgreen')
    ax3.set_title('Test MAE by Model', fontsize=14)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('MAE')
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, test_mae_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # CV R2 comparison
    ax4 = axes[1, 0]
    bars4 = ax4.bar(model_names, cv_r2_scores, color='gold', edgecolor='orange')
    ax4.set_title('Cross-Validation R2 by Model', fontsize=14)
    ax4.set_xlabel('Model')
    ax4.set_ylabel('CV R2')
    ax4.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars4, cv_r2_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Actual vs Predicted for best models
    # Random Forest
    ax5 = axes[1, 1]
    y_pred_rf = best_rf.predict(X_test)
    ax5.scatter(y_test, y_pred_rf, alpha=0.6, color='blue', edgecolor='black')
    ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax5.set_xlabel('Actual Prices')
    ax5.set_ylabel('Predicted Prices')
    ax5.set_title('Random Forest: Actual vs Predicted', fontsize=14)
    
    # Gradient Boosting
    ax6 = axes[1, 2]
    y_pred_gb = best_gb.predict(X_test)
    ax6.scatter(y_test, y_pred_gb, alpha=0.6, color='green', edgecolor='black')
    ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax6.set_xlabel('Actual Prices')
    ax6.set_ylabel('Predicted Prices')
    ax6.set_title('Gradient Boosting: Actual vs Predicted', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Feature Importance from Random Forest
    plt.figure(figsize=(12, 6))
    feature_names = list(df.drop('MEDV', axis=1).columns)  # Get actual feature names
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title('Feature Importance from Random Forest', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Residual Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Random Forest residuals
    residuals_rf = y_test - y_pred_rf
    axes[0].scatter(y_pred_rf, residuals_rf, alpha=0.6, color='blue', edgecolor='black')
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Random Forest: Residual Plot', fontsize=14)
    
    # Gradient Boosting residuals
    residuals_gb = y_test - y_pred_gb
    axes[1].scatter(y_pred_gb, residuals_gb, alpha=0.6, color='green', edgecolor='black')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Gradient Boosting: Residual Plot', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Step 7: Summary Report
def generate_summary_report(results, best_rf, best_gb, X_test, y_test, df):
    """
    Generate a comprehensive summary report
    """
    print("\n" + "=" * 60)
    print("FINAL SUMMARY REPORT")
    print("=" * 60)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test R2': [results[model]['Test R2'] for model in results],
        'Test MSE': [results[model]['Test MSE'] for model in results],
        'Test MAE': [results[model]['Test MAE'] for model in results],
        'CV R2 Mean': [results[model]['CV R2 Mean'] for model in results]
    }).sort_values('Test R2', ascending=False)
    
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Identify best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_test_r2 = comparison_df.iloc[0]['Test R2']
    
    print(f"\n{'='*40}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test R2 Score: {best_test_r2:.4f}")
    print(f"{'='*40}")
    
    # Evaluate best model on test set
    if best_model_name == 'Random Forest':
        best_model = best_rf
    elif best_model_name == 'Gradient Boosting':
        best_model = best_gb
    else:
        # Find the model from results
        for name, model_info in results.items():
            if name == best_model_name:
                best_model = model_info['Model']
                break
    
    y_pred_best = best_model.predict(X_test)
    
    print("\nDetailed Metrics for Best Model:")
    print(f"  R2 Score: {r2_score(y_test, y_pred_best):.4f}")
    print(f"  MSE: {mean_squared_error(y_test, y_pred_best):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred_best):.4f}")
    
    # Get feature names
    feature_names = list(df.drop('MEDV', axis=1).columns)
    
    # Recommendations
    print("\n" + "=" * 40)
    print("RECOMMENDATIONS")
    print("=" * 40)
    print(f"""
    1. Best Performing Model: Based on the results, ensemble methods (Random Forest/Gradient Boosting)
       outperform simple linear models for this housing dataset.
    
    2. Feature Engineering: Consider creating interaction terms or polynomial features
       to capture non-linear relationships.
    
    3. Further Improvement: Try advanced techniques like:
       - XGBoost or LightGBM
       - Neural Networks
       - Stacking multiple models
    
    4. Business Insights: The most important features for house price prediction are:
       - {feature_names[0]} (most important feature)
       - {feature_names[1]} (second most important)
       - {feature_names[2]} (third most important)
    """)

# Main execution
def main():
    """
    Main function to execute the complete pipeline
    """
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - COMPREHENSIVE MODELING PIPELINE")
    print("=" * 70)
    
    # Load and preprocess data
    global df  # Make df available to other functions
    df = load_and_preprocess_data()
    
    # Exploratory Data Analysis
    correlation_matrix = exploratory_data_analysis(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Hyperparameter tuning
    best_rf, best_gb = hyperparameter_tuning(X_train, y_train)
    
    # Visualize results
    visualize_results(results, X_test, y_test, best_rf, best_gb)
    
    # Generate summary report
    generate_summary_report(results, best_rf, best_gb, X_test, y_test, df)
    
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
