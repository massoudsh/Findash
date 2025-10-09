#!/usr/bin/env python3
"""
Scikit-learn Gradient Boosting Bitcoin Analysis Script
Uses scikit-learn's GradientBoostingRegressor with processed features from dataset folder
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the processed features data"""
    print("\n📊 Loading processed features data...")
    
    data_path = "dataset/btc_processed_features_demo.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"✓ Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    
    return df

def prepare_features(df):
    """Prepare features and target for training"""
    print("\n🔧 Preparing features...")
    
    # Drop non-feature columns
    feature_cols = [col for col in df.columns if col not in ['Date', 'target']]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    # Handle any NaN values
    nan_mask = np.isnan(X).any(axis=1) | np.isnan(y)
    X = X[~nan_mask]
    y = y[~nan_mask]
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Removed {nan_mask.sum()} rows with NaN values")
    
    return X, y, feature_cols

def analyze_features(df, feature_cols):
    """Analyze feature importance and correlations"""
    print("\n📈 Analyzing features...")
    
    # Feature correlation with target
    correlations = df[feature_cols + ['target']].corr()['target'].abs().sort_values(ascending=False)
    
    print("\n🔝 Top 10 features by correlation with target:")
    for i, (feature, corr) in enumerate(correlations.head(11).items()):  # 11 to skip 'target' itself
        if feature != 'target':
            print(f"  {i:2d}. {feature:<25} {corr:.4f}")
    
    return correlations

def train_gradient_boosting_model(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting model using scikit-learn"""
    print("\n🤖 Training Gradient Boosting model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Gradient Boosting parameters optimized for financial time series
    gb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'max_features': 0.8,
        'alpha': 0.1,
        'random_state': 42,
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
        'tol': 1e-4
    }
    
    print(f"✓ Parameters: {gb_params}")
    
    # Initialize and train model
    model = GradientBoostingRegressor(**gb_params)
    model.fit(X_train_scaled, y_train)
    
    print(f"✓ Training completed with {model.n_estimators_} estimators")
    
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test, X_train, y_train):
    """Evaluate model performance"""
    print("\n📊 Evaluating model performance...")
    
    # Scale data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_metrics = {
        'MSE': mean_squared_error(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'R²': r2_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'MSE': mean_squared_error(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'R²': r2_score(y_test, y_test_pred)
    }
    
    print("\n📈 Training Performance:")
    for metric, value in train_metrics.items():
        print(f"  {metric:<6}: {value:.6f}")
    
    print("\n📉 Test Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric:<6}: {value:.6f}")
    
    # Price prediction analysis
    print("\n💰 Price Prediction Analysis:")
    current_return = y_test[-1]
    predicted_return = y_test_pred[-1]
    
    print(f"  Current return:    {current_return:.4f} ({current_return*100:.2f}%)")
    print(f"  Predicted return:  {predicted_return:.4f} ({predicted_return*100:.2f}%)")
    print(f"  Prediction error:  {abs(current_return - predicted_return):.4f}")
    
    return y_train_pred, y_test_pred, train_metrics, test_metrics

def plot_feature_importance(model, feature_cols):
    """Plot feature importance"""
    print("\n📊 Plotting feature importance...")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title('Top 20 Feature Importances - Gradient Boosting Bitcoin Model')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('dataset/plots', exist_ok=True)
    plt.savefig('dataset/plots/sklearn_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved feature importance plot to dataset/plots/sklearn_feature_importance.png")
    
    return importance_df

def plot_training_curve(model):
    """Plot training curve"""
    print("\n📈 Plotting training curve...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot training deviance
    train_scores = model.train_score_
    test_scores = model.validation_score_ if hasattr(model, 'validation_score_') else None
    
    plt.plot(range(1, len(train_scores) + 1), train_scores, 'b-', label='Training Score')
    if test_scores is not None:
        plt.plot(range(1, len(test_scores) + 1), test_scores, 'r-', label='Validation Score')
    
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Gradient Boosting Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('dataset/plots/sklearn_training_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training curve to dataset/plots/sklearn_training_curve.png")

def plot_predictions(y_test, y_test_pred):
    """Plot actual vs predicted values"""
    print("\n📈 Plotting predictions...")
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Actual vs Predicted scatter
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Actual vs Predicted Returns')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Time series of last 100 predictions
    plt.subplot(2, 2, 2)
    last_n = min(100, len(y_test))
    indices = range(len(y_test) - last_n, len(y_test))
    plt.plot(indices, y_test[-last_n:], label='Actual', linewidth=2)
    plt.plot(indices, y_test_pred[-last_n:], label='Predicted', linewidth=2)
    plt.xlabel('Time Index')
    plt.ylabel('Returns')
    plt.title(f'Last {last_n} Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Residuals
    plt.subplot(2, 2, 3)
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Returns')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Residuals histogram
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('dataset/plots/sklearn_predictions_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved predictions analysis to dataset/plots/sklearn_predictions_analysis.png")

def save_model_and_results(model, scaler, feature_cols, train_metrics, test_metrics, importance_df):
    """Save model and results"""
    print("\n💾 Saving model and results...")
    
    # Save model and scaler
    model_dir = "dataset/models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'sklearn_gradient_boosting_model.joblib')
    scaler_path = os.path.join(model_dir, 'sklearn_scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"✓ Saved Gradient Boosting model to {model_path}")
    print(f"✓ Saved scaler to {scaler_path}")
    
    # Save feature names
    feature_path = os.path.join(model_dir, 'sklearn_feature_names.txt')
    with open(feature_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    print(f"✓ Saved feature names to {feature_path}")
    
    # Save results summary
    results_path = os.path.join(model_dir, 'sklearn_results_summary.txt')
    with open(results_path, 'w') as f:
        f.write("Scikit-learn Gradient Boosting Bitcoin Analysis Results\n")
        f.write("=" * 55 + "\n\n")
        
        f.write("Model Parameters:\n")
        f.write(f"  n_estimators: {model.n_estimators}\n")
        f.write(f"  learning_rate: {model.learning_rate}\n")
        f.write(f"  max_depth: {model.max_depth}\n")
        f.write(f"  subsample: {model.subsample}\n")
        f.write(f"  max_features: {model.max_features}\n\n")
        
        f.write("Training Metrics:\n")
        for metric, value in train_metrics.items():
            f.write(f"  {metric}: {value:.6f}\n")
        
        f.write("\nTest Metrics:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.6f}\n")
        
        f.write(f"\nTop 10 Feature Importances:\n")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            f.write(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:.6f}\n")
    
    print(f"✓ Saved results summary to {results_path}")

def main():
    """Main execution function"""
    print("🚀 Scikit-learn Gradient Boosting Bitcoin Analysis")
    print("=" * 60)
    
    # Load data
    df = load_processed_data()
    if df is None:
        return
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Analyze features
    correlations = analyze_features(df, feature_cols)
    
    # Split data (time series split - no shuffling)
    print("\n✂️ Splitting data...")
    split_idx = int(0.8 * len(X))
    val_split_idx = int(0.9 * len(X))
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:val_split_idx]
    y_val = y[split_idx:val_split_idx]
    X_test = X[val_split_idx:]
    y_test = y[val_split_idx:]
    
    print(f"✓ Training set:   {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples") 
    print(f"✓ Test set:       {X_test.shape[0]} samples")
    
    # Train model
    model, scaler = train_gradient_boosting_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    y_train_pred, y_test_pred, train_metrics, test_metrics = evaluate_model(
        model, scaler, X_test, y_test, X_train, y_train
    )
    
    # Feature importance analysis
    importance_df = plot_feature_importance(model, feature_cols)
    
    # Plot training curve
    plot_training_curve(model)
    
    # Plot predictions
    plot_predictions(y_test, y_test_pred)
    
    # Save everything
    save_model_and_results(model, scaler, feature_cols, train_metrics, test_metrics, importance_df)
    
    print("\n🎉 Analysis completed successfully!")
    print("\nFiles generated:")
    print("  • dataset/models/ - Trained Gradient Boosting model & scaler")
    print("  • dataset/plots/ - Visualization plots")
    print("  • dataset/models/sklearn_results_summary.txt - Performance summary")
    
    # Final summary
    print(f"\n📊 Final Model Performance:")
    print(f"  Test R²:    {test_metrics['R²']:.4f}")
    print(f"  Test RMSE:  {test_metrics['RMSE']:.6f}")
    print(f"  Test MAE:   {test_metrics['MAE']:.6f}")

if __name__ == "__main__":
    main() 