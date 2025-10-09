#!/usr/bin/env python3
"""
XGBoost Teacher Bitcoin Analysis Script
Uses the original XGBoost teacher from src/training/xgboost_teacher.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import our modules
sys.path.append('src')

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

def train_with_xgboost_teacher(X_train, y_train, X_val, y_val):
    """Train using the original XGBoost teacher"""
    print("\n🤖 Training with XGBoost Teacher...")
    
    try:
        from training.xgboost_teacher import XGBoostTeacher
        print("✓ Successfully imported XGBoostTeacher")
        
        # XGBoost parameters optimized for financial time series
        xgb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 0.1,
            'lambda': 1.0,
            'objective': 'reg:squarederror',
            'n_jobs': 1,  # Use single thread to avoid OpenMP issues
            'random_state': 42
        }
        
        print(f"✓ Parameters: {xgb_params}")
        
        # Initialize teacher
        teacher = XGBoostTeacher(xgb_params)
        
        # Train model
        eval_set = [(X_val, y_val)]
        teacher.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=50)
        
        print("✓ XGBoost Teacher training completed successfully")
        return teacher
        
    except ImportError as e:
        print(f"❌ XGBoost import error: {e}")
        return None
    except Exception as e:
        print(f"❌ XGBoost training error: {e}")
        return None

def evaluate_teacher_model(teacher, X_test, y_test, X_train, y_train):
    """Evaluate the teacher model performance"""
    print("\n📊 Evaluating XGBoost Teacher performance...")
    
    try:
        # Get predictions
        y_train_pred = teacher.predict(X_train)
        y_test_pred = teacher.predict(X_test)
        
        # Calculate metrics using teacher's evaluate method
        train_metrics = teacher.evaluate(X_train, y_train)
        test_metrics = teacher.evaluate(X_test, y_test)
        
        print("\n📈 Training Performance:")
        for metric, value in train_metrics.items():
            print(f"  {metric.upper():<6}: {value:.6f}")
        
        print("\n📉 Test Performance:")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper():<6}: {value:.6f}")
        
        # Price prediction analysis
        print("\n💰 Price Prediction Analysis:")
        current_return = y_test[-1]
        predicted_return = y_test_pred[-1]
        
        print(f"  Current return:    {current_return:.4f} ({current_return*100:.2f}%)")
        print(f"  Predicted return:  {predicted_return:.4f} ({predicted_return*100:.2f}%)")
        print(f"  Prediction error:  {abs(current_return - predicted_return):.4f}")
        
        return y_train_pred, y_test_pred, train_metrics, test_metrics
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return None, None, {}, {}

def plot_xgboost_feature_importance(teacher, feature_cols):
    """Plot XGBoost feature importance"""
    print("\n📊 Plotting XGBoost feature importance...")
    
    try:
        # Get feature importance from XGBoost model
        importance = teacher.model.feature_importances_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top 20 Feature Importances - XGBoost Teacher Bitcoin Model')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('dataset/plots', exist_ok=True)
        plt.savefig('dataset/plots/xgboost_teacher_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved XGBoost feature importance plot")
        
        return importance_df
        
    except Exception as e:
        print(f"❌ Feature importance plotting error: {e}")
        return pd.DataFrame()

def plot_xgboost_predictions(y_test, y_test_pred):
    """Plot XGBoost predictions analysis"""
    print("\n📈 Plotting XGBoost predictions...")
    
    try:
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Actual vs Predicted scatter
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
        plt.title('XGBoost: Actual vs Predicted Returns')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Time series of last 100 predictions
        plt.subplot(2, 2, 2)
        last_n = min(100, len(y_test))
        indices = range(len(y_test) - last_n, len(y_test))
        plt.plot(indices, y_test[-last_n:], label='Actual', linewidth=2)
        plt.plot(indices, y_test_pred[-last_n:], label='Predicted', linewidth=2)
        plt.xlabel('Time Index')
        plt.ylabel('Returns')
        plt.title(f'XGBoost: Last {last_n} Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Residuals
        plt.subplot(2, 2, 3)
        residuals = y_test - y_test_pred
        plt.scatter(y_test_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Returns')
        plt.ylabel('Residuals')
        plt.title('XGBoost: Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Residuals histogram
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('XGBoost: Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('dataset/plots/xgboost_teacher_predictions_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved XGBoost predictions analysis plot")
        
    except Exception as e:
        print(f"❌ Predictions plotting error: {e}")

def save_teacher_model(teacher, feature_cols, train_metrics, test_metrics, importance_df):
    """Save XGBoost teacher model and results"""
    print("\n💾 Saving XGBoost Teacher model and results...")
    
    try:
        # Save model using teacher's save method
        model_dir = "dataset/models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the teacher model
        teacher.save(os.path.join(model_dir, 'xgboost_teacher'))
        print("✓ Saved XGBoost Teacher model")
        
        # Save feature names
        feature_path = os.path.join(model_dir, 'xgboost_teacher_feature_names.txt')
        with open(feature_path, 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        print(f"✓ Saved feature names to {feature_path}")
        
        # Save results summary
        results_path = os.path.join(model_dir, 'xgboost_teacher_results_summary.txt')
        with open(results_path, 'w') as f:
            f.write("XGBoost Teacher Bitcoin Analysis Results\n")
            f.write("=" * 45 + "\n\n")
            
            f.write("Model Parameters:\n")
            for param, value in teacher.params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            f.write("Training Metrics:\n")
            for metric, value in train_metrics.items():
                f.write(f"  {metric.upper()}: {value:.6f}\n")
            
            f.write("\nTest Metrics:\n")
            for metric, value in test_metrics.items():
                f.write(f"  {metric.upper()}: {value:.6f}\n")
            
            if not importance_df.empty:
                f.write(f"\nTop 10 Feature Importances:\n")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    f.write(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:.6f}\n")
        
        print(f"✓ Saved results summary to {results_path}")
        
    except Exception as e:
        print(f"❌ Model saving error: {e}")

def main():
    """Main execution function"""
    print("🚀 XGBoost Teacher Bitcoin Analysis")
    print("=" * 50)
    
    # Load data
    df = load_processed_data()
    if df is None:
        return
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
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
    
    # Train XGBoost teacher
    teacher = train_with_xgboost_teacher(X_train, y_train, X_val, y_val)
    
    if teacher is None:
        print("\n❌ XGBoost Teacher training failed. Please check XGBoost installation.")
        print("You can install XGBoost with: pip install xgboost")
        return
    
    # Evaluate model
    y_train_pred, y_test_pred, train_metrics, test_metrics = evaluate_teacher_model(
        teacher, X_test, y_test, X_train, y_train
    )
    
    if y_test_pred is None:
        print("\n❌ Model evaluation failed.")
        return
    
    # Feature importance analysis
    importance_df = plot_xgboost_feature_importance(teacher, feature_cols)
    
    # Plot predictions
    plot_xgboost_predictions(y_test, y_test_pred)
    
    # Save everything
    save_teacher_model(teacher, feature_cols, train_metrics, test_metrics, importance_df)
    
    print("\n🎉 XGBoost Teacher analysis completed successfully!")
    print("\nFiles generated:")
    print("  • dataset/models/xgboost_teacher/ - XGBoost Teacher model")
    print("  • dataset/plots/ - Visualization plots")
    print("  • dataset/models/xgboost_teacher_results_summary.txt - Performance summary")
    
    # Final summary
    if test_metrics:
        print(f"\n📊 Final XGBoost Teacher Performance:")
        print(f"  Test R²:    {test_metrics.get('r2', 'N/A'):.4f}")
        print(f"  Test RMSE:  {test_metrics.get('rmse', 'N/A'):.6f}")
        print(f"  Test MAE:   {test_metrics.get('mae', 'N/A'):.6f}")

if __name__ == "__main__":
    main() 