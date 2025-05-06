import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
from joblib import dump
import os

# Create directory for plots
plots_dir = 'training_plots'
os.makedirs(plots_dir, exist_ok=True)

# Load pre-split data
print("Loading data...")
train_df = pd.read_csv('Dataset_1/train_data.csv')
val_df = pd.read_csv('Dataset_1/validation_data.csv')
test_df = pd.read_csv('Dataset_1/test_data.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Separate features and target
feature_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
X_train = train_df[feature_columns]
y_train = train_df['Class']
X_val = val_df[feature_columns]
y_val = val_df['Class']
X_test = test_df[feature_columns]
y_test = test_df['Class']

# Scale Time and Amount features
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_val[['Time', 'Amount']] = scaler.transform(X_val[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Train XGBoost
print("\nTraining XGBoost...")
start_time = time.time()

# Set training parameters
n_epochs = 150           # Total number of epochs (n_estimators)
verbose_freq = 1        # Show progress every 10 epochs
early_stopping = 40      # Stop if no improvement for 20 rounds

print(f"Training for {n_epochs} epochs...")
print(f"Progress will be shown every {verbose_freq} epochs")
print(f"Early stopping patience: {early_stopping} rounds")

model = xgb.XGBClassifier(
    random_state=42,
    n_estimators=n_epochs,          # Number of epochs
    eval_metric=['aucpr', 'logloss'],
    early_stopping_rounds=early_stopping
)

# Fit model
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=verbose_freq    # Print progress every verbose_freq epochs
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"Best iteration: {model.best_iteration}")
print(f"Actual epochs completed: {model.best_iteration + 1}")

# Save model and metadata
save_dict = {
    'model': model,
    'scaler': scaler,
    'features': feature_columns,
    'training_history': model.evals_result(),
    'training_params': {
        'n_epochs': n_epochs,
        'early_stopping': early_stopping,
        'completed_epochs': model.best_iteration + 1
    }
}

model_filename = f'basic_model_{datetime.now().strftime("%Y%m%d_%H%M")}.joblib'
dump(save_dict, model_filename)
print(f"\nModel saved as: {model_filename}")
print("Now you can run ShowMetrics.py to see the detailed performance metrics and visualizations.")
