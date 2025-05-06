import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import os
from joblib import dump
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Custom callback to print epoch results
class EpochPrintCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            print(f"\nEpoch {epoch + 1}")
            print(f"Training Loss: {logs.get('loss'):.4f}")
            print(f"Training Accuracy: {logs.get('accuracy'):.4f}")
            print(f"Training AUC: {logs.get('auc'):.4f}")
            print(f"Validation Loss: {logs.get('val_loss'):.4f}")
            print(f"Validation Accuracy: {logs.get('val_accuracy'):.4f}")
            print(f"Validation AUC: {logs.get('val_auc'):.4f}")

# Create directories for outputs
plots_dir = 'training_plots'
logs_dir = 'training_logs'
models_dir = 'saved_models'
for dir_path in [plots_dir, logs_dir, models_dir]:
    os.makedirs(dir_path, exist_ok=True)

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

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Calculate class weights for imbalanced data
n_neg, n_pos = np.bincount(y_train)
class_weight = {0: 1., 1: n_neg/n_pos}
print("\nClass weights:", class_weight)

# Build model with optimized architecture
print("\nBuilding Keras model...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

# Compile model with optimized parameters
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Define callbacks with optimized parameters
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(models_dir, 'best_model.keras'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=15,  # Reduced patience
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # More aggressive LR reduction
        patience=5,   # Reduced patience for LR reduction
        min_lr=0.00001,
        verbose=1
    ),
    CSVLogger(os.path.join(logs_dir, 'training_log.csv'))
]

# Start timing the training
start_time = time.time()

# Train model with optimized parameters
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,         # Reduced max epochs
    batch_size=512,     # Increased batch size
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Calculate training time
training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Plot training history
def plot_metric(history, metric, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric], label='Training')
    plt.plot(history.history[f'val_{metric}'], label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plots_dir}/{metric}_history.png')
    plt.close()

# Generate plots
plot_metric(history, 'loss', 'Training and Validation Loss')
plot_metric(history, 'accuracy', 'Training and Validation Accuracy')
plot_metric(history, 'auc', 'Training and Validation AUC')

# Save model and metadata
save_dict = {
    'keras_model_path': os.path.join(models_dir, 'best_model.keras'),
    'scaler': scaler,
    'features': feature_columns,
    'training_history': history.history,
    'training_params': {
        'batch_size': 512,
        'class_weights': class_weight,
        'training_time': training_time,
        'architecture': model.get_config()
    }
}

model_filename = f'keras_model_{datetime.now().strftime("%Y%m%d_%H%M")}.joblib'
dump(save_dict, model_filename)

print("\nTraining Results:")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Training AUC: {history.history['auc'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Validation AUC: {history.history['val_auc'][-1]:.4f}")

print(f"\nModel saved as: {model_filename}")
print(f"Training logs saved in: {logs_dir}/")
print(f"Plots saved in: {plots_dir}/")
print(f"Best model saved in: {models_dir}/")
