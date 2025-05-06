import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import parallel_backend
from joblib import parallel_backend

# Optimize CPU performance since GPU acceleration isn't available
import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

print(f"Optimizing CPU performance using {os.cpu_count()} cores")

def load_data(filepath, target_column, features=None, test_size=0.2, random_state=42, max_samples=None):
    """
    Load and split dataset for SVM training
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the dataset
    target_column : str
        Name of the column containing the target labels
    features : list or None
        List of feature column names to use, if None uses all columns except target
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    max_samples : int or None
        If specified, limit the number of samples to use (for faster training)
        
    Returns:
    --------
    Training and test data with fitted scaler
    """
    start_time = time.time()
    print(f"Loading dataset from {filepath}...")
    
    # Load data
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None
    
    # If we want to limit the number of samples (for faster training)
    if max_samples is not None and max_samples < data.shape[0]:
        # Make sure we maintain class distribution when sampling
        fraud_samples = data[data[target_column] == 1]
        normal_samples = data[data[target_column] == 0]
        
        # Calculate how many samples to take from each class
        fraud_ratio = len(fraud_samples) / len(data)
        fraud_count = int(max_samples * fraud_ratio)
        normal_count = max_samples - fraud_count
        
        # Sample from each class
        if len(normal_samples) > normal_count:
            normal_samples = normal_samples.sample(n=normal_count, random_state=random_state)
        
        if len(fraud_samples) > fraud_count:
            fraud_samples = fraud_samples.sample(n=fraud_count, random_state=random_state)
        
        # Combine samples
        data = pd.concat([normal_samples, fraud_samples])
        print(f"Using {len(data)} samples ({fraud_count} fraud, {normal_count} normal)")
    
    # Separate features and target
    y = data[target_column]
    
    if features is None:
        X = data.drop(target_column, axis=1)
    else:
        X = data[features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    load_time = time.time() - start_time
    print(f"Data loaded and preprocessed in {load_time:.2f} seconds")
    print(f"Training set size: {X_train_scaled.shape[0]}, Testing set size: {X_test_scaled.shape[0]}")
    print(f"Number of features: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale',
             probability=True, class_weight='balanced', random_state=42):
    """
    Train SVM model with specified parameters
    """
    print(f"Training SVM with kernel={kernel}, C={C}, gamma={gamma}")
    start_time = time.time()
    
    # Create the SVM model
    clf = svm.SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=probability,
        class_weight=class_weight,
        random_state=random_state,
        verbose=True,
        cache_size=2000  # MB, increase for faster training with more memory
    )
    
    # Train the model with parallelization
    with parallel_backend('threading', n_jobs=-1):
        clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"SVM training completed in {train_time:.2f} seconds")
    
    return clf

def evaluate_fraud_detection_model(model, X_test, y_test):
    """
    Evaluate SVM model for fraud detection with specialized metrics
    """
    start_time = time.time()
    print("Evaluating fraud detection model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate decision scores
    if hasattr(model, 'decision_function'):
        y_scores = model.decision_function(X_test)
    else:
        try:
            y_scores = model.predict_proba(X_test)[:, 1]
        except:
            y_scores = y_pred
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate AUPRC (Area Under Precision-Recall Curve)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    auprc = auc(recall, precision)
    avg_precision = average_precision_score(y_test, y_scores)
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except:
        roc_auc = None
    
    # Print evaluation results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")
    if roc_auc:
        print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Prepare metrics dictionary for saving
    metrics = {
        'accuracy': float(accuracy),
        'auprc': float(auprc),
        'average_precision': float(avg_precision),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    if roc_auc:
        metrics['roc_auc'] = float(roc_auc)
    
    # Calculate threshold metrics
    threshold_data = calculate_threshold_metrics(y_test, y_scores)
    
    return metrics, y_scores, threshold_data

def calculate_threshold_metrics(y_test, y_scores):
    """
    Calculate metrics across different decision thresholds
    """
    # Define thresholds
    threshold_values = np.linspace(min(y_scores), max(y_scores), 100)
    precision_values = []
    recall_values = []
    f1_values = []
    
    # Calculate metrics for each threshold
    for threshold in threshold_values:
        y_pred_t = (y_scores >= threshold).astype(int)
        precision_values.append(precision_score(y_test, y_pred_t, zero_division=0))
        recall_values.append(recall_score(y_test, y_pred_t))
        f1_values.append(f1_score(y_test, y_pred_t))
    
    # Create data dictionary
    threshold_data = {
        'thresholds': threshold_values.tolist(),
        'precision': precision_values,
        'recall': recall_values,
        'f1': f1_values
    }
    
    return threshold_data

def save_model_with_metrics(model, scaler, filepath, X_test, y_test, y_scores, metrics, 
                           feature_names, target_names, threshold_data):
    """
    Save trained SVM model with comprehensive metrics for fraud detection
    """
    start_time = time.time()
    print(f"Saving model to {filepath}...")
    
    # Prepare model data
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_type': 'SVM_FraudDetection',
        'parameters': model.get_params(),
        'date_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feature_names': feature_names,
        'target_names': target_names,
        'metrics': metrics,
        'threshold_analysis': threshold_data
    }
    
    # Store a subset of predictions for visualization
    max_samples = min(1000, len(X_test))  # Limit sample size
    indices = np.random.choice(len(X_test), max_samples, replace=False)
    
    # Convert to list for serialization
    X_test_sample = X_test[indices].tolist() if isinstance(X_test, np.ndarray) else X_test.iloc[indices].values.tolist()
    y_test_sample = y_test.iloc[indices].tolist() if isinstance(y_test, pd.Series) else y_test[indices].tolist()
    y_scores_sample = y_scores[indices].tolist() if isinstance(y_scores, np.ndarray) else y_scores[indices]
    
    model_data['sample_data'] = {
        'X_test_sample': X_test_sample,
        'y_test_sample': y_test_sample,
        'y_scores_sample': y_scores_sample
    }
    
    # Support vectors visualization data
    if hasattr(model, 'support_vectors_'):
        sv_sample_size = min(100, model.support_vectors_.shape[0])
        model_data['support_vectors'] = model.support_vectors_[:sv_sample_size].tolist()
    
    # Save model data
    joblib.dump(model_data, filepath)
    
    # Verify the file was created
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath) / 1024
        print(f"Successfully saved model file: {filepath} ({file_size:.2f} KB)")
        
        # Create a metadata JSON file
        metadata_path = os.path.splitext(filepath)[0] + '_metadata.json'
        
        # Copy model_data without the actual model and scaler
        metadata = {k: v for k, v in model_data.items() if k not in ['model', 'scaler']}
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata also saved to {metadata_path} for easier inspection")
    else:
        print(f"Warning: Failed to verify model file at {filepath}")
    
    save_time = time.time() - start_time
    print(f"Model saving completed in {save_time:.2f} seconds")

def main():
    """
    Main function to run the optimized SVM training pipeline for credit card fraud detection
    """
    total_start_time = time.time()
    print("Starting optimized SVM training for fraud detection")
    
    # Set parameters for credit card fraud detection dataset
    data_path = "svm/train_data.csv"  # Path to credit card dataset
    target_column = "Class"           # Target column (1=fraud, 0=normal)
    model_output_path = "svm/fraud_detection_model.pkl"
    max_samples = None  # Set to a number to limit dataset size, or None to use all data
    
    # Inspect dataset
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset has {df.shape[0]} transactions and {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Check class imbalance
        fraud_count = df[df[target_column] == 1].shape[0]
        normal_count = df[df[target_column] == 0].shape[0]
        imbalance_ratio = fraud_count / df.shape[0] * 100
        
        print(f"\nClass distribution:")
        print(f"  Normal transactions: {normal_count} ({100-imbalance_ratio:.3f}%)")
        print(f"  Fraudulent transactions: {fraud_count} ({imbalance_ratio:.3f}%)")
        print(f"  Imbalance ratio: 1:{normal_count/fraud_count:.1f}")
        
        # Define features - exclude 'Time' as it's not useful for prediction
        if 'Time' in df.columns:
            feature_columns = [col for col in df.columns if col != target_column and col != 'Time']
            print(f"\nExcluding 'Time' column from features")
        else:
            feature_columns = [col for col in df.columns if col != target_column]
        
        print(f"Using {len(feature_columns)} features for training")
        feature_names = feature_columns
        target_names = ['Normal Transaction', 'Fraudulent Transaction']
        
    except Exception as e:
        print(f"Error inspecting dataset: {e}")
        return
    
    # Load and preprocess data with optional sample limitation
    data_result = load_data(data_path, target_column, feature_columns, max_samples=max_samples)
    
    if data_result[0] is None:
        print("Failed to load data. Exiting.")
        return
    
    X_train, X_test, y_train, y_test, scaler = data_result
    
    # Train SVM model with optimized parameters for fraud detection
    print("\nTraining SVM model with fraud detection optimizations...")
    model = train_svm(
        X_train, 
        y_train,
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    
    # Evaluate model
    metrics, y_scores, threshold_data = evaluate_fraud_detection_model(
        model, X_test, y_test
    )
    
    # Save model with metrics
    save_model_with_metrics(
        model=model,
        scaler=scaler,
        filepath=model_output_path,
        X_test=X_test,
        y_test=y_test,
        y_scores=y_scores,
        metrics=metrics,
        feature_names=feature_names,
        target_names=target_names,
        threshold_data=threshold_data
    )
    
    total_time = time.time() - total_start_time
    print(f"\nComplete fraud detection pipeline executed in {total_time:.2f} seconds")
    print(f"Model saved to: {model_output_path}")
    print("The model includes comprehensive metrics and threshold analysis for fraud detection visualization with ShowMetrics")
    print("\nTip: If training is too slow, you can set max_samples to a smaller number in the main() function")

if __name__ == "__main__":
    main()
