#!/usr/bin/env python3
"""
Pipeline-friendly version of main.py for Kubeflow
Simplified to work in containerized environments
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(data_path, indicator_list):
    """Read data from CSV files"""
    logger.info(f"Reading data from {data_path}")

    X = []
    Y = []

    # Look for CSV files in the data directory
    data_path = Path(data_path)
    csv_files = list(data_path.glob("*.csv"))

    logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    # For this simplified version, assume we have one main dataset
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {csv_file.name} with shape {df.shape}")

            # Extract features (assuming numeric columns except last one is target)
            if len(df.columns) > 1:
                feature_cols = df.columns[:-1]  # All but last column
                target_col = df.columns[-1]     # Last column as target

                # For time series data, we'll treat each row as a sample
                for _, row in df.iterrows():
                    X.append(row[feature_cols].values)
                    Y.append(row[target_col])

        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
            continue

    if not X:
        raise ValueError("No valid data found in CSV files")

    logger.info(f"Loaded {len(X)} samples")
    return X, Y

def add_padding(X, indicator_list):
    """Add padding to make all sequences the same length"""
    if not X:
        return np.array([])

    # Convert to numpy arrays
    X_arrays = [np.array(x) for x in X]

    # Find max length
    max_len = max(len(x) for x in X_arrays)

    # Pad sequences
    X_padded = []
    for x in X_arrays:
        if len(x) < max_len:
            padding = np.zeros(max_len - len(x))
            x_padded = np.concatenate([x, padding])
        else:
            x_padded = x
        X_padded.append(x_padded)

    return np.array(X_padded)

def encode_response_variable(Y):
    """Encode response variable"""
    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)

    # Convert to one-hot encoding
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False)
    Y_encoded = onehot_encoder.fit_transform(Y_encoded.reshape(-1, 1))

    return Y_encoded

def split_data(X, Y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

def train_simple_model(X_train, y_train):
    """Train a simple RandomForest model"""
    logger.info("Training RandomForest model...")

    # For this simplified version, flatten the data if needed
    if len(X_train.shape) > 2:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
    else:
        X_train_flat = X_train

    # Convert one-hot encoded y to single labels
    if len(y_train.shape) > 1:
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        y_train_labels = y_train

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_flat, y_train_labels)

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    logger.info("Evaluating model...")

    # Flatten test data if needed
    if len(X_test.shape) > 2:
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_test_flat = X_test

    # Convert one-hot encoded y to single labels
    if len(y_test.shape) > 1:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test

    y_pred = model.predict(X_test_flat)

    accuracy = accuracy_score(y_test_labels, y_pred)
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test_labels, y_pred))

    return accuracy

def main():
    """Main pipeline function"""
    logger.info("Starting pipeline-friendly main...")

    # Define paths
    data_path = Path("data")  # Expect data in ./data directory
    output_path = Path("output")
    output_path.mkdir(exist_ok=True)

    # Check if data directory exists
    if not data_path.exists():
        logger.error(f"Data directory {data_path} not found")
        sys.exit(1)

    # List of indicators (simplified)
    indicator_list = ["f3"]  # Can be configured as needed

    try:
        # Read data
        X, Y = read_data(data_path, indicator_list)
        logger.info(f"Data shape: X={len(X)}, Y={len(Y)}")

        # Add padding
        X_pad = add_padding(X, indicator_list)
        logger.info(f"Padded X shape: {X_pad.shape}")

        # Encode response variable
        Y_encoded = encode_response_variable(Y)
        logger.info(f"Encoded Y shape: {Y_encoded.shape}")

        # Split data
        X_train, X_test, y_train, y_test = split_data(X_pad, Y_encoded)
        logger.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")

        # Train model
        model = train_simple_model(X_train, y_train)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)

        # Save model
        model_path = output_path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

        # Save results
        results = {
            'accuracy': accuracy,
            'n_samples': len(X),
            'n_features': X_pad.shape[1] if len(X_pad.shape) > 1 else 1,
            'n_classes': Y_encoded.shape[1] if len(Y_encoded.shape) > 1 else 1
        }

        results_path = output_path / "results.txt"
        with open(results_path, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"Results saved to {results_path}")
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()