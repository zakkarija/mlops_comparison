'''
Katib-optimized Pipeline script for neural network hyperparameter tuning
Handles flat directory of ZIP files from LakeFS with hyperparameter optimization
'''

import os
import sys
import zipfile
import logging
import glob
import pandas as pd
import numpy as np
import argparse
import time
import json
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments including hyperparameters for Katib"""
    parser = argparse.ArgumentParser(description='ML Pipeline for multiclass classification with Katib optimization')

    # Original arguments
    parser.add_argument('--model_output_path', type=str, default='../output',
                        help='Path to save the trained model')

    # Katib hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--hidden_units_1', type=int, default=64,
                        help='Number of units in first hidden layer')
    parser.add_argument('--hidden_units_2', type=int, default=32,
                        help='Number of units in second hidden layer')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate for regularization')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop'],
                        help='Optimizer to use')

    return parser.parse_args()

def read_zip_files_flat(data_path, indicator_list):
    """
    Read data from flat directory of ZIP files
    Infers labels from filenames or uses default labeling

    Parameters:
        data_path (str): path to directory containing ZIP files
        indicator_list (list): list of indicators to read from files

    Returns:
        X: list of arrays containing the data
        Y: list of labels
    """
    logger.info(f"Reading data from flat directory: {data_path}")
    logger.info(f"Looking for indicators: {indicator_list}")

    # Find all ZIP files
    zip_files = glob.glob(os.path.join(data_path, "*.zip"))
    logger.info(f"Found {len(zip_files)} ZIP files")

    if len(zip_files) == 0:
        logger.error("No ZIP files found in data directory")
        return [], []

    X = []
    Y = []

    # Simple labeling strategy - you may need to adjust this based on your filename patterns
    for i, zip_path in enumerate(zip_files):
        try:
            filename = os.path.basename(zip_path)
            logger.info(f"Processing file {i+1}/{len(zip_files)}: {filename}")

            # Open ZIP file and read CSV
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                # Get the first (and presumably only) file in the ZIP
                csv_filename = zip_file.namelist()[0]

                # Read CSV data
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file, delimiter=";")

                    # Check if required indicators exist
                    available_indicators = [ind for ind in indicator_list if ind in df.columns]
                    if not available_indicators:
                        logger.warning(f"No required indicators found in {filename}, using first numeric column")
                        # Use first numeric column if indicators not found
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            available_indicators = [numeric_cols[0]]
                        else:
                            logger.warning(f"No numeric columns found in {filename}, skipping")
                            continue

                    # Extract data for available indicators
                    data = df[available_indicators].to_numpy()
                    X.append(data)

                    # Simple labeling strategy - adjust as needed
                    if "anomal" in filename.lower():
                        if "electrical" in filename.lower():
                            label = "electrical anomaly"
                        elif "mechanical" in filename.lower():
                            label = "mechanical anomaly"
                        else:
                            label = "anomaly"
                    else:
                        label = "not anomalous"

                    Y.append(label)
                    logger.info(f"Added data with shape {data.shape} and label '{label}'")

        except Exception as e:
            logger.error(f"Error processing {zip_path}: {e}")
            continue

    logger.info(f"Successfully loaded {len(X)} files")
    return X, Y

def add_padding_simple(X):
    """
    Simple padding function using numpy
    """
    if not X:
        return np.array([])

    # Find maximum length
    max_length = max(len(x) for x in X)
    n_features = X[0].shape[1] if len(X) > 0 else 1

    logger.info(f"Padding sequences to max length: {max_length}")

    # Pad sequences
    X_padded = []
    for x in X:
        if len(x) < max_length:
            # Pad with zeros
            padding = np.zeros((max_length - len(x), n_features))
            x_padded = np.vstack([x, padding])
        else:
            x_padded = x
        X_padded.append(x_padded)

    return np.array(X_padded)

def encode_labels_simple(Y):
    """
    Simple label encoding
    """
    from sklearn.preprocessing import LabelEncoder

    # Encode labels to integers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(Y)

    # One-hot encode
    n_classes = len(set(Y))
    y_onehot = np.eye(n_classes)[y_encoded]

    logger.info(f"Encoded {len(Y)} labels into {n_classes} classes")
    logger.info(f"Classes: {encoder.classes_}")

    return y_onehot, encoder.classes_

def create_katib_neural_network(input_shape, n_classes, args):
    """
    Create a configurable neural network for Katib hyperparameter optimization
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Build model with hyperparameters
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(args.hidden_units_1, activation='relu'),
            layers.Dropout(args.dropout_rate),
            layers.Dense(args.hidden_units_2, activation='relu'),
            layers.Dropout(args.dropout_rate),
            layers.Dense(n_classes, activation='softmax')
        ])

        # Configure optimizer based on hyperparameter
        if args.optimizer == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=args.learning_rate)
        elif args.optimizer == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=args.learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Created Katib-optimized neural network")
        logger.info(f"Model input shape: {input_shape}")
        logger.info(f"Model output classes: {n_classes}")
        logger.info(f"Hidden units: {args.hidden_units_1}, {args.hidden_units_2}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Dropout rate: {args.dropout_rate}")
        logger.info(f"Optimizer: {args.optimizer}")

        return model

    except ImportError:
        logger.error("TensorFlow/Keras not available, skipping model creation")
        return None

def output_katib_metrics(accuracy, loss, args):
    """
    Output metrics in the format Katib expects
    Katib looks for metrics in logs with specific format
    """
    # Primary metric for Katib optimization
    print(f"accuracy={accuracy:.6f}")
    print(f"loss={loss:.6f}")

    # Additional metrics for tracking
    print(f"learning_rate={args.learning_rate}")
    print(f"batch_size={args.batch_size}")
    print(f"hidden_units_1={args.hidden_units_1}")
    print(f"hidden_units_2={args.hidden_units_2}")
    print(f"dropout_rate={args.dropout_rate}")
    print(f"optimizer={args.optimizer}")

def main():
    """
    Main pipeline function optimized for Katib
    """
    # Parse command line arguments (includes hyperparameters)
    args = parse_args()

    logger.info("Starting Katib-optimized pipeline...")
    logger.info(f"Model output path: {args.model_output_path}")
    logger.info(f"Hyperparameters: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.epochs}")
    logger.info(f"Architecture: {args.hidden_units_1}-{args.hidden_units_2}, dropout={args.dropout_rate}, optimizer={args.optimizer}")

    try:
        # Define paths
        data_path = "data"
        output_path = "../output"
        model_output_path = args.model_output_path

        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(model_output_path, exist_ok=True)

        # Check if data directory exists
        if not os.path.exists(data_path):
            logger.error(f"Data directory '{data_path}' not found")
            return False

        # List data directory contents
        data_files = os.listdir(data_path)
        logger.info(f"Data directory contains: {data_files}")

        # Check for ZIP files
        zip_files = [f for f in data_files if f.endswith('.zip')]
        logger.info(f"Found {len(zip_files)} ZIP files: {zip_files}")

        if len(zip_files) == 0:
            logger.error("No ZIP files found in data directory")
            return False

        # Define indicators to extract
        indicator_list = ["f3"]

        # Read data
        logger.info("Processing ZIP files...")
        X, Y = read_zip_files_flat(data_path, indicator_list)

        if len(X) == 0:
            logger.error("No data loaded successfully")
            return False

        logger.info(f"Loaded {len(X)} samples")

        # Add padding
        logger.info("Adding padding to sequences...")
        X_padded = add_padding_simple(X)

        if X_padded.size == 0:
            logger.error("Padding failed")
            return False

        # Encode labels
        logger.info("Encoding labels...")
        Y_encoded, class_names = encode_labels_simple(Y)

        logger.info(f"Data shape after padding: {X_padded.shape}")
        logger.info(f"Labels shape: {Y_encoded.shape}")

        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, Y_encoded, test_size=0.2, random_state=42
        )

        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")

        # Create model with hyperparameters
        input_shape = X_train.shape[1:]
        n_classes = Y_encoded.shape[1]

        model = create_katib_neural_network(input_shape, n_classes, args)

        if model is not None:
            logger.info("Training model with Katib hyperparameters...")

            # Train with hyperparameters
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=1
            )

            # Evaluate model
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            logger.info(f"Training accuracy: {train_acc:.6f}")
            logger.info(f"Test accuracy: {test_acc:.6f}")

            # Output metrics for Katib (this is crucial!)
            output_katib_metrics(test_acc, test_loss, args)

            # Save model (for best trials)
            model_path = os.path.join(model_output_path, "katib_model.keras")
            model.save(model_path)
            logger.info(f"Model saved to: {model_path}")

            # Save hyperparameters and results
            trial_results = {
                "hyperparameters": {
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "hidden_units_1": args.hidden_units_1,
                    "hidden_units_2": args.hidden_units_2,
                    "dropout_rate": args.dropout_rate,
                    "optimizer": args.optimizer
                },
                "results": {
                    "train_accuracy": float(train_acc),
                    "test_accuracy": float(test_acc),
                    "train_loss": float(train_loss),
                    "test_loss": float(test_loss)
                },
                "model_info": {
                    "input_shape": list(input_shape),
                    "n_classes": n_classes,
                    "class_names": list(class_names)
                }
            }

            with open(os.path.join(model_output_path, "trial_results.json"), "w") as f:
                json.dump(trial_results, f, indent=2)

            logger.info("Trial results saved")

        print("SUCCESS: Katib trial completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

    print("Katib trial completed successfully!")