'''
Pipeline-friendly script for model training
Handles flat directory of ZIP files from LakeFS
'''

import os
import sys
import zipfile
import logging
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    # For now, we'll use a simple approach
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
                    # For now, assign labels based on file index or filename patterns
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

def simple_neural_network(input_shape, n_classes):
    """
    Create a simple neural network for demonstration
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Created simple neural network")
        logger.info(f"Model input shape: {input_shape}")
        logger.info(f"Model output classes: {n_classes}")

        return model

    except ImportError:
        logger.error("TensorFlow/Keras not available, skipping model creation")
        return None

def main():
    """
    Main pipeline function
    """
    logger.info("Starting pipeline-friendly main...")

    try:
        # Define paths (relative to src directory)
        data_path = "../data"  # Data is one level up from src
        output_path = "../output"  # Output is one level up from src

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Check if data directory exists
        if not os.path.exists(data_path):
            logger.error(f"Data directory '{data_path}' not found")
            return False

        # List data directory contents
        data_files = os.listdir(data_path)
        logger.info(f"Data directory contains: {data_files}")

        # Check for ZIP files
        zip_files = [f for f in data_files if f.endswith('.zip')]
        csv_files = [f for f in data_files if f.endswith('.csv')]

        logger.info(f"Found {len(zip_files)} ZIP files: {zip_files}")
        logger.info(f"Found {len(csv_files)} CSV files: {csv_files}")

        if len(zip_files) == 0 and len(csv_files) == 0:
            logger.error("No ZIP or CSV files found in data directory")
            return False

        # Define indicators to extract
        indicator_list = ["f3"]  # Same as original code

        # Read data
        if len(zip_files) > 0:
            logger.info("Processing ZIP files...")
            X, Y = read_zip_files_flat(data_path, indicator_list)
        else:
            logger.info("Processing CSV files...")
            # Add CSV processing logic if needed
            X, Y = [], []

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

        # Simple train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_padded, Y_encoded, test_size=0.2, random_state=42
        )

        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")

        # Create and train a simple model
        input_shape = X_train.shape[1:]  # Remove batch dimension
        n_classes = Y_encoded.shape[1]

        model = simple_neural_network(input_shape, n_classes)

        if model is not None:
            logger.info("Training model...")

            # Train for just a few epochs for demonstration
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=5,  # Short training for pipeline testing
                batch_size=32,
                verbose=1
            )

            # Evaluate model
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            logger.info(f"Training accuracy: {train_acc:.4f}")
            logger.info(f"Test accuracy: {test_acc:.4f}")

            # Save model
            model_path = os.path.join(output_path, "simple_model.keras")
            model.save(model_path)
            logger.info(f"Model saved to: {model_path}")

            # Save results summary
            results = {
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
                "n_samples": len(X),
                "n_classes": n_classes,
                "class_names": class_names.tolist(),
                "input_shape": list(input_shape)
            }

            import json
            with open(os.path.join(output_path, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

            logger.info("Results saved to results.json")

        # Create a simple summary file
        summary = f"""
Pipeline Execution Summary
=========================

Data Processing:
- Loaded {len(X)} samples from {len(zip_files)} ZIP files
- Data shape: {X_padded.shape}
- Classes: {len(set(Y))} ({', '.join(set(Y))})

Model Training:
- Input shape: {input_shape}
- Number of classes: {n_classes}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}

Results:
- Training accuracy: {train_acc:.4f} (if model trained)
- Test accuracy: {test_acc:.4f} (if model trained)

Output files created in '{output_path}' directory.
"""

        with open(os.path.join(output_path, "summary.txt"), "w") as f:
            f.write(summary)

        print("SUCCESS: Pipeline completed successfully!")
        print(f"Check the '{output_path}' directory for results")

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

    print("Pipeline completed successfully!")