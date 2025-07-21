"""
Pipeline‑friendly model‑training script.
Reads flat ZIPs, trains toy NN, writes Keras model + metadata, registers in local (SQLite) MLMD.
"""
import os, sys, logging, argparse, glob, zipfile, json, time
from pathlib import Path
import pandas as pd, numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────── utils ───────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Directory where the trained model and metadata must be written")
    return p.parse_args()

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

# ---------- MLMD helper  (SQLite, no MySQL) ----------
def register_model_in_mlmd(model_uri: str, model_metadata: dict):
    """Register a model artifact in an on‑disk SQLite MLMD store."""
    try:
        from ml_metadata.metadata_store import metadata_store
        from ml_metadata.proto import metadata_store_pb2

        logger.info("Registering model in local SQLite MLMD...")

        cfg = metadata_store_pb2.ConnectionConfig()
        cfg.sqlite.filename_uri = str(Path(model_uri).parent / "metadata.db")
        store = metadata_store.MetadataStore(cfg)

        # declare (or reuse) the ArtifactType
        model_type = metadata_store_pb2.ArtifactType(
            name="Model",
            properties={
                "framework": metadata_store_pb2.STRING,
                "model_type": metadata_store_pb2.STRING,
                "accuracy":   metadata_store_pb2.DOUBLE,
                "n_classes":  metadata_store_pb2.INT,
            }
        )
        type_id = store.put_artifact_type(model_type)

        art = metadata_store_pb2.Artifact(
            uri=model_uri,
            type_id=type_id,
            properties={
                "framework":  metadata_store_pb2.Value(string_value=model_metadata["framework"]),
                "model_type": metadata_store_pb2.Value(string_value=model_metadata["model_type"]),
                "accuracy":   metadata_store_pb2.Value(double_value=model_metadata["train_accuracy"]),
                "n_classes":  metadata_store_pb2.Value(int_value=model_metadata["n_classes"]),
            }
        )
        artifact_id = store.put_artifacts([art])[0]
        logger.info(f"Model registered in MLMD (ID={artifact_id})")
        return artifact_id
    except Exception as e:
        logger.warning(f"MLMD registration failed: {e}")
        return None

# ─────────── main ───────────
def main():
    args = parse_args()
    model_output = Path(args.model_path)
    model_output.mkdir(parents=True, exist_ok=True)

    # Paths relative to this script:
    data_path   = Path(__file__).parent / "data"
    local_out   = Path(__file__).parent.parent / "output"
    local_out.mkdir(exist_ok=True)

    # read data
    X, Y = read_zip_files_flat(data_path, ["f3"])
    if not X:
        logger.error("No data found, aborting.")
        return False

    X_pad   = add_padding_simple(X)
    Y_onehot, class_names = encode_labels_simple(Y)

    # simple split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_pad, Y_onehot, test_size=0.2, random_state=42)

    # train toy model
    model = simple_neural_network(X_tr.shape[1:], Y_onehot.shape[1])
    history = model.fit(X_tr, y_tr, epochs=5, batch_size=32,
                        validation_data=(X_te, y_te), verbose=1)

    train_acc = float(history.history["accuracy"][-1])
    test_acc  = float(history.history["val_accuracy"][-1])

    # save model
    keras_path = model_output / "model.keras"
    model.save(keras_path)
    model.save(local_out / "simple_model.keras")

    # No SavedModel for now - keeping it simple

    # metadata
    metadata = {
        "framework": "tensorflow",
        "model_type": "neural_network",
        "n_classes": Y_onehot.shape[1],
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "class_names": list(class_names),
        "timestamp": int(time.time())
    }
    (model_output / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    # MLMD (SQLite)
    mlmd_id = register_model_in_mlmd(str(keras_path), metadata)
    if mlmd_id:
        metadata["mlmd_id"] = mlmd_id

    logger.info("Pipeline completed OK")
    return True

if __name__ == "__main__":
    ok = main()
    if not ok:
        sys.exit(1)