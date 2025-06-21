import os
import numpy as np
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper
from helpers.inference_helper import load_random_inference_data, preprocess_for_inference

# Try to import MLflow - if not available, disable MLflow features
try:
    import mlflow
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

def load_model_from_mlflow(model_arch, logger):
    """Attempt to load model from MLflow using pyfunc. Returns None if not successful."""
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not available, skipping MLflow model loading")
        return None

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

        # Try to load from registered model first
        try:
            model_name = f"{model_arch}_FailurePrediction"
            client = mlflow.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"

            logger.info(f"Loading registered model: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Successfully loaded registered model version: {latest_version.version}")
            return model

        except Exception as registered_error:
            logger.info(f"Registered model not found: {registered_error}")

            # Fallback to searching recent runs
            experiment = mlflow.get_experiment_by_name("Default")
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName LIKE '%{model_arch}%'",
                    order_by=["start_time DESC"],
                    max_results=1
                )

                if len(runs) > 0:
                    run_id = runs.iloc[0]['run_id']  # Access as dict instead of pandas
                    model_uri = f"runs:/{run_id}/model"

                    logger.info(f"Loading model from run: {model_uri}")
                    model = mlflow.pyfunc.load_model(model_uri)
                    logger.info(f"Successfully loaded model from MLflow run: {run_id}")
                    return model

        logger.info("No suitable MLflow models found")
        return None

    except Exception as e:
        logger.warning(f"MLflow model loading failed: {e}")
        return None

def run_inference():
    """Main function to run the model inference pipeline."""
    # --- 0. INITIALIZATION ---
    LoggerHelper.init_logger()
    logger = logging.getLogger(__name__)
    config = ConfigHelper.instance("models")

    # Define paths
    SRC_ROOT = os.path.dirname(__file__)
    DATA_ROOT = os.path.join(SRC_ROOT, "..", "data")
    OUTPUT_ROOT = os.path.join(SRC_ROOT, "output")

    # --- 1. LOAD A TRAINED MODEL ---
    MODEL_ARCH = "NeuralNetwork"  # Can be changed to CNN, RNN, or LSTM
    model = None
    model_source = "unknown"

    # First try MLflow if available
    if MLFLOW_AVAILABLE:
        logger.info("Attempting to load model from MLflow...")
        model = load_model_from_mlflow(MODEL_ARCH, logger)
        if model is not None:
            model_source = "MLflow"

    # Fallback to local model loading
    if model is None:
        logger.info("Loading model from local file system...")
        try:
            import tensorflow as tf
            model_folder = os.path.join(OUTPUT_ROOT, config[MODEL_ARCH]["name_parameters"]["folder_name"])
            model_name = config[MODEL_ARCH]["name_parameters"]["model_name"].replace(".keras", "_feast.keras")
            model_path = os.path.join(model_folder, model_name)

            if not os.path.exists(model_path):
                logger.error(f"Model not found at: {model_path}")
                return

            logger.info(f"Loading model from: {model_path}")
            model = tf.keras.models.load_model(model_path)
            model_source = "Local file"

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return

    # --- 2. LOAD AND PREPROCESS DATA ---
    try:
        # Use the helper to load a random file
        inference_df, file_name, actual_class_str = load_random_inference_data(DATA_ROOT)

        # Use the helper to preprocess the data
        FEATURE_COLUMNS = ["f3_current", "f3_rolling_mean_10", "f3_rolling_std_10"]
        SEQ_LEN = 10  # Must match the training sequence length
        X_inference = preprocess_for_inference(inference_df, FEATURE_COLUMNS, SEQ_LEN)

        if X_inference.shape[0] == 0:
            logger.error("Could not create any sequences from the inference data.")
            return

    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        logger.error(f"Error during data loading or preprocessing: {e}")
        return

    # --- 3. RUN INFERENCE ---
    logger.info(f"Running inference on {X_inference.shape[0]} sequences...")
    logger.info(f"Model loaded from: {model_source}")

    # if the model expects 2D inputs (e.g. NeuralNetwork), flatten the last two dims
    if len(model.input_shape) == 2 and X_inference.ndim == 3:
        batch, sl, nf = X_inference.shape
        X_inference = X_inference.reshape((batch, sl * nf))
        logger.info(f"Reshaped inference data to {X_inference.shape} to match model input {model.input_shape}")

    predictions = model.predict(X_inference)
    predicted_classes = np.argmax(predictions, axis=1)

    # --- 4. EVALUATE AND DISPLAY RESULTS ---
    LABEL_MAP = {0: "normal", 1: "mechanical_anomaly", 2: "electrical_anomaly"}

    # Get the most predicted class (fix numpy int issue)
    predicted_class_int = int(np.bincount(predicted_classes).argmax())
    predicted_class_str = LABEL_MAP.get(predicted_class_int, "unknown")

    logger.info(f"\n--- INFERENCE RESULTS ---")
    logger.info(f"Data file used: {file_name}")
    logger.info(f"Actual Anomaly Type: {actual_class_str}")
    logger.info(f"Predicted Anomaly Type (most frequent): {predicted_class_str}")

    # Calculate sequence-level accuracy for the file
    label_to_int = {v: k for k, v in LABEL_MAP.items()}
    actual_class_int = label_to_int.get(actual_class_str)

    if actual_class_int is not None:
        num_correct = np.sum(predicted_classes == actual_class_int)
        accuracy = num_correct / len(predicted_classes)
        logger.info(f"Sequence-level accuracy on this file: {accuracy:.2%}")
    else:
        logger.warning("Could not determine the integer value for the actual class.")

    bincount = np.bincount(predicted_classes, minlength=len(LABEL_MAP))
    dist_dict = {LABEL_MAP[i]: count for i, count in enumerate(bincount)}
    logger.info(f"Distribution of predicted classes: {dist_dict}")

if __name__ == "__main__":
    run_inference()
