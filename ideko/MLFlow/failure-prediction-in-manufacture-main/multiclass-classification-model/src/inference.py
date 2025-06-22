import os
import numpy as np
import mlflow
import mlflow.keras
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper
from helpers.inference_helper import load_random_inference_data, preprocess_for_inference

def load_model_from_mlflow(model_arch, logger):
    """Load model from MLflow using keras flavor."""
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    try:
        model_name = f"{model_arch}_FailurePrediction"
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"

        logger.info(f"Loading model: {model_uri}")
        model = mlflow.keras.load_model(model_uri)
        logger.info(f"Successfully loaded model version: {latest_version.version}")
        return model

    except Exception as e:
        logger.warning(f"MLflow model loading failed: {e}")
        return None

def run_inference():
    """Main function to run the model inference pipeline."""
    LoggerHelper.init_logger()
    logger = logging.getLogger(__name__)
    config = ConfigHelper.instance("models")

    # Get the active model from config (find which one is enabled)
    MODEL_ARCH = None
    for arch in ["NeuralNetwork", "CNN", "RNN", "LSTM"]:
        if config[arch]["enabled"]:
            MODEL_ARCH = arch
            break

    if MODEL_ARCH is None:
        logger.error("No model is enabled in config. Please enable one model in models.yaml")
        return

    logger.info(f"Using model architecture: {MODEL_ARCH}")

    # Start MLflow tracking for inference
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Enable system metrics for inference runs
    mlflow.enable_system_metrics_logging()

    # Define paths
    SRC_ROOT = os.path.dirname(__file__)
    DATA_ROOT = os.path.join(SRC_ROOT, "..", "data")
    OUTPUT_ROOT = os.path.join(SRC_ROOT, "output")

    # Load model from MLflow
    model = load_model_from_mlflow(MODEL_ARCH, logger)
    if model is None:
        logger.error("Could not load model from MLflow")
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
    with mlflow.start_run(run_name=f"{MODEL_ARCH}_inference"):
        logger.info(f"Running inference on {X_inference.shape[0]} sequences...")

        # Add tracing for inference pipeline
        with mlflow.start_span(name="data_preprocessing") as span:
            # if the model expects 2D inputs (e.g. NeuralNetwork), flatten the last two dims
            if len(model.input_shape) == 2 and X_inference.ndim == 3:
                batch, sl, nf = X_inference.shape
                X_inference = X_inference.reshape((batch, sl * nf))
                logger.info(f"Reshaped inference data to {X_inference.shape} to match model input {model.input_shape}")
            span.set_attribute("input_shape", str(X_inference.shape))
            span.set_attribute("model_input_shape", str(model.input_shape))

        with mlflow.start_span(name="model_prediction") as span:
            predictions = model.predict(X_inference)
            span.set_attribute("num_predictions", len(predictions))
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
            accuracy = 0.0

        bincount = np.bincount(predicted_classes, minlength=len(LABEL_MAP))
        dist_dict = {LABEL_MAP[i]: count for i, count in enumerate(bincount)}
        logger.info(f"Distribution of predicted classes: {dist_dict}")

        # Log inference results to MLflow
        mlflow.log_params({
            "model_architecture": MODEL_ARCH,
            "inference_file": file_name,
            "actual_class": actual_class_str,
            "predicted_class": predicted_class_str
        })

        mlflow.log_metrics({
            "inference_accuracy": accuracy,
            "num_sequences": X_inference.shape[0]
        })

if __name__ == "__main__":
    run_inference()
