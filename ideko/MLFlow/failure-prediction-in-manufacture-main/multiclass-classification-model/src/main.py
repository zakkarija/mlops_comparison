import os, gc, numpy as np, pandas as pd
from feast import FeatureStore
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper
from helpers.data_helper import DataHelper
from classes import preprocessing_functions
import mlflow
import mlflow.keras
from classes.multiclass_models import (
    NeuralNetwork,
    ConvolutionalNeuralNetwork,
    RecurrentNeuralNetwork,
    LongShortTermMemory,
)

# Initialize logging and configuration
LoggerHelper.init_logger()
logger = logging.getLogger(__name__)
config = ConfigHelper.instance("models")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Helper function to check class distribution at different stages
def _check(label_arr, stage):
    uniques, counts = np.unique(label_arr, return_counts=True)
    logger.info(f"[CHECK] {stage}: {dict(zip(uniques, counts))}")

# === MLFLOW TRACKING SETUP ===
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# === 1 · FEAST DATA RETRIEVAL ===
FEAST_REPO = "feast_demo/feature_repo"
FEATURE_COLUMNS = ["f3_current", "f3_rolling_mean_10", "f3_rolling_std_10"]
LABEL_FEATURE = "f3_timeseries_features:anomaly_class"

# Read entity_df for Feast queries
logger.info("Reading entity DataFrame for Feast queries")
timeseries_df = pd.read_parquet(
    f"{FEAST_REPO}/data/offline/f3_timeseries.parquet",
    columns=["equipment_id", "event_timestamp"]
)

timeseries_df = timeseries_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

entity_df = timeseries_df.copy(); del timeseries_df

# Use DataHelper to batch read from Feast with memory optimization
training_df = DataHelper.batch_read_feast_data(
    feast_repo_path=FEAST_REPO,
    entity_df=entity_df,
    feature_columns=FEATURE_COLUMNS,
    label_feature=LABEL_FEATURE,
    batch_size=2000,
    max_batches=20
)
del entity_df; gc.collect()

# Check class distribution after Feast data pull
_check(training_df['anomaly_class'].values, "raw Feast pull")

# Verify data has multiple classes
class_counts = training_df["anomaly_class"].value_counts()
if len(class_counts) < 2:
    logger.error(f"WARNING: Only found {len(class_counts)} classes in data: {class_counts.to_dict()}")
    logger.error("This will result in a model that only predicts one class.")
    logger.error("Check your Feast feature definitions and label extraction.")

# === 2 · TRAIN/TEST SPLIT ===
logger.info("Splitting data into train and test sets")
train_df, test_df = DataHelper.stratified_equipment_split(
    df=training_df,
    test_size=0.2,
    random_state=123
)

# Check class distribution after train/test split
_check(train_df['anomaly_class'].values, "after train/test split (train)")
_check(test_df['anomaly_class'].values, "after train/test split (test)")

# === 3 · WINDOW BUILDING ===
logger.info("Building sequences for time series modeling")
X_train, y_train_int = DataHelper.build_sequences(
    df=train_df,
    feature_columns=FEATURE_COLUMNS,
    seq_len=10,
    stride=2
)
X_test, y_test_int = DataHelper.build_sequences(
    df=test_df,
    feature_columns=FEATURE_COLUMNS,
    seq_len=10,
    stride=2
)

# Check class distribution after window building
_check(y_train_int, "after window build (train)")
_check(y_test_int, "after window build (test)")

# === 4 · LABEL ENCODING ===
# Define label mapping consistent with original implementation
LABEL_MAP = {0:"normal", 1:"mechanical_anomaly", 2:"electrical_anomaly"}

# Convert integer labels to text labels
logger.info("Encoding labels")
y_train_txt = DataHelper.encode_labels(y_train_int, LABEL_MAP)
y_test_txt = DataHelper.encode_labels(y_test_int, LABEL_MAP)

# One-hot encode the labels
# First using DataHelper for consistency and to ensure we have all classes represented
Y_train = DataHelper.encode_response_variable(y_train_txt)
Y_test = DataHelper.encode_response_variable(y_test_txt)

# Check shapes and verify we have multiple classes
logger.info(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
logger.info(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

N_TS = X_train.shape[1]  # sequence length
N_FEAT = X_train.shape[2]  # number of features
N_CLS = Y_train.shape[1]  # number of classes

logger.info(f"Sequence length: {N_TS}, Features: {N_FEAT}, Classes: {N_CLS}")

if N_CLS <= 1:
    logger.error("ERROR: Only one class is present in the encoded labels.")
    logger.error("This will cause the softmax warning and perfect metrics.")
    logger.error("Verify your data source contains multiple classes.")

# === 5 · MODEL CALLBACKS ===
def common_callbacks(folder, fname, model_obj):
    """Create common callbacks for all models"""
    os.makedirs(folder, exist_ok=True)
    return [
        model_obj.early_stopping_callback(),
        model_obj.model_checkpoint_callback(os.path.join(folder, fname))
    ]

# === 6 · MODEL TRAINING LOOP ===
# Iterate each architecture if enabled in config
for arch_name, cls in [
    ("NeuralNetwork", NeuralNetwork),
    ("CNN", ConvolutionalNeuralNetwork),
    ("RNN", RecurrentNeuralNetwork),
    ("LSTM", LongShortTermMemory),
]:
    cfg = config[arch_name]
    if not cfg["enabled"]:
        logger.info(f"Skipping {arch_name} (disabled in config)")
        continue

    logger.info(f"🚀 Training {arch_name} model...")
    with mlflow.start_run(run_name=f"{arch_name}_training"):
        folder = os.path.join(OUTPUT_ROOT, cfg["name_parameters"]["folder_name"])
        fname  = cfg["name_parameters"]["model_name"].replace(".keras","_feast.keras")

        # Create model instance
        model = cls(
            N_TS, N_FEAT,
            cfg["model_parameters"]["activation_function"],
            cfg["model_parameters"].get("units", cfg["model_parameters"].get("filters")),
            N_CLS
        )

        # Build and compile model
        model.create_model()

        # Log model summary for debugging
        logger.info(f"Model architecture for {arch_name}:")
        model.model.summary(print_fn=lambda x: logger.info(x))

        # Verify output layer shape matches number of classes
        output_shape = model.model.output_shape
        logger.info(f"Model output shape: {output_shape}")
        if output_shape[-1] != N_CLS:
            logger.error(f"ERROR: Model output dimension ({output_shape[-1]}) doesn't match number of classes ({N_CLS})")

        model.model_compilation(model.model)

        # Train model
        history = model.model_fitting(
            model.model,
            X_train, Y_train,
            X_test, Y_test,
            common_callbacks(folder, fname, model),
            cfg["training_parameters"]["epochs"],
            cfg["training_parameters"]["batch_size"],
        )

        # Evaluate and visualize
        preprocessing_functions.plot_model_history(history, folder)

        # Following the original code pattern for evaluation
        logger.info("Evaluating on combined dataset")
        model.model_evaluation(
            model.model,
            np.concatenate([X_train, X_test]),
            np.concatenate([Y_train, Y_test]),
            X_test, Y_test
        )

        logger.info("Computing metrics on test set")
        model.compute_metrics(model.model, X_test, Y_test)

        # === LOG MODEL TO MLFLOW ===
        from mlflow.models.signature import infer_signature

        # Use a small sample for signature inference and input example
        sample_input = X_test[:5]  # Take first 5 samples
        sample_predictions = model.model.predict(sample_input, verbose=0)

        # Create signature
        signature = infer_signature(sample_input, sample_predictions)

        # Log the Keras model to MLflow using the keras flavor
        logger.info(f"Logging {arch_name} model to MLflow...")
        mlflow.keras.log_model(
            keras_model=model.model,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
            registered_model_name=f"{arch_name}_FailurePrediction"
        )

        # Log model parameters and metrics for tracking
        mlflow.log_params({
            "model_architecture": arch_name,
            "sequence_length": N_TS,
            "n_features": N_FEAT,
            "n_classes": N_CLS,
            "activation_function": cfg["model_parameters"]["activation_function"],
            "epochs": cfg["training_parameters"]["epochs"],
            "batch_size": cfg["training_parameters"]["batch_size"]
        })

        # Log final test metrics
        test_loss, test_accuracy = model.model.evaluate(X_test, Y_test, verbose=0)
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        logger.info(f"✅ {arch_name} model logged to MLflow registry as '{arch_name}_FailurePrediction'")

logger.info("🎉 All models trained on Feast-sourced features!")
