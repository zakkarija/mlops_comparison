import os, gc, numpy as np, pandas as pd
from feast import FeatureStore
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper
from helpers.data_helper import DataHelper
from classes import preprocessing_functions
import mlflow
import mlflow.keras
import mlflow.data
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

# MLflow run configuration
MLFLOW_RUN_NAME = "SystemMetricsLogTest"

# Helper function to check class distribution at different stages
def _check(label_arr, stage):
    uniques, counts = np.unique(label_arr, return_counts=True)
    logger.info(f"[CHECK] {stage}: {dict(zip(uniques, counts))}")

# === MLFLOW TRACKING SETUP ===
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Enable MLflow autologging for TensorFlow/Keras (full autologging)
mlflow.tensorflow.autolog(log_models=True, registered_model_name=None)

# Enable system metrics tracking (CPU, memory, GPU)
mlflow.enable_system_metrics_logging()

# === 1 · FEAST DATA RETRIEVAL ===
FEAST_REPO = "feast_demo/feature_repo"
FEATURE_COLUMNS = ["f3_current", "f3_rolling_mean_10", "f3_rolling_std_10"]
LABEL_FEATURE = "f3_timeseries_features:anomaly_class"

# Read entity_df for Feast queries
logger.info("Reading entity DataFrame for Feast queries")
with mlflow.start_span(name="data_loading") as span:
    timeseries_df = pd.read_parquet(
        f"{FEAST_REPO}/data/offline/f3_timeseries.parquet",
        columns=["equipment_id", "event_timestamp"]
    )
    span.set_attribute("rows_loaded", len(timeseries_df))

    timeseries_df = timeseries_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    entity_df = timeseries_df.copy(); del timeseries_df

    # Use DataHelper to batch read from Feast with memory optimization
    training_df = DataHelper.batch_read_feast_data(
        feast_repo_path=FEAST_REPO,
        entity_df=entity_df,
        feature_columns=FEATURE_COLUMNS,
        label_feature=LABEL_FEATURE,
        batch_size=2000,
        max_batches=3
    )
    span.set_attribute("final_rows", len(training_df))
    span.set_attribute("feature_columns", len(FEATURE_COLUMNS))

del entity_df; gc.collect()

# Prepare dataset for MLflow input logging
parquet_source_path = f"{FEAST_REPO}/data/offline/f3_timeseries.parquet"

# Check class distribution after Feast data pull
_check(training_df['anomaly_class'].values, "raw Feast pull")

# Verify data has multiple classes
class_counts = training_df["anomaly_class"].value_counts()
if len(class_counts) < 2:
    logger.error(f"WARNING: Only found {len(class_counts)} classes in data: {class_counts.to_dict()}. This will result in a model that only predicts one class. Check your Feast feature definitions and label extraction.")

# === 2 · TRAIN/TEST SPLIT ===
logger.info("Splitting data into train and test sets")
with mlflow.start_span(name="data_preprocessing") as span:
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

    span.set_attribute("train_sequences", len(X_train))
    span.set_attribute("test_sequences", len(X_test))
    span.set_attribute("sequence_length", 10)

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

# === 6 · MODEL TRAINING ===

##################################
# NEURAL NETWORK
##################################
config_nn = config["NeuralNetwork"]
if config_nn["enabled"]:
    logger.info("🚀 Training NEURAL NETWORK model...")

    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_NeuralNetwork"):
        # Log input dataset using MLflow's automatic metadata extraction
        dataset = mlflow.data.from_pandas(
            training_df,
            source=parquet_source_path,
            targets="anomaly_class"
        )
        mlflow.log_input(dataset, context="training")

        # Model paths and parameters
        folder = os.path.join(OUTPUT_ROOT, config_nn["name_parameters"]["folder_name"])
        fname = config_nn["name_parameters"]["model_name"].replace(".keras", "_feast.keras")

        activation_function = config_nn["model_parameters"]["activation_function"]
        units = config_nn["model_parameters"]["units"]
        epochs = config_nn["training_parameters"]["epochs"]
        batch_size = config_nn["training_parameters"]["batch_size"]

        # Create and train model
        model_nn = NeuralNetwork(N_TS, N_FEAT, activation_function, units, N_CLS)
        model_nn.create_model()
        model_nn.model_compilation(model_nn.model)

        history = model_nn.model_fitting(
            model_nn.model, X_train, Y_train, X_test, Y_test,
            common_callbacks(folder, fname, model_nn), epochs, batch_size
        )

        # Evaluate and log
        preprocessing_functions.plot_model_history(history, folder)
        model_nn.model_evaluation(model_nn.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_nn.compute_metrics(model_nn.model, X_test, Y_test)

        # Register model manually (autologging handles metrics/params automatically)
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "NeuralNetwork_FailurePrediction")

        logger.info("✅ NeuralNetwork model logged to MLflow registry")

##################################
# CONVOLUTIONAL NEURAL NETWORK
##################################
config_cnn = config["CNN"]
if config_cnn["enabled"]:
    logger.info("🚀 Training CNN model...")

    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_CNN"):
        # Model paths and parameters
        folder = os.path.join(OUTPUT_ROOT, config_cnn["name_parameters"]["folder_name"])
        fname = config_cnn["name_parameters"]["model_name"].replace(".keras", "_feast.keras")

        activation_function = config_cnn["model_parameters"]["activation_function"]
        filters = config_cnn["model_parameters"]["filters"]
        kernel_size = config_cnn["model_parameters"]["kernel_size"]
        pool_size = config_cnn["model_parameters"]["pool_size"]
        epochs = config_cnn["training_parameters"]["epochs"]
        batch_size = config_cnn["training_parameters"]["batch_size"]

        # Create and train model
        model_cnn = ConvolutionalNeuralNetwork(N_TS, N_FEAT, activation_function, filters, kernel_size, pool_size, N_CLS)
        model_cnn.create_model()
        model_cnn.model_compilation(model_cnn.model)

        history = model_cnn.model_fitting(
            model_cnn.model, X_train, Y_train, X_test, Y_test,
            common_callbacks(folder, fname, model_cnn), epochs, batch_size
        )

        # Evaluate and log
        preprocessing_functions.plot_model_history(history, folder)
        model_cnn.model_evaluation(model_cnn.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_cnn.compute_metrics(model_cnn.model, X_test, Y_test)

        # Log to MLflow
        from mlflow.models.signature import infer_signature
        sample_input = X_test[:5]
        sample_predictions = model_cnn.model.predict(sample_input, verbose=0)
        signature = infer_signature(sample_input, sample_predictions)

        mlflow.keras.log_model(
            model=model_cnn.model, artifact_path="model", signature=signature,
            registered_model_name="CNN_FailurePrediction"
        )

        test_loss, test_accuracy = model_cnn.model.evaluate(X_test, Y_test, verbose=0)
        mlflow.log_params({
            "model_architecture": "CNN", "sequence_length": N_TS, "n_features": N_FEAT,
            "n_classes": N_CLS, "activation_function": activation_function, "epochs": epochs, "batch_size": batch_size
        })
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})

        logger.info("✅ CNN model logged to MLflow registry")

##################################
# RECURRENT NEURAL NETWORK
##################################
config_rnn = config["RNN"]
if config_rnn["enabled"]:
    logger.info("🚀 Training RNN model...")

    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_RNN"):
        # Model paths and parameters
        folder = os.path.join(OUTPUT_ROOT, config_rnn["name_parameters"]["folder_name"])
        fname = config_rnn["name_parameters"]["model_name"].replace(".keras", "_feast.keras")

        activation_function = config_rnn["model_parameters"]["activation_function"]
        hidden_units = config_rnn["model_parameters"]["hidden_units"]
        epochs = config_rnn["training_parameters"]["epochs"]
        batch_size = config_rnn["training_parameters"]["batch_size"]

        # Create and train model
        model_rnn = RecurrentNeuralNetwork(N_TS, N_FEAT, activation_function, hidden_units, N_CLS)
        model_rnn.create_model()
        model_rnn.model_compilation(model_rnn.model)

        history = model_rnn.model_fitting(
            model_rnn.model, X_train, Y_train, X_test, Y_test,
            common_callbacks(folder, fname, model_rnn), epochs, batch_size
        )

        # Evaluate and log
        preprocessing_functions.plot_model_history(history, folder)
        model_rnn.model_evaluation(model_rnn.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_rnn.compute_metrics(model_rnn.model, X_test, Y_test)

        # Log to MLflow
        from mlflow.models.signature import infer_signature
        sample_input = X_test[:5]
        sample_predictions = model_rnn.model.predict(sample_input, verbose=0)
        signature = infer_signature(sample_input, sample_predictions)

        mlflow.keras.log_model(
            model=model_rnn.model, artifact_path="model", signature=signature,
            registered_model_name="RNN_FailurePrediction"
        )

        test_loss, test_accuracy = model_rnn.model.evaluate(X_test, Y_test, verbose=0)
        mlflow.log_params({
            "model_architecture": "RNN", "sequence_length": N_TS, "n_features": N_FEAT,
            "n_classes": N_CLS, "activation_function": activation_function, "epochs": epochs, "batch_size": batch_size
        })
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})

        logger.info("✅ RNN model logged to MLflow registry")

##################################
# LONG SHORT TERM MEMORY
##################################
config_lstm = config["LSTM"]
if config_lstm["enabled"]:
    logger.info("🚀 Training LSTM model...")

    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_LSTM"):
        # Model paths and parameters
        folder = os.path.join(OUTPUT_ROOT, config_lstm["name_parameters"]["folder_name"])
        fname = config_lstm["name_parameters"]["model_name"].replace(".keras", "_feast.keras")

        activation_function = config_lstm["model_parameters"]["activation_function"]
        hidden_units = config_lstm["model_parameters"]["hidden_units"]
        epochs = config_lstm["training_parameters"]["epochs"]
        batch_size = config_lstm["training_parameters"]["batch_size"]

        # Create and train model
        model_lstm = LongShortTermMemory(N_TS, N_FEAT, activation_function, hidden_units, N_CLS)
        model_lstm.create_model()
        model_lstm.model_compilation(model_lstm.model)

        history = model_lstm.model_fitting(
            model_lstm.model, X_train, Y_train, X_test, Y_test,
            common_callbacks(folder, fname, model_lstm), epochs, batch_size
        )

        # Evaluate and log
        preprocessing_functions.plot_model_history(history, folder)
        model_lstm.model_evaluation(model_lstm.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_lstm.compute_metrics(model_lstm.model, X_test, Y_test)

        # Log to MLflow
        from mlflow.models.signature import infer_signature
        sample_input = X_test[:5]
        sample_predictions = model_lstm.model.predict(sample_input, verbose=0)
        signature = infer_signature(sample_input, sample_predictions)

        mlflow.keras.log_model(
            model=model_lstm.model, artifact_path="model", signature=signature,
            registered_model_name="LSTM_FailurePrediction"
        )

        test_loss, test_accuracy = model_lstm.model.evaluate(X_test, Y_test, verbose=0)
        mlflow.log_params({
            "model_architecture": "LSTM", "sequence_length": N_TS, "n_features": N_FEAT,
            "n_classes": N_CLS, "activation_function": activation_function, "epochs": epochs, "batch_size": batch_size
        })
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})

        logger.info("✅ LSTM model logged to MLflow registry")

logger.info("🎉 All enabled models trained with Feast-sourced features and logged to MLflow!")
