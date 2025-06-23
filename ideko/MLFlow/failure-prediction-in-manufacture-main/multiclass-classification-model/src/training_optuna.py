import os, gc, numpy as np, pandas as pd
from feast import FeatureStore
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper
from helpers.data_helper import DataHelper
from helpers.optuna_helper import OptunaHelper
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
MLFLOW_RUN_NAME = "OptunaTuned"

# Optuna configuration
OPTUNA_N_TRIALS = 3  # Number of hyperparameter trials per model (reduced for testing)

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

# === 6 · MODEL TRAINING WITH OPTUNA OPTIMIZATION ===

##################################
# NEURAL NETWORK
##################################
config_nn = config["NeuralNetwork"]
if config_nn["enabled"]:
    logger.info("🚀 Training NEURAL NETWORK model with Optuna optimization...")

    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_NeuralNetwork"):
        # Log input dataset
        dataset = mlflow.data.from_pandas(
            training_df,
            source=parquet_source_path,
            targets="anomaly_class"
        )
        mlflow.log_input(dataset, context="training")

        # Get optimized parameters from Optuna
        best_params, study = OptunaHelper.optimize_model(
            NeuralNetwork, "NeuralNetwork", config_nn,
            X_train, Y_train, X_test, Y_test,
            N_TS, N_FEAT, N_CLS, OPTUNA_N_TRIALS
        )
        
        # Log Optuna results to MLflow
        mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("optuna_best_accuracy", study.best_value)

        # Use optimized parameters
        activation_function = best_params["activation"]
        units = best_params["units"]
        batch_size = best_params["batch_size"]
        learning_rate = best_params["learning_rate"]

        # Model paths and parameters
        folder = os.path.join(OUTPUT_ROOT, config_nn["name_parameters"]["folder_name"])
        fname = config_nn["name_parameters"]["model_name"].replace(".keras", "_optuna.keras")
        epochs = config_nn["training_parameters"]["epochs"]

        # Create and train model with optimized parameters
        model_nn = NeuralNetwork(N_TS, N_FEAT, activation_function, units, N_CLS)
        model_nn.create_model()
        
        # Compile with optimized learning rate
        model_nn.model.compile(
            optimizer=model_nn.model.optimizer.__class__(learning_rate=learning_rate),
            loss=model_nn.model.loss,
            metrics=model_nn.model.metrics
        )

        history = model_nn.model_fitting(
            model_nn.model, X_train, Y_train, X_test, Y_test,
            common_callbacks(folder, fname, model_nn), epochs, batch_size
        )

        # Evaluate and log
        preprocessing_functions.plot_model_history(history, folder)
        model_nn.model_evaluation(model_nn.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_nn.compute_metrics(model_nn.model, X_test, Y_test)

        # Register model
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "NeuralNetwork_OptunaTuned")

        logger.info("✅ NeuralNetwork model with Optuna optimization logged to MLflow registry")

##################################
# CONVOLUTIONAL NEURAL NETWORK
##################################
config_cnn = config["CNN"]
if config_cnn["enabled"]:
    logger.info("🚀 Training CNN model with Optuna optimization...")

    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_CNN"):
        # Get optimized parameters from Optuna
        best_params, study = OptunaHelper.optimize_model(
            ConvolutionalNeuralNetwork, "CNN", config_cnn,
            X_train, Y_train, X_test, Y_test,
            N_TS, N_FEAT, N_CLS, OPTUNA_N_TRIALS
        )
        
        # Log Optuna results to MLflow
        mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("optuna_best_accuracy", study.best_value)

        # Use optimized parameters
        activation_function = best_params["activation"]
        filters = best_params["filters"]
        kernel_size = best_params["kernel_size"]
        pool_size = best_params["pool_size"]
        batch_size = best_params["batch_size"]
        learning_rate = best_params["learning_rate"]

        # Model paths and parameters
        folder = os.path.join(OUTPUT_ROOT, config_cnn["name_parameters"]["folder_name"])
        fname = config_cnn["name_parameters"]["model_name"].replace(".keras", "_optuna.keras")
        epochs = config_cnn["training_parameters"]["epochs"]

        # Create and train model with optimized parameters
        model_cnn = ConvolutionalNeuralNetwork(N_TS, N_FEAT, activation_function, filters, kernel_size, pool_size, N_CLS)
        model_cnn.create_model()
        
        # Compile with optimized learning rate
        model_cnn.model.compile(
            optimizer=model_cnn.model.optimizer.__class__(learning_rate=learning_rate),
            loss=model_cnn.model.loss,
            metrics=model_cnn.model.metrics
        )

        history = model_cnn.model_fitting(
            model_cnn.model, X_train, Y_train, X_test, Y_test,
            common_callbacks(folder, fname, model_cnn), epochs, batch_size
        )

        # Evaluate and log
        preprocessing_functions.plot_model_history(history, folder)
        model_cnn.model_evaluation(model_cnn.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_cnn.compute_metrics(model_cnn.model, X_test, Y_test)

        # Register model
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "CNN_OptunaTuned")

        logger.info("✅ CNN model with Optuna optimization logged to MLflow registry")

##################################
# RNN AND LSTM (Similar pattern)
##################################
config_rnn = config["RNN"]
if config_rnn["enabled"]:
    logger.info("🚀 Training RNN model with Optuna optimization...")

    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_RNN"):
        best_params, study = OptunaHelper.optimize_model(
            RecurrentNeuralNetwork, "RNN", config_rnn,
            X_train, Y_train, X_test, Y_test,
            N_TS, N_FEAT, N_CLS, OPTUNA_N_TRIALS
        )
        
        mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("optuna_best_accuracy", study.best_value)

        activation_function = best_params["activation"]
        hidden_units = best_params["hidden_units"]
        batch_size = best_params["batch_size"]
        learning_rate = best_params["learning_rate"]

        folder = os.path.join(OUTPUT_ROOT, config_rnn["name_parameters"]["folder_name"])
        fname = config_rnn["name_parameters"]["model_name"].replace(".keras", "_optuna.keras")
        epochs = config_rnn["training_parameters"]["epochs"]

        model_rnn = RecurrentNeuralNetwork(N_TS, N_FEAT, activation_function, hidden_units, N_CLS)
        model_rnn.create_model()
        
        model_rnn.model.compile(
            optimizer=model_rnn.model.optimizer.__class__(learning_rate=learning_rate),
            loss=model_rnn.model.loss,
            metrics=model_rnn.model.metrics
        )

        history = model_rnn.model_fitting(
            model_rnn.model, X_train, Y_train, X_test, Y_test,
            common_callbacks(folder, fname, model_rnn), epochs, batch_size
        )

        preprocessing_functions.plot_model_history(history, folder)
        model_rnn.model_evaluation(model_rnn.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_rnn.compute_metrics(model_rnn.model, X_test, Y_test)

        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "RNN_OptunaTuned")
        logger.info("✅ RNN model with Optuna optimization logged to MLflow registry")

config_lstm = config["LSTM"]
if config_lstm["enabled"]:
    logger.info("🚀 Training LSTM model with Optuna optimization...")

    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_LSTM"):
        best_params, study = OptunaHelper.optimize_model(
            LongShortTermMemory, "LSTM", config_lstm,
            X_train, Y_train, X_test, Y_test,
            N_TS, N_FEAT, N_CLS, OPTUNA_N_TRIALS
        )
        
        mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("optuna_best_accuracy", study.best_value)

        activation_function = best_params["activation"]
        hidden_units = best_params["hidden_units"]
        batch_size = best_params["batch_size"]
        learning_rate = best_params["learning_rate"]

        folder = os.path.join(OUTPUT_ROOT, config_lstm["name_parameters"]["folder_name"])
        fname = config_lstm["name_parameters"]["model_name"].replace(".keras", "_optuna.keras")
        epochs = config_lstm["training_parameters"]["epochs"]

        model_lstm = LongShortTermMemory(N_TS, N_FEAT, activation_function, hidden_units, N_CLS)
        model_lstm.create_model()
        
        model_lstm.model.compile(
            optimizer=model_lstm.model.optimizer.__class__(learning_rate=learning_rate),
            loss=model_lstm.model.loss,
            metrics=model_lstm.model.metrics
        )

        history = model_lstm.model_fitting(
            model_lstm.model, X_train, Y_train, X_test, Y_test,
            common_callbacks(folder, fname, model_lstm), epochs, batch_size
        )

        preprocessing_functions.plot_model_history(history, folder)
        model_lstm.model_evaluation(model_lstm.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_lstm.compute_metrics(model_lstm.model, X_test, Y_test)

        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "LSTM_OptunaTuned")
        logger.info("✅ LSTM model with Optuna optimization logged to MLflow registry")

logger.info("🎉 All enabled models trained with Optuna optimization and logged to MLflow!")