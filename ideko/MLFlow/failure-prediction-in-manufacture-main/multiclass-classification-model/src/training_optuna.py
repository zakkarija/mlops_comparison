import os, gc, numpy as np, pandas as pd
from feast import FeatureStore
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper
from helpers.data_helper import DataHelper
from classes import preprocessing_functions
import mlflow  # type: ignore
import mlflow.keras  # type: ignore
import optuna
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

# MLflow and Optuna configuration
MLFLOW_RUN_NAME = "OptunaTuned"
OPTUNA_N_TRIALS = 3

# Helper function to check class distribution at different stages
def _check(label_arr, stage):
    uniques, counts = np.unique(label_arr, return_counts=True)
    logger.info(f"[CHECK] {stage}: {dict(zip(uniques, counts))}")

# === MLFLOW TRACKING SETUP ===
mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
mlflow.keras.autolog(log_models=True)  # type: ignore[attr-defined]
mlflow.enable_system_metrics_logging()

# === 1 · FEAST DATA RETRIEVAL ===
FEAST_REPO = os.path.join(os.path.dirname(__file__), "feast_demo/feature_repo")
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
# Callbacks are now defined inline for each model

# === 6 · OPTUNA OBJECTIVE FUNCTION ===
def objective(trial):  # type: ignore
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # Suggest hyperparameters
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        units_0 = trial.suggest_int('units_0', 32, 128)
        units_1 = trial.suggest_int('units_1', 16, 64)

        # Log trial parameters
        mlflow.log_params({
            'activation': activation,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'units_0': units_0,
            'units_1': units_1
        })

        # Create model with suggested parameters
        model_nn = NeuralNetwork(N_TS, N_FEAT, activation, [units_0, units_1], N_CLS)
        model_nn.create_model()

        # Compile with suggested learning rate
        model_nn.model.compile(  # type: ignore
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        history = model_nn.model.fit(  # type: ignore
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            epochs=10,
            batch_size=batch_size,
            verbose=0  # type: ignore
        )

        # Log metrics
        val_accuracy = max(history.history['val_accuracy'])
        mlflow.log_metric("val_accuracy", val_accuracy)

        return val_accuracy

# === 7 · MODEL TRAINING WITH OPTUNA ===
config_nn = config["NeuralNetwork"]
if config_nn["enabled"]:
    logger.info("🚀 Training NEURAL NETWORK with Optuna optimization...")

    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_NeuralNetwork"):
        # Run Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS)  # type: ignore

        # Log best parameters and results to parent run
        best_params = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_accuracy", study.best_value)

        # Train final model with best parameters
        folder = os.path.join(OUTPUT_ROOT, config_nn["name_parameters"]["folder_name"])
        fname = config_nn["name_parameters"]["model_name"].replace(".keras", "_optuna.keras")
        os.makedirs(folder, exist_ok=True)

        model_nn = NeuralNetwork(N_TS, N_FEAT, best_params['activation'],
                                [best_params['units_0'], best_params['units_1']], N_CLS)
        model_nn.create_model()
        model_nn.model_compilation(model_nn.model)

        # Train with callbacks
        callbacks = [
            model_nn.early_stopping_callback(),
            model_nn.model_checkpoint_callback(os.path.join(folder, fname))
        ]

        history = model_nn.model_fitting(
            model_nn.model, X_train, Y_train, X_test, Y_test,
            callbacks, config_nn["training_parameters"]["epochs"], best_params['batch_size']
        )

        # Log final model
        mlflow.keras.log_model(model_nn.model, "model")  # type: ignore[attr-defined]

        # Evaluate and log results
        preprocessing_functions.plot_model_history(history, folder)
        model_nn.model_evaluation(model_nn.model, np.concatenate([X_train, X_test]),
                                 np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_nn.compute_metrics(model_nn.model, X_test, Y_test)

        logger.info("✅ NeuralNetwork model with Optuna optimization complete")

# CNN, RNN, and LSTM models follow similar pattern
config_cnn = config["CNN"]
if config_cnn["enabled"]:
    logger.info("🚀 Training CNN model...")
    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_CNN"):
        folder = os.path.join(OUTPUT_ROOT, config_cnn["name_parameters"]["folder_name"])
        fname = config_cnn["name_parameters"]["model_name"]
        os.makedirs(folder, exist_ok=True)

        model_cnn = ConvolutionalNeuralNetwork(
            N_TS, N_FEAT,
            config_cnn["model_parameters"]["activation_function"],
            config_cnn["model_parameters"]["filters"],
            config_cnn["model_parameters"]["kernel_size"],
            config_cnn["model_parameters"]["pool_size"],
            N_CLS
        )
        model_cnn.create_model()
        model_cnn.model_compilation(model_cnn.model)

        callbacks = [model_cnn.early_stopping_callback(),
                    model_cnn.model_checkpoint_callback(os.path.join(folder, fname))]

        history = model_cnn.model_fitting(
            model_cnn.model, X_train, Y_train, X_test, Y_test, callbacks,
            config_cnn["training_parameters"]["epochs"],
            config_cnn["training_parameters"]["batch_size"]
        )

        preprocessing_functions.plot_model_history(history, folder)
        model_cnn.model_evaluation(model_cnn.model, np.concatenate([X_train, X_test]),
                                  np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_cnn.compute_metrics(model_cnn.model, X_test, Y_test)

config_rnn = config["RNN"]
if config_rnn["enabled"]:
    logger.info("🚀 Training RNN model...")
    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_RNN"):
        folder = os.path.join(OUTPUT_ROOT, config_rnn["name_parameters"]["folder_name"])
        fname = config_rnn["name_parameters"]["model_name"]
        os.makedirs(folder, exist_ok=True)

        model_rnn = RecurrentNeuralNetwork(
            N_TS, N_FEAT,
            config_rnn["model_parameters"]["activation_function"],
            config_rnn["model_parameters"]["hidden_units"],
            N_CLS
        )
        model_rnn.create_model()
        model_rnn.model_compilation(model_rnn.model)

        callbacks = [model_rnn.early_stopping_callback(),
                    model_rnn.model_checkpoint_callback(os.path.join(folder, fname))]

        history = model_rnn.model_fitting(
            model_rnn.model, X_train, Y_train, X_test, Y_test, callbacks,
            config_rnn["training_parameters"]["epochs"],
            config_rnn["training_parameters"]["batch_size"]
        )

        preprocessing_functions.plot_model_history(history, folder)
        model_rnn.model_evaluation(model_rnn.model, np.concatenate([X_train, X_test]),
                                  np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_rnn.compute_metrics(model_rnn.model, X_test, Y_test)

config_lstm = config["LSTM"]
if config_lstm["enabled"]:
    logger.info("🚀 Training LSTM model...")
    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_LSTM"):
        folder = os.path.join(OUTPUT_ROOT, config_lstm["name_parameters"]["folder_name"])
        fname = config_lstm["name_parameters"]["model_name"]
        os.makedirs(folder, exist_ok=True)

        model_lstm = LongShortTermMemory(
            N_TS, N_FEAT,
            config_lstm["model_parameters"]["activation_function"],
            config_lstm["model_parameters"]["hidden_units"],
            N_CLS
        )
        model_lstm.create_model()
        model_lstm.model_compilation(model_lstm.model)

        callbacks = [model_lstm.early_stopping_callback(),
                    model_lstm.model_checkpoint_callback(os.path.join(folder, fname))]

        history = model_lstm.model_fitting(
            model_lstm.model, X_train, Y_train, X_test, Y_test, callbacks,
            config_lstm["training_parameters"]["epochs"],
            config_lstm["training_parameters"]["batch_size"]
        )

        preprocessing_functions.plot_model_history(history, folder)
        model_lstm.model_evaluation(model_lstm.model, np.concatenate([X_train, X_test]),
                                   np.concatenate([Y_train, Y_test]), X_test, Y_test)
        model_lstm.compute_metrics(model_lstm.model, X_test, Y_test)

logger.info("🎉 All enabled models trained and logged to MLflow!")