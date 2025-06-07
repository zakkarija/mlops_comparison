# ------------------------------------------------------------------------------------
# Feast‑powered Time‑series Anomaly‑Detection Training Pipeline
# ------------------------------------------------------------------------------------
# This script replaces the old “read many CSVs ➜ pad irregular series” approach with a
# clean **Feast** offline ­store query.  The rest of the modelling code (NN / CNN / RNN
# / LSTM) is preserved, but now receives a dynamic **n_features** that automatically
# adjusts to however many columns we decide to pull from Feast.
#
# High‑level flow
# 1.  Pull a *minimal* feature set (+ label) from Feast in RAM‑bounded batches.
# 2.  Split *equipment_id* into train / test first → prevents temporal leakage.
# 3.  Build sliding‑window sequences per split (overlap stride = 2).
# 4.  Encode labels, train the four architectures, log metrics.
# ------------------------------------------------------------------------------------

# ===== Standard library
import os
import sys
import gc                                    # manual garbage collection after each batch

# ===== Third‑party
import numpy as np
import pandas as pd
from feast import FeatureStore               # offline feature retrieval
from sklearn.model_selection import train_test_split

# ===== Project helpers
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper
from classes import preprocessing_functions
from classes.multiclass_models import (
    NeuralNetwork,
    ConvolutionalNeuralNetwork,
    RecurrentNeuralNetwork,
    LongShortTermMemory,
)

# ------------------------------------------------------------------------------------
# 0 · HOUSE‑KEEPING (logging, config, output dirs)
# ------------------------------------------------------------------------------------
LoggerHelper.init_logger()
logger = logging.getLogger(__name__)
config = ConfigHelper.instance("models")

SCRIPT_DIR   = os.path.dirname(os.path.realpath(__file__))
OUTPUT_ROOT  = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ------------------------------------------------------------------------------------
# 1 · FEATURE DEFINITIONS & FEAST INITIALISATION
# ------------------------------------------------------------------------------------
# ✏️  Keep the feature list short – four signals are enough to *demonstrate* how Feast
#     plugs into the pipeline without exploding dimensionality or memory use.
FEATURE_COLUMNS = [
    "f3_current",          # raw sensor reading
    "f3_rolling_mean_10", # recent average → trend
    "f3_rolling_std_10",  # recent volatility → noise level / vibration
    "movement_direction", # engineered categorical (‑1 / 0 / +1)
]

# Pre‑pend the feature‑view name expected by Feast (matches repo yaml).
FEAST_FEATURES = [f"f3_timeseries_features:{col}" for col in FEATURE_COLUMNS]
LABEL_FEATURE  = "f3_timeseries_features:anomaly_class"  # 0=normal / 1 / 2

# Initialise the FeatureStore client (points to local repo – could be env‑var driven)
FEAST_REPO = "feast_demo/feature_repo"
fs = FeatureStore(repo_path=FEAST_REPO)

# ------------------------------------------------------------------------------------
# 2 · BATCH‑WISE OFFLINE RETRIEVAL  (memory‑friendly)
# ------------------------------------------------------------------------------------
logger.info("📡  Querying Feast offline store in batches…")

# 2‑A · Read only *entity* & *timestamp* columns first – ultra‑cheap.
timeseries_path = os.path.join(FEAST_REPO, "data", "offline", "f3_timeseries.parquet")
timeseries_df   = pd.read_parquet(timeseries_path, columns=["equipment_id", "event_timestamp"])

entity_df = timeseries_df.copy()  # just (equipment_id, timestamp)

# 2‑B · Iterate through entity_df in fixed‑size slices; retrieve features+label per slice.
BATCH_ROWS = 2_000
total_batches = (len(entity_df) + BATCH_ROWS - 1) // BATCH_ROWS
num_batches   = min(total_batches, 10)

batches = []
for i in range(num_batches):
    start, end = i * BATCH_ROWS, min((i + 1) * BATCH_ROWS, len(entity_df))
    batch_entity = entity_df.iloc[start:end]

    logger.info(
        f"🗂️  Batch {i+1}/{num_batches}  (rows {start}‑{end}) → Feast lookup"
    )

    batch_df = fs.get_historical_features(
        entity_df=batch_entity,
        features=FEAST_FEATURES + [LABEL_FEATURE],
    ).to_df()

    batches.append(batch_df)
    gc.collect()   # release Arrow / PyArrow arena from Feast

training_df = pd.concat(batches, ignore_index=True)
del batches, timeseries_df; gc.collect()

logger.info(f"✅  Retrieved  {training_df.shape[0]:,} rows × {training_df.shape[1]} columns")

# Rename the label for ergonomic access later.
training_df.rename(columns={LABEL_FEATURE: "anomaly_class"}, inplace=True)

# ------------------------------------------------------------------------------------
# 3 · TRAIN / TEST SPLIT  *before* windowing  (prevents leakage)
# ------------------------------------------------------------------------------------
logger.info("🔀  Splitting by equipment_id to avoid cross‑series leakage…")

all_equip_ids = training_df["equipment_id"].unique()

# For stratification we need one label *per equipment*.  Use the modal class in that
# equipment's history – pragmatic and keeps sklearn.train_test_split happy.
majority_label_per_eq = (
    training_df.groupby("equipment_id")["anomaly_class"]
    .agg(lambda s: s.value_counts().idxmax())
    .reindex(all_equip_ids)
)

train_eq, test_eq = train_test_split(
    all_equip_ids,
    test_size=0.2,
    random_state=123,
    stratify=majority_label_per_eq,
)

logger.info(f"🛠️  Train = {len(train_eq)} equipment,  Test = {len(test_eq)} equipment")

# Sub‑dataframes
train_df = training_df[training_df["equipment_id"].isin(train_eq)].copy()
test_df  = training_df[training_df["equipment_id"].isin(test_eq)].copy()

# ------------------------------------------------------------------------------------
# 4 · SEQUENCE ( WINDOW ) CONSTRUCTION
# ------------------------------------------------------------------------------------
# Each model expects input shaped : (batch, timesteps, features).  We build overlapping
# windows of 10 time‑steps with stride = 2 (75 % overlap) for a richer training set.

def build_sequences(df: pd.DataFrame, seq_len: int = 10, stride: int = 2):
    """Return X, y arrays where X.shape = (n_seq, seq_len, n_feat)."""
    feat_cols = FEATURE_COLUMNS
    X, y = [], []
    for eq_id in df["equipment_id"].unique():
        sub = df[df["equipment_id"] == eq_id].sort_values("event_timestamp")
        F   = sub[feat_cols].values.astype(np.float32)
        L   = sub["anomaly_class"].values
        if len(F) < seq_len:
            continue  # not enough points to form one window
        for start in range(0, len(F) - seq_len + 1, stride):
            X.append(F[start : start + seq_len])
            y.append(L[start + seq_len - 1])  # label = last item in window
    return np.asarray(X), np.asarray(y)

SEQ_LEN  = 10
STRIDE   = 2

X_train, y_train_int = build_sequences(train_df, SEQ_LEN, STRIDE)
X_test,  y_test_int  = build_sequences(test_df,  SEQ_LEN, STRIDE)

logger.info(
    f"📐  Built {len(X_train):,} train & {len(X_test):,} test sequences  "
    f"({SEQ_LEN} timesteps × {len(FEATURE_COLUMNS)} features)"
)

# ------------------------------------------------------------------------------------
# 5 · LABEL ENCODING  (one‑hot → categorical_crossentropy)
# ------------------------------------------------------------------------------------
LABEL_MAP = {0: "normal", 1: "mechanical_anomaly", 2: "electrical_anomaly"}

# Translate integer → string → one‑hot  (re‑use helper)
y_train_txt = np.vectorize(LABEL_MAP.get)(y_train_int)
y_test_txt  = np.vectorize(LABEL_MAP.get)(y_test_int)

Y_train = preprocessing_functions.encode_response_variable(y_train_txt)
Y_test  = preprocessing_functions.encode_response_variable(y_test_txt)

# ------------------------------------------------------------------------------------
# 6 · SHAPES & META
# ------------------------------------------------------------------------------------
N_TIMESTAMPS = SEQ_LEN
N_FEATURES   = len(FEATURE_COLUMNS)
N_CLASSES    = Y_train.shape[1]

logger.info(f"🔢  Model input dims  =  {N_TIMESTAMPS} × {N_FEATURES}")
logger.info(f"🔢  Num classes       =  {N_CLASSES}")

# ------------------------------------------------------------------------------------
# 7 · MODEL TRAINING LOOP (4 architectures)
# ------------------------------------------------------------------------------------
# Each block is gated by a config flag (config/*.yaml).  All architectures share:
#   * Early‑stopping (patience = helper default)
#   * ModelCheckpoint  (saves .keras under output/<arch_name>)
# The inputs X_train/X_test & Y_train/Y_test are identical across models.
# ------------------------------------------------------------------------------------

# ---- Helper to DRY callbacks ---------------------------------------------------------

def common_callbacks(folder: str, model_fname: str, model_cls):
    """Return [early_stop, checkpoint] list for a given model class."""
    os.makedirs(folder, exist_ok=True)
    early  = model_cls.early_stopping_callback()
    ckpt   = model_cls.model_checkpoint_callback(model_path=os.path.join(folder, model_fname))
    return [early, ckpt]

# ---- Neural Network -----------------------------------------------------------------
config_nn = config["NeuralNetwork"]
if config_nn["enabled"]:
    logger.info("🚀  Training **Neural Network** (dense) …")

    nn_folder = os.path.join(OUTPUT_ROOT, config_nn["name_parameters"]["folder_name"] + "_timeseries")
    nn_model  = config_nn["name_parameters"]["model_name"].replace(".keras", "_timeseries.keras")

    nn = NeuralNetwork(
        N_TIMESTAMPS,
        N_FEATURES,
        config_nn["model_parameters"]["activation_function"],
        config_nn["model_parameters"]["units"],
        N_CLASSES,
    )
    nn.create_model()
    nn.model_compilation(nn.model)

    history_nn = nn.model_fitting(
        nn.model,
        X_train, Y_train,
        X_test,  Y_test,
        common_callbacks(nn_folder, nn_model, nn),
        config_nn["training_parameters"]["epochs"],
        config_nn["training_parameters"]["batch_size"],
    )

    preprocessing_functions.plot_model_history(history_nn, nn_folder)
    nn.model_evaluation(nn.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
    logger.info("📊  NN metrics (test)")
    nn.compute_metrics(nn.model, X_test, Y_test)

# ---- Convolutional Neural Network ----------------------------------------------------
config_cnn = config["CNN"]
if config_cnn["enabled"]:
    logger.info("🚀  Training **CNN** …")

    cnn_folder = os.path.join(OUTPUT_ROOT, config_cnn["name_parameters"]["folder_name"] + "_timeseries")
    cnn_model  = config_cnn["name_parameters"]["model_name"].replace(".keras", "_timeseries.keras")

    # Ensure kernel width ≤ N_FEATURES (otherwise Keras will complain)
    kernel_size = tuple(config_cnn["model_parameters"]["kernel_size"])
    if kernel_size[1] > N_FEATURES:
        kernel_size = (kernel_size[0], N_FEATURES)

    cnn = ConvolutionalNeuralNetwork(
        N_TIMESTAMPS,
        N_FEATURES,
        config_cnn["model_parameters"]["activation_function"],
        config_cnn["model_parameters"]["filters"],
        kernel_size,
        config_cnn["model_parameters"]["pool_size"],
        N_CLASSES,
    )
    cnn.create_model()
    cnn.model_compilation(cnn.model)

    history_cnn = cnn.model_fitting(
        cnn.model,
        X_train, Y_train,
        X_test,  Y_test,
        common_callbacks(cnn_folder, cnn_model, cnn),
        config_cnn["training_parameters"]["epochs"],
        config_cnn["training_parameters"]["batch_size"],
    )

    preprocessing_functions.plot_model_history(history_cnn, cnn_folder)
    cnn.model_evaluation(cnn.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
    logger.info("📊  CNN metrics (test)")
    cnn.compute_metrics(cnn.model, X_test, Y_test)

# ---- Recurrent Neural Network --------------------------------------------------------
config_rnn = config["RNN"]
if config_rnn["enabled"]:
    logger.info("🚀  Training **RNN** …")

    rnn_folder = os.path.join(OUTPUT_ROOT, config_rnn["name_parameters"]["folder_name"] + "_timeseries")
    rnn_model  = config_rnn["name_parameters"]["model_name"].replace(".keras", "_timeseries.keras")

    rnn = RecurrentNeuralNetwork(
        N_TIMESTAMPS,
        N_FEATURES,
        config_rnn["model_parameters"]["activation_function"],
        config_rnn["model_parameters"]["hidden_units"],
        N_CLASSES,
    )
    rnn.create_model()
    rnn.model_compilation(rnn.model)

    history_rnn = rnn.model_fitting(
        rnn.model,
        X_train, Y_train,
        X_test,  Y_test,
        common_callbacks(rnn_folder, rnn_model, rnn),
        config_rnn["training_parameters"]["epochs"],
        config_rnn["training_parameters"]["batch_size"],
    )

    preprocessing_functions.plot_model_history(history_rnn, rnn_folder)
    rnn.model_evaluation(rnn.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
    logger.info("📊  RNN metrics (test)")
    rnn.compute_metrics(rnn.model, X_test, Y_test)

# ---- Long Short‑Term Memory ----------------------------------------------------------
config_lstm = config["LSTM"]
if config_lstm["enabled"]:
    logger.info("🚀  Training **LSTM** …")

    lstm_folder = os.path.join(OUTPUT_ROOT, config_lstm["name_parameters"]["folder_name"] + "_timeseries")
    lstm_model  = config_lstm["name_parameters"]["model_name"].replace(".keras", "_timeseries.keras")

    lstm = LongShortTermMemory(
        N_TIMESTAMPS,
        N_FEATURES,
        config_lstm["model_parameters"]["activation_function"],
        config_lstm["model_parameters"]["hidden_units"],
        N_CLASSES,
    )
    lstm.create_model()
    lstm.model_compilation(lstm.model)

    history_lstm = lstm.model_fitting(
        lstm.model,
        X_train, Y_train,
        X_test,  Y_test,
        common_callbacks(lstm_folder, lstm_model, lstm),
        config_lstm["training_parameters"]["epochs"],
        config_lstm["training_parameters"]["batch_size"],
    )

    preprocessing_functions.plot_model_history(history_lstm, lstm_folder)
    lstm.model_evaluation(lstm.model, np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), X_test, Y_test)
    logger.info("📊  LSTM metrics (test)")
    lstm.compute_metrics(lstm.model, X_test, Y_test)

# ------------------------------------------------------------------------------------
# 8 · SAVE & FINAL LOG
# ------------------------------------------------------------------------------------
training_df.to_csv(os.path.join(OUTPUT_ROOT, "feast_timeseries_features.csv"), index=False)

logger.info("🎉  Finished training with Feast‑backed time‑series features!")
logger.info(f"📝  Sequences per split → train {len(X_train):,} / test {len(X_test):,}")
logger.info(f"💡  Feature dims        → {N_FEATURES}")
