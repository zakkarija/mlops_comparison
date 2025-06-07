import os
import sys
from helpers.logger import LoggerHelper, logging
from helpers.config import ConfigHelper
import numpy as np
import pandas as pd
from classes import preprocessing_functions
from classes.multiclass_models import LongShortTermMemory
from feast import FeatureStore
import gc
from sklearn.model_selection import train_test_split

# Initialize logging and configuration
LoggerHelper.init_logger()
logger = logging.getLogger(__name__)
config = ConfigHelper.instance("models")

# Set up output directory for results
output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
os.makedirs(output_path, exist_ok=True)

# Initialize Feast feature store
feast_repo_path = "feast_demo/feature_repo"
fs = FeatureStore(repo_path=feast_repo_path)

################################
# STEP 1: DEFINE FEATURES TO USE
################################
FEATURE_COLUMNS = [
    "f3_current",           # Current sensor reading
    "f3_rolling_mean_10",   # Average of last 10 readings
    "f3_rolling_std_10",    # Standard deviation of last 10 readings  
    "movement_direction",   # Direction of movement (-1, 0, 1)
]

minimal_feast_features = [
    f"f3_timeseries_features:{col}"
    for col in FEATURE_COLUMNS
]

################################
# STEP 2: LOAD RAW DATA FROM PARQUET & SAMPLE BY EQUIPMENT_ID
################################
logger.info("Loading data from Parquet file...")

# Load the entire dataset from Parquet
timeseries_df = pd.read_parquet(
    os.path.join(feast_repo_path, "data", "offline", "f3_timeseries.parquet")
)

# Instead of sampling 20% of rows, pick 20% of equipment IDs and keep all their rows
all_equip_ids = timeseries_df["equipment_id"].unique()
n_equip_to_keep = int(len(all_equip_ids) * 0.2)
# Randomly pick 20% of equipment IDs
kept_equip_ids = np.random.RandomState(seed=42).choice(
    all_equip_ids, size=n_equip_to_keep, replace=False
)
timeseries_df = timeseries_df[timeseries_df["equipment_id"].isin(kept_equip_ids)].copy()
logger.info(
    f"Kept {n_equip_to_keep} equipment IDs out of {len(all_equip_ids)} "
    f"({100 * n_equip_to_keep / len(all_equip_ids):.1f}% of machines)"
)
logger.info(f"Remaining rows: {len(timeseries_df)}")

################################
# STEP 3: CREATE ENTITY DATAFRAME
################################
entity_df = timeseries_df[["equipment_id", "event_timestamp"]].copy()

################################
# STEP 4: FETCH FEATURES IN BATCHES
################################
batch_size = 2_000      # Process 2000 rows at a time
max_batches = 10        # Due to POC only 10 batches => 20 000 rows
n_batches = min(
    max_batches, (len(entity_df) - 1) // batch_size + 1
)
all_features = []

logger.info(f"Fetching features from Feast in {n_batches} batches...")

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(entity_df))
    batch_entity = entity_df.iloc[start_idx:end_idx]

    logger.info(f"Processing batch {i+1}/{n_batches} (rows {start_idx}-{end_idx})")
    batch_features = fs.get_historical_features(
        entity_df=batch_entity,
        features=minimal_feast_features + ["f3_timeseries_features:anomaly_class"],
    ).to_df()
    all_features.append(batch_features)
    gc.collect()

training_df = pd.concat(all_features, ignore_index=True)
del all_features
gc.collect()
logger.info(f"Total features retrieved: {training_df.shape}")

# Log class distribution in the raw “full” served data
class_counts = training_df["anomaly_class"].value_counts().sort_index()
logger.info("Class distribution in served data:")
for class_id, count in class_counts.items():
    logger.info(f"  Class {class_id}: {count} samples")

################################
# STEP 5: CREATE TIME SERIES SEQUENCES FOR ALL DATA (temporarily)
################################
# We’ll build sequences **after** splitting by equipment to avoid leakage.
# But let’s just confirm we have data for each equipment:
unique_equipment = training_df["equipment_id"].unique()
logger.info(f"Will process {len(unique_equipment)} equipment IDs total")

# (No sequence creation here yet—will do it in STEP 6.)

################################
# STEP 6: SPLIT BY EQUIPMENT, THEN BUILD SEQUENCES
################################
# 6A: Split equipment IDs into train vs. test (80/20)
all_equip_ids = training_df["equipment_id"].unique()
train_equip, test_equip = train_test_split(
    all_equip_ids,
    test_size=0.2,
    random_state=123,
    stratify=training_df.groupby("equipment_id")["anomaly_class"].agg(
        lambda x: x.value_counts().idxmax()
    ).reindex(all_equip_ids)
)
logger.info(f"Training on {len(train_equip)} equipment IDs; testing on {len(test_equip)}")

# 6B: Separate the DataFrame by those splits
df_train_equip = training_df[training_df["equipment_id"].isin(train_equip)].copy()
df_test_equip = training_df[training_df["equipment_id"].isin(test_equip)].copy()

# 6C: Now create overlapping sequences **per-split** to guarantee no overlap leak
def create_sequences_with_overlap(df, sequence_length=10, stride=2):
    """
    Create overlapping sequences from time-series data for each equipment in df.
    Returns: (X, y) where
      - X is shape (num_sequences, sequence_length, n_features)
      - y is shape (num_sequences,) with integer labels in {0,1,2}
    """
    feature_columns = FEATURE_COLUMNS
    sequences = []
    labels = []
    equipment_sequence_counts = {}

    for equipment_id in df["equipment_id"].unique():
        equip_data = (
            df[df["equipment_id"] == equipment_id]
            .sort_values("event_timestamp")
        )

        if len(equip_data) < sequence_length:
            # Not enough rows to form a single sequence
            equipment_sequence_counts[equipment_id] = 0
            continue

        feat_vals = equip_data[feature_columns].values
        lbl_vals = equip_data["anomaly_class"].values

        seq_count = 0
        for start in range(0, len(feat_vals) - sequence_length + 1, stride):
            seq = feat_vals[start : start + sequence_length]
            label = lbl_vals[start + sequence_length - 1]
            sequences.append(seq)
            labels.append(label)
            seq_count += 1

        equipment_sequence_counts[equipment_id] = seq_count

    # Log a few equipment sequence counts
    logger.info("Sequences created per equipment (first 5 shown):")
    for eq_id, count in list(equipment_sequence_counts.items())[:5]:
        logger.info(f"  {eq_id}: {count} sequences")

    return np.array(sequences), np.array(labels)

# 6D: Build train sequences
sequence_length = 10  # size of each LSTM window
stride = 2            # overlap stride
X_train, y_train = create_sequences_with_overlap(
    df_train_equip, sequence_length=sequence_length, stride=stride
)

# 6E: Build test sequences
X_test, y_test = create_sequences_with_overlap(
    df_test_equip, sequence_length=sequence_length, stride=stride
)

logger.info(f"Built {len(X_train)} train sequences and {len(X_test)} test sequences")

# One‐hot encode labels
label_map = {0: "normal", 1: "mechanical_anomaly", 2: "electrical_anomaly"}
y_train_text = [label_map[l] for l in y_train]
y_test_text = [label_map[l] for l in y_test]
Y_train_encoded = preprocessing_functions.encode_response_variable(y_train_text)
Y_test_encoded = preprocessing_functions.encode_response_variable(y_test_text)

logger.info("Train class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    logger.info(f"  Class {u} ({label_map[u]}): {c} sequences")

logger.info("Test class distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    logger.info(f"  Class {u} ({label_map[u]}): {c} sequences")

# Clean up raw DataFrames
del training_df, df_train_equip, df_test_equip, timeseries_df
gc.collect()

################################
# STEP 7: TRAIN LSTM MODEL
################################
n_timestamps = sequence_length
n_features = len(FEATURE_COLUMNS)
n_classes = Y_train_encoded.shape[1]

logger.info(f"Model inputs: {n_timestamps} timesteps × {n_features} features, {n_classes} classes")

# Hyperparameters
activation_function = "tanh"
hidden_units = [64, 32]
epochs = 20
batch_size = 32

model_lstm = LongShortTermMemory(
    n_timestamps, n_features, activation_function, hidden_units, n_classes
)
model_lstm.create_model()

early_stopping = model_lstm.early_stopping_callback(patience=5)
model_checkpoint = model_lstm.model_checkpoint_callback(
    model_path=os.path.join(output_path, "lstm_feast_poc.keras")
)
callbacks = [early_stopping, model_checkpoint]

model_lstm.model_compilation(model_lstm.model)
history_lstm = model_lstm.model_fitting(
    model_lstm.model,
    X_train,
    Y_train_encoded,
    X_test,
    Y_test_encoded,
    callbacks,
    epochs,
    batch_size,
)
preprocessing_functions.plot_model_history(history_lstm, output_path)

################################
# STEP 8: EVALUATE MODEL PROPERLY
################################
logger.info("\n" + "=" * 60)
logger.info("MODEL EVALUATION")
logger.info("=" * 60)

# 8A: Full‐dataset evaluation (sanity, but trains on train+test)
#    We skip this if we strictly want only test metrics, but we include it here for completeness.
full_X = np.concatenate([X_train, X_test], axis=0)
full_Y = np.concatenate([Y_train_encoded, Y_test_encoded], axis=0)
model_lstm.model_evaluation(model_lstm.model, full_X, full_Y, X_test, Y_test_encoded)

# 8B: Test‐only evaluation (the true hold‐out score)
logger.info("\nTEST SET EVALUATION (Most Important):")
model_lstm.compute_metrics(model_lstm.model, X_test, Y_test_encoded)

# 8C: Train‐only evaluation (check for overfitting)
logger.info("\nTRAIN SET EVALUATION (For Comparison):")
model_lstm.compute_metrics(model_lstm.model, X_train, Y_train_encoded)

# 8D: Numeric train/test accuracy gap
train_loss, train_acc = model_lstm.model.evaluate(X_train, Y_train_encoded, verbose=0)
test_loss, test_acc = model_lstm.model.evaluate(X_test, Y_test_encoded, verbose=0)
accuracy_gap = train_acc - test_acc

logger.info("\nOVERFITTING CHECK:")
logger.info(f"  Training accuracy: {train_acc:.3f}")
logger.info(f"  Test accuracy: {test_acc:.3f}")
logger.info(f"  Accuracy gap: {accuracy_gap:.3f}")
if test_acc > 0.95:
    logger.warning(
        "Test accuracy > 95% — verify no leakage or too little data!"
    )
elif accuracy_gap > 0.15:
    logger.warning("Large accuracy gap indicates overfitting!")
else:
    logger.info("✅ Model appears to generalize well")

# Final summary
logger.info("\n" + "=" * 60)
logger.info("✅ Feast POC completed successfully!")
logger.info(f"📊 Created {len(X_train)+len(X_test)} total sequences "
            f"({n_timestamps} timesteps × {n_features} features)")
logger.info(f"🎯 Train accuracy = {train_acc:.3f}, Test accuracy = {test_acc:.3f}")
logger.info(f"💾 Model saved to: {os.path.join(output_path, 'lstm_feast_poc.keras')}")
logger.info("=" * 60)