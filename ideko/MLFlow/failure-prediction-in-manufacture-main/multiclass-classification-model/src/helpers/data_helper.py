import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from feast import FeatureStore
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DataHelper:
    """
    Helper class for data processing operations including:
    - Reading data from Feast in batches (memory-efficient)
    - Train/test splitting with stratification
    - Building sequences for time series models
    - Label encoding for multiclass classification

    This implementation aligns with the original og.py while
    maintaining memory optimization improvements.
    """

    @staticmethod
    def batch_read_feast_data(
        feast_repo_path: str,
        entity_df: pd.DataFrame,
        feature_columns: List[str],
        label_feature: str,
        batch_size: int = 2000,
        max_batches: int = 10
    ) -> pd.DataFrame:
        """
        Read data from Feast in batches to handle large datasets efficiently.
        
        Args:
            feast_repo_path: Path to the Feast repository
            entity_df: DataFrame containing entity and timestamp information
            feature_columns: List of feature column names
            label_feature: The name of the label feature
            batch_size: Size of each batch to process
            max_batches: Maximum number of batches to process
            
        Returns:
            Combined DataFrame with features and labels
        """
        logger.info(f"Reading data from Feast in batches (size: {batch_size}, max: {max_batches})")
        fs = FeatureStore(repo_path=feast_repo_path)
        feast_features = [f"f3_timeseries_features:{c}" for c in feature_columns]
        
        batches = []
        for i in range(min((len(entity_df) + batch_size - 1) // batch_size, max_batches)):
            logger.info(f"Processing batch {i+1}/{min((len(entity_df) + batch_size - 1) // batch_size, max_batches)}")
            e = entity_df.iloc[i * batch_size:(i + 1) * batch_size]
            dfb = fs.get_historical_features(
                entity_df=e,
                features=feast_features + [label_feature],
            ).to_df()
            batches.append(dfb)
            gc.collect()
        
        training_df = pd.concat(batches, ignore_index=True)
        training_df.rename(columns={label_feature: "anomaly_class"}, inplace=True)

        logger.info(
            "Final class counts: %s",
            training_df["anomaly_class"].value_counts().to_dict()
        )

        missing = {0, 1, 2} - set(training_df["anomaly_class"].unique())
        if missing:
            raise ValueError(f"Still missing class(es) {missing}; raise max_batches or debug entity_df selection.")

        # Log class distribution for debugging
        class_counts = training_df["anomaly_class"].value_counts()
        logger.info(f"Class distribution in loaded data: {class_counts.to_dict()}")

        return training_df
    
    @staticmethod
    def stratified_equipment_split(
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 123
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by equipment ID with stratification based on most common anomaly class.
        Similar to train_test_split in the original code but preserving equipment boundaries.

        Args:
            df: DataFrame containing equipment_id and anomaly_class columns
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Performing stratified split by equipment ID")
        all_eq = df["equipment_id"].unique()
        logger.info(f"Total unique equipment IDs: {len(all_eq)}")

        # For each equipment, get the most common class
        strat = (
            df.groupby("equipment_id")["anomaly_class"]
            .agg(lambda s: s.value_counts().idxmax())
            .reindex(all_eq)
        )
        
        # Count unique classes in stratification variable for debugging
        class_counts = strat.value_counts()
        logger.info(f"Equipment distribution by dominant class: {class_counts.to_dict()}")

        train_eq, test_eq = train_test_split(
            all_eq, test_size=test_size, random_state=random_state, stratify=strat
        )
        
        train_df = df[df["equipment_id"].isin(train_eq)]
        test_df = df[df["equipment_id"].isin(test_eq)]
        
        logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

        return train_df, test_df
    
    @staticmethod
    def build_sequences(
        df: pd.DataFrame, 
        feature_columns: List[str],
        seq_len: int = 10, 
        stride: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sequences for time series modeling from a DataFrame.
        Creates sliding windows from time series data.

        Args:
            df: DataFrame containing time series data
            feature_columns: List of feature column names to include in sequences
            seq_len: Length of each sequence
            stride: Step size between consecutive sequences
            
        Returns:
            Tuple of (X, y) arrays for modeling
        """
        logger.info(f"Building sequences with length={seq_len}, stride={stride}")
        X, y = [], []

        # Process each equipment separately to maintain time continuity
        equipment_count = len(df["equipment_id"].unique())
        sequence_count = 0

        for eq in df["equipment_id"].unique():
            sub = df[df["equipment_id"] == eq].sort_values("event_timestamp")
            features = sub[feature_columns].values
            labels = sub["anomaly_class"].values

            if len(features) < seq_len:
                continue
                
            for i in range(0, len(features) - seq_len + 1, stride):
                X.append(features[i:i + seq_len])
                y.append(labels[i + seq_len - 1])
                sequence_count += 1

        X_array = np.array(X)
        y_array = np.array(y)

        logger.info(f"Created {sequence_count} sequences from {equipment_count} equipment IDs")
        logger.info(f"X shape: {X_array.shape}, y shape: {y_array.shape}")

        # Debug: Check class distribution in sequences
        unique_classes, class_counts = np.unique(y_array, return_counts=True)
        logger.info(f"Class distribution in sequences: {dict(zip(unique_classes, class_counts))}")

        return X_array, y_array

    @staticmethod
    def encode_labels(
        y_int: np.ndarray, 
        label_map: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """
        Encode integer labels to text representation.

        Args:
            y_int: Array of integer class labels
            label_map: Dictionary mapping integers to class names
            
        Returns:
            Array of text labels
        """
        if label_map is None:
            label_map = {0: "normal", 1: "mechanical_anomaly", 2: "electrical_anomaly"}

        # Convert integers to text labels
        y_txt = np.vectorize(label_map.get)(y_int)
        
        # Log unique labels to verify correct encoding
        unique_labels = np.unique(y_txt)
        logger.info(f"Encoded labels to: {unique_labels}")

        return y_txt

    @staticmethod
    def encode_response_variable(labels: np.ndarray) -> np.ndarray:
        """
        One-hot encode text labels for multiclass classification.
        Similar to encode_response_variable in preprocessing_functions but
        duplicated here for clarity.

        Args:
            labels: Array of text labels

        Returns:
            One-hot encoded array
        """
        # Get unique classes (sorted to ensure consistent order)
        classes = np.sort(np.unique(labels))
        logger.info(f"Encoding {len(classes)} unique classes: {classes}")

        # Initialize one-hot encoded matrix
        n_samples = len(labels)
        n_classes = len(classes)
        encoded = np.zeros((n_samples, n_classes))

        # Fill in one-hot encodings
        for i, label in enumerate(labels):
            class_index = np.where(classes == label)[0][0]
            encoded[i, class_index] = 1

        logger.info(f"One-hot encoded shape: {encoded.shape}")
        return encoded
