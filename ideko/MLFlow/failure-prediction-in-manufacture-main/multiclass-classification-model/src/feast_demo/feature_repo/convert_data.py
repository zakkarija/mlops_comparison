import os
import sys
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def find_data_directory():
    """Find the data directory by looking in common locations"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    possible_paths = [
        os.path.join(current_dir, '../../data'),
        os.path.join(current_dir, '../../../data'),
        os.path.join(current_dir, '../data'),
        os.path.join(current_dir, 'data'),
    ]

    project_root = current_dir
    while project_root != '/' and not os.path.exists(os.path.join(project_root, 'multiclass-classification-model')):
        project_root = os.path.dirname(project_root)

    if os.path.exists(os.path.join(project_root, 'multiclass-classification-model')):
        possible_paths.extend([
            os.path.join(project_root, 'multiclass-classification-model', 'data'),
            os.path.join(project_root, 'data_subset'),
        ])

    print(f"🔍 Looking for data directory from: {current_dir}")

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"  ✅ Found data directory at: {abs_path}")
            return abs_path

    return None


def detect_movement_direction(position_data):
    """Detect movement direction from position data"""
    if len(position_data) < 2:
        return 0

    velocity = np.diff(position_data)
    avg_velocity = np.mean(velocity)

    if avg_velocity > 0.1:
        return 1  # Forward
    elif avg_velocity < -0.1:
        return -1  # Backward
    else:
        return 0  # Stationary


def calculate_cycle_position(position_data):
    """Calculate position within the movement cycle (0-1)"""
    if len(position_data) < 2:
        return 0.0

    min_pos = np.min(position_data)
    max_pos = np.max(position_data)

    if max_pos == min_pos:
        return 0.0

    current_pos = position_data[-1]
    normalized_pos = (current_pos - min_pos) / (max_pos - min_pos)

    return float(normalized_pos)


def process_timeseries_data():
    """Process manufacturing data into time series features for Feast"""
    print("🚀 Processing time series data for Feast...")
    print("=" * 50)

    data_path = find_data_directory()
    if not data_path:
        print("❌ Could not find data directory!")
        return False

    expected_dirs = ['electrical_anomalies', 'mechanical_anomalies', 'not_anomalous']
    missing_dirs = [d for d in expected_dirs if not os.path.exists(os.path.join(data_path, d))]

    if missing_dirs:
        print(f"❌ Missing required subdirectories: {missing_dirs}")
        return False

    print(f"✅ Using data directory: {data_path}")

    labels = {
        "not_anomalous": "normal",
        "mechanical_anomalies": "mechanical_anomaly",
        "electrical_anomalies": "electrical_anomaly"
    }

    all_timeseries_data = []

    for folder_name, label in labels.items():
        folder_path = os.path.join(data_path, folder_name)

        print(f"📁 Processing {folder_name}...")

        try:
            days = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        except Exception as e:
            print(f"❌ Error reading folder {folder_path}: {e}")
            continue

        for day in days:
            day_path = os.path.join(folder_path, day)

            try:
                files = [f for f in os.listdir(day_path) if f.endswith('.zip')]
            except Exception as e:
                continue

            for file in files:
                file_path = os.path.join(day_path, file)

                try:
                    filename_parts = file.replace('.zip', '').split('-')
                    if len(filename_parts) >= 2:
                        base_timestamp_ms = int(filename_parts[0])
                        base_timestamp = datetime.fromtimestamp(base_timestamp_ms / 1000.0)
                    else:
                        base_timestamp = datetime.now()

                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        csv_filename = zip_file.namelist()[0]

                        with zip_file.open(csv_filename) as csv_file:
                            df = pd.read_csv(csv_file, delimiter=';')

                            if 'f3' not in df.columns:
                                continue

                            equipment_id = file.replace('.zip', '')

                            f3_data = df['f3'].values
                            position_data = df['f2'].values if 'f2' in df.columns else np.arange(len(f3_data))

                            # Sample the data
                            sample_rate = 10
                            f3_sampled = f3_data[::sample_rate]
                            position_sampled = position_data[::sample_rate]

                            time_intervals = [base_timestamp + timedelta(seconds=i * 0.002 * sample_rate)
                                              for i in range(len(f3_sampled))]

                            for i, (timestamp, f3_value) in enumerate(zip(time_intervals, f3_sampled)):
                                start_10 = max(0, i - 9)
                                start_50 = max(0, i - 49)
                                start_100 = max(0, i - 99)

                                window_10 = f3_sampled[start_10:i + 1]
                                window_50 = f3_sampled[start_50:i + 1]
                                window_100 = f3_sampled[start_100:i + 1]

                                position_window = position_sampled[start_50:i + 1]

                                features = {
                                    'event_timestamp': timestamp,
                                    'created_timestamp': timestamp,
                                    'equipment_id': equipment_id,

                                    'f3_current': float(f3_value),
                                    'f3_rolling_mean_10': float(np.mean(window_10)),
                                    'f3_rolling_mean_50': float(np.mean(window_50)),
                                    'f3_rolling_mean_100': float(np.mean(window_100)),
                                    'f3_rolling_std_10': float(np.std(window_10)) if len(window_10) > 1 else 0.0,
                                    'f3_rolling_std_50': float(np.std(window_50)) if len(window_50) > 1 else 0.0,
                                    'f3_rolling_min_50': float(np.min(window_50)),
                                    'f3_rolling_max_50': float(np.max(window_50)),
                                    'f3_rate_of_change': float(window_10[-1] - window_10[0]) if len(
                                        window_10) > 1 else 0.0,
                                    'movement_direction': detect_movement_direction(position_window),
                                    'cycle_position': calculate_cycle_position(position_window),
                                    'anomaly_class': 0 if label == 'normal' else (
                                        1 if label == 'mechanical_anomaly' else 2),
                                    'label': label
                                }

                                all_timeseries_data.append(features)

                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
                    continue

    if not all_timeseries_data:
        print("❌ No data was successfully processed!")
        return False

    timeseries_df = pd.DataFrame(all_timeseries_data)

    timeseries_df['event_timestamp'] = pd.to_datetime(timeseries_df['event_timestamp'])
    timeseries_df['created_timestamp'] = pd.to_datetime(timeseries_df['created_timestamp'])

    timeseries_df = timeseries_df.sort_values(['equipment_id', 'event_timestamp'])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    offline_dir = os.path.join(current_dir, "data", "offline")
    os.makedirs(offline_dir, exist_ok=True)

    out_path = os.path.join(offline_dir, "f3_timeseries.parquet")
    timeseries_df.to_parquet(out_path, index=False)

    print(f"✅ wrote {len(timeseries_df)} rows → {out_path}")

    print(f"\n📊 Data Summary:")
    print(f"   Total time series points: {len(timeseries_df)}")
    print(f"   Unique equipment: {timeseries_df['equipment_id'].nunique()}")
    print(f"   Date range: {timeseries_df['event_timestamp'].min()} to {timeseries_df['event_timestamp'].max()}")

    label_counts = timeseries_df['label'].value_counts()
    print(f"   Label distribution:")
    for label, count in label_counts.items():
        print(f"     - {label}: {count}")

    return True


if __name__ == "__main__":
    success = process_timeseries_data()
    if not success:
        print("\n❌ Failed to process data.")
        sys.exit(1)
    else:
        print("\n🎉 Time series data processing completed!")
        print("You can now run: feast apply")