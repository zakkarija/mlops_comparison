import pandas as pd
import os
import zipfile
from datetime import datetime, timedelta


def process_real_training_data():
    """Process your actual training data into Feast format"""

    # Path to your data (relative to feature_repo directory)
    data_path = "../../data"  # Goes up to src/, then to data/

    labels = {
        "not_anomalous": "not_anomalous",
        "mechanical_anomalies": "mechanical_anomalies",
        "electrical_anomalies": "electrical_anomalies"
    }

    all_data = []
    file_counter = 0

    print(f"🔍 Looking for data in: {os.path.abspath(data_path)}")

    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        print("Current directory:", os.getcwd())
        return None

    # Process each label folder
    for folder_name, label in labels.items():
        folder_path = os.path.join(data_path, folder_name)

        if not os.path.exists(folder_path):
            print(f"⚠️  Folder not found: {folder_path}")
            continue

        print(f"📁 Processing folder: {folder_name}")

        # Get all days in the folder
        try:
            days = os.listdir(folder_path)
        except Exception as e:
            print(f"❌ Error reading folder {folder_path}: {e}")
            continue

        for day in days:
            day_path = os.path.join(folder_path, day)

            if not os.path.isdir(day_path):
                continue

            try:
                files = os.listdir(day_path)
            except Exception as e:
                print(f"❌ Error reading day {day_path}: {e}")
                continue

            # Process each zip file
            for file in files:
                if not file.endswith('.zip'):
                    continue

                file_path = os.path.join(day_path, file)

                try:
                    # Extract and read CSV from zip
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        filename = zip_file.namelist()[0]
                        with zip_file.open(filename) as csv_file:
                            df = pd.read_csv(csv_file, delimiter=";")

                    # Convert to Feast format
                    feast_data = convert_csv_to_feast_format(df, label, day, file, file_counter)
                    all_data.append(feast_data)
                    file_counter += 1

                    if file_counter % 10 == 0:
                        print(f"   Processed {file_counter} files...")

                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
                    continue

    if not all_data:
        print("❌ No data processed. Check your data path and folder structure.")
        return None

    # Combine all data
    print(f"🔗 Combining {len(all_data)} files...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save as parquet
    os.makedirs('data', exist_ok=True)
    output_path = 'data/f3_basic.parquet'
    combined_df.to_parquet(output_path, index=False)

    print(f"✅ Created {output_path} with {len(combined_df)} rows from {file_counter} files")
    print(f"📊 Labels distribution:")
    print(combined_df['label'].value_counts())
    print(f"📋 Sample data:")
    print(combined_df.head(3))
    print(f"📋 Data shape: {combined_df.shape}")
    print(f"📋 Columns: {list(combined_df.columns)}")

    return combined_df


def convert_csv_to_feast_format(df, label, day, filename, file_id):
    """Convert a single CSV DataFrame to Feast format"""

    # Create unique equipment ID for this file
    equipment_id = f"{label}_{day}_{filename.replace('.zip', '')}_{file_id}"

    # Create timestamps (assuming 2-second intervals based on your data)
    base_time = datetime.now() - timedelta(days=30) + timedelta(hours=file_id)
    df['event_timestamp'] = [base_time + timedelta(seconds=i * 2) for i in range(len(df))]
    df['created_timestamp'] = datetime.now()
    df['equipment_id'] = equipment_id
    df['f3_value'] = df['f3']  # Extract f3 column
    df['label'] = label  # Add label for this file

    # Select only the columns needed for Feast
    feast_df = df[['equipment_id', 'event_timestamp', 'created_timestamp', 'f3_value', 'label']]

    return feast_df


def verify_data_structure():
    """Helper function to check your data structure"""

    data_path = "../../data"
    print(f"🔍 Checking data structure in: {os.path.abspath(data_path)}")

    if os.path.exists(data_path):
        print("📁 Found directories:")
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                print(f"  - {item}/")
                try:
                    subdirs = os.listdir(item_path)
                    print(f"    Contains: {len(subdirs)} items")
                    if subdirs:
                        first_subdir = os.path.join(item_path, subdirs[0])
                        if os.path.isdir(first_subdir):
                            files = [f for f in os.listdir(first_subdir) if f.endswith('.zip')]
                            print(f"    Example: {subdirs[0]}/ has {len(files)} zip files")
                except Exception as e:
                    print(f"    Error reading: {e}")
    else:
        print(f"❌ Data path does not exist: {data_path}")
        print("Current working directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))


if __name__ == "__main__":
    print("🚀 Processing your real training data for Feast...")
    print("=" * 50)

    # First, verify the data structure
    verify_data_structure()
    print("=" * 50)

    # Process the data
    result = process_real_training_data()

    if result is not None:
        print("🎉 Success! Your training data is now ready for Feast.")
        print("Next step: Run 'feast apply' to register the features.")
    else:
        print("❌ Failed to process data. Check the error messages above.")