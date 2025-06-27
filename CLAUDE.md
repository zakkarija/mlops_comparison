# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLOps comparison project for manufacturing anomaly detection using deep learning. The system analyzes CNC machine sensor data (specifically the f3 current/intensity values) to detect three types of states:
- **Normal** (0): Machine operating in healthy state
- **Mechanical Anomaly** (1): Bearing/screw failures affecting one movement direction
- **Electrical Anomaly** (2): Motor/drive failures causing inconsistent acceleration peaks

The project contains **two parallel implementations** comparing different MLOps ecosystems:

### MLFlow Implementation (Primary - Near Completion)
Located in `ideko/MLFlow/` - Complete MLOps pipeline using:
- **Orchestration**: Airflow DAGs
- **Experiment Tracking**: MLflow 
- **Data Versioning**: DVC
- **Feature Store**: Feast
- **Hyperparameter Optimization**: Optuna

### Kubeflow Implementation (Placeholder - Future Work)
*To be implemented* - Will demonstrate the same pipeline using:
- **Orchestration**: Kubeflow Pipelines
- **Experiment Tracking**: Kubeflow Metadata
- **Data Versioning**: TBD
- **Feature Store**: TBD
- **Hyperparameter Optimization**: Katib

## Architecture Overview

### MLFlow Implementation Architecture
- **Data Pipeline**: Raw ZIP files → Parquet with engineered features → Feast feature store
- **Model Training**: LSTM-based multiclass classification using Keras/TensorFlow
- **Orchestration**: Airflow DAG managing the complete ML pipeline
- **Versioning**: DVC for data/model artifacts, Git for code
- **Feature Store**: Feast for feature management and serving

### Key Directories
- `ideko/MLFlow/failure-prediction-in-manufacture-main/multiclass-classification-model/`: Main ML project
- `ideko/MLFlow/airflow_home/dags/`: Airflow pipeline definitions
- `ideko/MLFlow/failure-prediction-in-manufacture-main/multiclass-classification-model/src/`: Core ML code
- `ideko/MLFlow/failure-prediction-in-manufacture-main/multiclass-classification-model/src/feast_demo/feature_repo/`: Feast configuration
- `ideko/MLFlow/failure-prediction-in-manufacture-main/multiclass-classification-model/data/`: Raw sensor data
- `ideko/MLFlow/failure-prediction-in-manufacture-main/multiclass-classification-model/artifacts/`: Model outputs and processed features

## Development Commands

**Note**: All commands below are for the MLFlow implementation. Navigate to `ideko/MLFlow/failure-prediction-in-manufacture-main/multiclass-classification-model/` first.

### Environment Setup
```bash
# Navigate to MLFlow implementation
cd ideko/MLFlow/failure-prediction-in-manufacture-main/multiclass-classification-model/

# Install dependencies
pip install -r requirements.txt

# Manual Feast installation (required for CLI)
pip install feast==0.37.1
```

### Data Processing
```bash
# Convert raw ZIP files to Parquet with features
cd src/feast_demo/feature_repo
python convert_data.py

# Apply Feast feature store configuration
feast apply

# Validate feature store
feast feature-views list
```

### Model Training and Inference
```bash
# Train models using Feast features (POC version)
cd src
python main_poc.py

# Train models with all available models (original)
python main_all_models.py

# Run inference on test data
python inference.py

# Original bare-bones version (no MLOps components)
python og.py
```

### Airflow Pipeline
```bash
# Start Airflow (from MLFlow directory)
cd ideko/MLFlow/
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow standalone

# Trigger the ML pipeline manually
airflow dags trigger ideko_multiclass_ml_pipeline
```

### DVC Operations
```bash
# Pull latest data and models (from multiclass-classification-model directory)
dvc pull --allow-missing --force

# Add new artifacts to DVC tracking
dvc add artifacts

# Push to remote storage
dvc push
```

## Kubeflow Implementation (Future Work)

### Planned Architecture
- **Pipeline Orchestration**: Kubeflow Pipelines (KFP) with Python SDK
- **Experiment Tracking**: Kubeflow Metadata and Tensorboard
- **Hyperparameter Tuning**: Katib for automated optimization
- **Model Serving**: KServe or Seldon Core
- **Data Versioning**: Integration with object storage (S3/GCS)

### Placeholder Commands
```bash
# TODO: Install Kubeflow Pipelines SDK
# pip install kfp

# TODO: Compile and run pipeline
# python kubeflow_pipeline.py

# TODO: Submit hyperparameter tuning job
# kubectl apply -f katib-experiment.yaml
```

### Directory Structure (Planned)
```
ideko/Kubeflow/
├── pipelines/           # KFP pipeline definitions
├── components/          # Reusable pipeline components
├── experiments/         # Katib experiment configs
├── models/             # Model definitions (shared with MLFlow)
└── notebooks/          # Analysis notebooks
```

## Model Configuration

Models are configured in `src/config/models.yaml`:
- **NeuralNetwork**: Simple dense layers (currently enabled)
- **CNN**: Convolutional layers for pattern detection
- **RNN**: Basic recurrent networks
- **LSTM**: Long Short-Term Memory networks

Only one model type should be enabled at a time. The LSTM model is primarily used in the Feast-integrated version (`main_poc.py`).

## Data Structure

### Raw Data Format
- High-frequency sensor data (500 Hz) from CNC machine hysteresis tests
- Files: `{timestamp}-{date}_COD020030.zip` containing CSV data
- Signals: f1 (encoder position), f2 (ruler position), f3 (current/intensity), f4 (commanded position)
- Organized into: `electrical_anomalies/`, `mechanical_anomalies/`, `not_anomalous/`

### Feature Engineering
The system creates rolling statistics from f3 current values:
- `f3_current`: Raw current reading
- `f3_rolling_mean_10`: 10-point rolling average
- `f3_rolling_std_10`: 10-point rolling standard deviation

### Sequence Processing
- Creates overlapping time series sequences (default: 10 timesteps, stride=2)
- Equipment-based train/test splitting to prevent data leakage
- Sequences are labeled with the anomaly class of the final timestep

## Important Notes

- **Equipment-based splitting**: Train/test splits are done by equipment ID, not randomly, to prevent temporal leakage
- **Feast integration**: The POC version (`main_poc.py`) uses Feast for feature serving, while other versions work directly with processed data
- **Class imbalance**: The dataset may have imbalanced classes; monitor class distributions during training
- **Deprecation**: The `binary-classification-model/` directory is deprecated; use `multiclass-classification-model/` instead
- **Performance**: Due to large dataset size, the POC processes only 20% of equipment IDs and uses batch processing for Feast feature retrieval

## Troubleshooting

- **Feast CLI issues**: Ensure Feast 0.37.1 is installed manually if Airflow conflicts occur
- **Memory issues**: The system uses garbage collection and batch processing to handle large datasets
- **DVC pull failures**: Use `--allow-missing --force` flags for partial data retrieval
- **Model overfitting**: Monitor train/test accuracy gap; >15% gap indicates overfitting