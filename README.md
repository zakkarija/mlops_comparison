# MLOps Comparison Thesis

**Overview**  
This repository contains a master's thesis comparing two end-to-end MLOps stacks applied to industrial failure prediction in manufacturing equipment:
- **Stack A (MLflow-centric):** MLflow + Airflow + Feast + DVC + Optuna
- **Stack B (Kubeflow-centric):** Kubeflow Pipelines + Katib + MLMD + Model Registry + LakeFS

**Use Case: Industrial Anomaly Detection**  
Time-series classification of machine tool failures using high-frequency sensor data from grinding machine movements. The dataset contains three classes:
- **Not Anomalous**: Normal machine operation (intensity ~4/-4)
- **Mechanical Anomalies**: Bearing/screw failures (asymmetric noise patterns) 
- **Electrical Anomalies**: Motor/drive failures (acceleration inconsistencies, peaks ~10/-10)

**Dataset Characteristics:**
- **Sensor Data**: Position encoders (f1,f2,f4) + current intensity (f3) at 500Hz
- **Movement Pattern**: 3x backward-forward hysteresis cycles per test
- **File Structure**: Timestamped CSV files per test cycle
- **Challenge**: Variable-length sequences requiring padding and deep learning models

## Current Implementation Status

### ✅ Stack A (MLflow) - COMPLETED (airflow branch)
**Environment**: `ideko-airflow` pyenv

**Implemented Components:**
- **Airflow Orchestration**: Complete DAG with 10 sequential tasks
- **MLflow Tracking**: Full experiment tracking with autologging + model registry
- **Feast Feature Store**: Time-series feature engineering (rolling stats, windowing)
- **DVC Data Versioning**: Dataset versioning and artifact management  
- **Optuna Hyperparameter Tuning**: Automated Neural Network optimization
- **Model Architecture**: 4 configurable models (NN, CNN, RNN, LSTM)

**Key Features:**
- End-to-end pipeline from raw data to model evaluation
- Sequence generation (seq_len=10, stride=2) for time-series modeling
- Equipment-stratified train/test splits maintaining class distribution
- Comprehensive validation at each pipeline stage
- Resource monitoring (CPU, memory, GPU) during training
- Automated model comparison and metrics collection

### 🚧 Stack B (Kubeflow) - IN DEVELOPMENT (kubeflow branch)
**Environment**: `ideko-kubeflow` pyenv

**Current Status**: Basic model training code exists, Kubeflow components pending

**Planned Implementation Phases:**

#### Phase 1: Kubeflow Pipelines Foundation
- Convert existing training script into pipeline components
- Implement data preprocessing, model training, and evaluation components
- Create pipeline definitions with parameter passing
- Add artifact management and metadata tracking

#### Phase 2: Advanced Pipeline Features  
- Implement parallel model training (NN, CNN, RNN, LSTM)
- Add conditional execution and dynamic branching
- Implement pipeline caching and resumption
- Add pipeline versioning and comparison

#### Phase 3: Katib Integration
- Hyperparameter optimization for all model architectures
- Multi-objective optimization (accuracy vs training time)
- Automated architecture search
- Resource-aware optimization

#### Phase 4: MLMD & Model Registry
- Experiment lineage tracking
- Model versioning and comparison
- Automated model evaluation and selection
- Model deployment pipeline integration

#### Phase 5: LakeFS Integration
- Data versioning and branching
- Experiment reproducibility
- Dataset lineage tracking
- Data quality monitoring

## Evaluation Framework

**Comparison Dimensions:**
1. **Usability**: Developer experience, learning curve, debugging capabilities
2. **Flexibility**: Customization, extensibility, integration options  
3. **Integration Effort**: Setup complexity, configuration management
4. **Vitality**: Community support, documentation quality, ecosystem maturity
5. **Performance**: Execution speed, resource utilization, scalability
6. **Reliability**: Error handling, monitoring, failure recovery

**Test Scenarios:**
- Pipeline development and modification
- Hyperparameter optimization workflows  
- Model comparison and selection
- Data versioning and reproducibility
- Failure handling and recovery
- Resource scaling and optimization

## Getting Started

### Stack A (MLflow + Airflow)
```bash
git checkout airflow
pyenv activate ideko-airflow
# Follow ideko/MLFlow/README.md for detailed setup
```

### Stack B (Kubeflow) 
```bash
git checkout kubeflow  
pyenv activate ideko-kubeflow
# Implementation in progress - see detailed plan below
```

## Repository Structure

```
ideko/
├── MLFlow/                          # Stack A Implementation
│   ├── airflow_home/dags/          # Airflow pipeline definitions
│   ├── experimentation-engine/     # MLflow experiment tracking
│   └── failure-prediction/         # Core ML models + Feast integration
└── kubeflow/                       # Stack B Implementation  
    └── multiclass-classification-model/  # Kubeflow pipeline components
```

---

**Next Steps**: Complete Kubeflow implementation following the detailed plan in `docs/kubeflow-implementation-plan.md`