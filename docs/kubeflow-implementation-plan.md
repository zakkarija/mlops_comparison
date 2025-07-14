# Kubeflow Implementation Plan

## Overview
This document outlines the detailed implementation plan for Stack B (Kubeflow-centric) of the MLOps comparison thesis. The implementation will mirror and extend the functionality of Stack A (MLflow + Airflow) to enable fair comparison.

## Current Code Analysis

### Existing Assets (kubeflow branch)
- **Base Model Training Script**: `ideko/kubeflow/multiclass-classification-model/src/main.py`
- **Model Architectures**: Neural Network, CNN, RNN, LSTM implementations
- **Configuration System**: YAML-based model configuration in `models.yaml`
- **Preprocessing Pipeline**: Data loading, padding, encoding, train/test splitting
- **Helper Infrastructure**: Logging, configuration management

### Code Structure Analysis
```python
# Current main.py structure (341 lines):
# 1. Data Preprocessing (lines 22-77)
#    - Data loading from local path
#    - Padding for variable-length sequences  
#    - Response variable encoding
#    - Train/test splitting
#
# 2. Model Training Loops (lines 78-341)
#    - Neural Network (lines 84-145)
#    - CNN (lines 148-212) 
#    - RNN (lines 216-276)
#    - LSTM (lines 280-341)
#    - Each with: config loading, model creation, training, evaluation
```

## Implementation Phases

### Phase 1: Kubeflow Pipelines Foundation

#### 1.1 Environment Setup
```bash
# Create ideko-kubeflow pyenv
pyenv virtualenv 3.10.14 ideko-kubeflow
pyenv activate ideko-kubeflow

# Install base requirements
pip install -r ideko/kubeflow/multiclass-classification-model/requirements.txt

# Add Kubeflow Pipelines SDK
pip install kfp==2.4.0
pip install kfp-server-api==2.0.5
```

#### 1.2 Component Creation Strategy
Transform the monolithic `main.py` into modular Kubeflow components:

**Component 1: Data Preprocessing**
- **Input**: Data path, indicator list
- **Output**: Processed datasets (X_train, X_test, y_train, y_test)
- **Functionality**: Read data, padding, encoding, splitting
- **Artifacts**: Dataset metadata, class distribution statistics

**Component 2: Model Training** (per architecture)
- **Input**: Training data, model configuration
- **Output**: Trained model, training history
- **Functionality**: Model creation, compilation, training with callbacks
- **Artifacts**: Model files, training plots, metrics

**Component 3: Model Evaluation**
- **Input**: Trained model, test data
- **Output**: Evaluation metrics, predictions
- **Functionality**: Model evaluation, metrics computation
- **Artifacts**: Confusion matrices, classification reports

**Component 4: Results Aggregation**
- **Input**: Multiple model evaluation results
- **Output**: Comparison results, best model selection
- **Functionality**: Model comparison, performance analysis
- **Artifacts**: Comparison charts, model rankings

#### 1.3 Pipeline Definition
```python
@dsl.pipeline(
    name="ideko-multiclass-classification",
    description="Industrial anomaly detection pipeline"
)
def ideko_pipeline(
    data_path: str = "/data",
    indicator_list: list = ["f3"],
    models_config: dict = {...}
):
    # Data preprocessing component
    prep_task = data_preprocessing_op(
        data_path=data_path,
        indicator_list=indicator_list
    )
    
    # Parallel model training
    model_tasks = []
    for model_type in ["NeuralNetwork", "CNN", "RNN", "LSTM"]:
        if models_config[model_type]["enabled"]:
            model_task = model_training_op(
                model_type=model_type,
                config=models_config[model_type],
                train_data=prep_task.outputs["train_data"],
                test_data=prep_task.outputs["test_data"]
            )
            model_tasks.append(model_task)
    
    # Model evaluation and comparison
    eval_task = model_comparison_op(
        model_results=[task.outputs["model"] for task in model_tasks]
    )
```

### Phase 2: Advanced Pipeline Features

#### 2.1 Parallel Execution Implementation
- **Parallel Model Training**: Execute NN, CNN, RNN, LSTM simultaneously
- **Resource Management**: Configure memory/CPU requirements per component
- **Dynamic Pipeline Generation**: Enable/disable models based on configuration

#### 2.2 Conditional Execution
```python
# Conditional model training based on previous results
with dsl.Condition(prep_task.outputs["data_quality"] == "PASS"):
    # Only proceed with training if data validation passes
    model_training_tasks = [...]
```

#### 2.3 Pipeline Caching and Resumption
- **Component Caching**: Reuse preprocessing results across runs
- **Artifact Management**: Efficient storage and retrieval of intermediate results
- **Pipeline Versioning**: Track pipeline evolution and changes

#### 2.4 Error Handling and Retry Logic
```python
@dsl.component(
    packages_to_install=["tensorflow==2.16.1"],
    retry_policy=dsl.RetryPolicy(max_retry_count=3, backoff_duration="30s")
)
def model_training_op(...):
    # Training logic with error handling
```

### Phase 3: Katib Integration

#### 3.1 Hyperparameter Optimization Setup
- **Multi-Model Optimization**: Optimize hyperparameters for all architectures
- **Search Algorithms**: Grid Search, Random Search, Bayesian Optimization
- **Multi-Objective Optimization**: Balance accuracy vs training time

#### 3.2 Katib Experiment Configuration
```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: ideko-multiclass-hpo
spec:
  algorithm:
    algorithmName: bayesian-optimization
  objective:
    type: maximize
    goal: 0.95
    objectiveMetricName: validation_accuracy
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.0001"
        max: "0.1"
    - name: batch_size
      parameterType: categorical
      feasibleSpace:
        list: ["16", "32", "64"]
    - name: units_layer1
      parameterType: int
      feasibleSpace:
        min: "32"
        max: "128"
```

#### 3.3 Integration with Pipeline
- **Dynamic Parameter Injection**: Use Katib suggestions in pipeline components
- **Trial Management**: Track and compare hyperparameter experiments
- **Early Stopping**: Implement resource-efficient optimization

### Phase 4: MLMD & Model Registry

#### 4.1 Metadata Tracking Implementation
- **Experiment Lineage**: Track data sources, transformations, and model training
- **Artifact Relationships**: Link datasets, models, and evaluation results
- **Execution Tracking**: Monitor pipeline runs and component performance

#### 4.2 Model Registry Integration
```python
from kubeflow.metadata import client

# Model registration
metadata_client = client.Client()
model_artifact = metadata_client.create_artifact(
    artifact_type="Model",
    name=f"ideko-{model_type}-{version}",
    properties={
        "accuracy": accuracy_score,
        "architecture": model_type,
        "training_time": training_duration
    }
)
```

#### 4.3 Automated Model Selection
- **Performance-Based Selection**: Choose best model based on validation metrics
- **Multi-Criteria Decision**: Consider accuracy, training time, model size
- **Model Promotion**: Automated staging and production deployment

### Phase 5: LakeFS Integration

#### 5.1 Data Versioning Setup
- **Data Branch Management**: Create branches for different experiment datasets
- **Commit-based Versioning**: Version control for data changes
- **Merge and Tag Operations**: Manage data evolution

#### 5.2 Integration with Pipeline
```python
# LakeFS data loading component
@dsl.component
def lakefs_data_loader_op(
    lakefs_repo: str,
    branch: str,
    commit_id: str = "latest"
) -> str:
    # Load data from specific LakeFS commit
    # Return data path for downstream components
```

#### 5.3 Experiment Reproducibility
- **Immutable Dataset References**: Link experiments to specific data commits
- **Automated Data Lineage**: Track data transformations and versions
- **Rollback Capabilities**: Revert to previous data states

## Detailed Implementation Steps

### Step 1: Setup Development Environment
1. Create `ideko-kubeflow` pyenv environment
2. Install current requirements + Kubeflow SDK
3. Set up local Kubeflow cluster (minikube + Kubeflow)
4. Validate environment with hello-world pipeline

### Step 2: Component Development
1. Extract data preprocessing logic into standalone component
2. Create model training components for each architecture
3. Implement evaluation and comparison components
4. Test components individually with sample data

### Step 3: Pipeline Assembly
1. Create main pipeline definition connecting all components
2. Implement parameter passing and artifact management
3. Add pipeline-level configuration and error handling
4. Test end-to-end pipeline execution

### Step 4: Advanced Features Implementation
1. Add parallel execution for model training
2. Implement conditional logic and dynamic branching
3. Add caching and resumption capabilities
4. Integrate comprehensive monitoring and logging

### Step 5: Katib Integration
1. Design hyperparameter search spaces for each model
2. Create Katib experiment definitions
3. Integrate Katib with pipeline components
4. Test multi-objective optimization scenarios

### Step 6: MLMD and Model Registry
1. Implement metadata tracking throughout pipeline
2. Set up model registry with versioning
3. Add automated model selection and promotion
4. Create model comparison and evaluation workflows

### Step 7: LakeFS Integration
1. Set up LakeFS repository for dataset management
2. Integrate data versioning with pipeline components
3. Implement branch-based experiment isolation
4. Add data lineage and reproducibility features

## Testing and Validation Plan

### Functional Testing
- [ ] Component-level testing with unit tests
- [ ] Pipeline integration testing
- [ ] End-to-end workflow validation
- [ ] Error handling and recovery testing

### Performance Testing
- [ ] Pipeline execution time measurement
- [ ] Resource utilization analysis
- [ ] Scalability testing with larger datasets
- [ ] Parallel execution efficiency validation

### Comparison Testing
- [ ] Feature parity validation with Airflow implementation
- [ ] Performance comparison (execution time, resource usage)
- [ ] Usability comparison (development experience, debugging)
- [ ] Reliability comparison (error handling, recovery)

## Success Criteria

### Phase 1 Success Criteria
- [ ] All four model architectures training successfully in pipeline
- [ ] Components can be executed independently and as part of pipeline
- [ ] Pipeline produces same results as original monolithic script
- [ ] Basic artifact management and metadata tracking working

### Phase 2 Success Criteria
- [ ] Parallel model training reduces total execution time by >50%
- [ ] Conditional execution and dynamic branching working correctly
- [ ] Pipeline caching reduces re-execution time by >80%
- [ ] Error handling and retry mechanisms prevent pipeline failures

### Phase 3 Success Criteria
- [ ] Katib successfully optimizes hyperparameters for all models
- [ ] Multi-objective optimization balances accuracy and training time
- [ ] Automated hyperparameter tuning improves model performance by >5%
- [ ] Integration with pipeline enables seamless HPO workflows

### Phase 4 Success Criteria
- [ ] Complete experiment lineage tracking from data to model
- [ ] Automated model selection chooses best performing model
- [ ] Model registry enables easy model comparison and versioning
- [ ] MLMD provides comprehensive pipeline execution insights

### Phase 5 Success Criteria
- [ ] LakeFS enables complete data versioning and branching
- [ ] Experiment reproducibility with exact data state reconstruction
- [ ] Data lineage tracking provides complete data provenance
- [ ] Integration enables data-centric ML workflows

## Resource Requirements

### Development Environment
- **Local Kubernetes Cluster**: minikube with 8GB RAM, 4 CPU cores
- **Kubeflow Installation**: Kubeflow 1.7+ with Pipelines, Katib, MLMD
- **Storage**: 50GB for datasets, models, and artifacts
- **Network**: Internet access for component image pulling

### Production Environment
- **Kubernetes Cluster**: 3+ nodes with 16GB RAM, 8 CPU cores each
- **GPU Support**: Optional NVIDIA GPU nodes for accelerated training
- **Storage**: Persistent volumes for artifacts and models
- **Monitoring**: Prometheus + Grafana for pipeline monitoring

### External Dependencies
- **Container Registry**: For storing custom component images
- **Artifact Storage**: S3-compatible storage for pipeline artifacts
- **LakeFS Server**: For data versioning and management
- **MLflow (for comparison)**: Maintain MLflow for baseline comparison

## Risk Mitigation

### Technical Risks
- **Kubeflow Complexity**: Start with simple pipelines, gradually add complexity
- **Component Dependencies**: Use standardized base images and dependency management
- **Performance Issues**: Implement comprehensive monitoring and profiling
- **Integration Challenges**: Prototype integrations early with minimal viable implementations

### Timeline Risks
- **Learning Curve**: Allocate extra time for Kubeflow ecosystem familiarization
- **Debugging Complexity**: Plan for significant debugging time in initial phases
- **Integration Complexity**: Prioritize core functionality over advanced features
- **Scope Creep**: Maintain focus on comparison objectives, avoid over-engineering

### Mitigation Strategies
- **Incremental Development**: Implement and test features incrementally
- **Documentation**: Maintain detailed documentation for troubleshooting
- **Community Support**: Leverage Kubeflow community and documentation
- **Fallback Plans**: Have simplified implementations ready if complex features fail

---

This implementation plan provides a comprehensive roadmap for building a production-ready Kubeflow-based MLOps stack that enables fair comparison with the existing MLflow + Airflow implementation while demonstrating the unique capabilities of the Kubeflow ecosystem.