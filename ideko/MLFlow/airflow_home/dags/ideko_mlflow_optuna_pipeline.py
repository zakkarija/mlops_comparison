from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT   = "/home/zak/PycharmProjects/mlops_comparison/ideko/MLFlow"
MODEL_ROOT     = os.path.join(
    PROJECT_ROOT,
    "failure-prediction-in-manufacture-main",
    "multiclass-classification-model",
)
SRC_PATH        = os.path.join(MODEL_ROOT, "src")
FEAST_REPO_PATH = os.path.join(SRC_PATH, "feast_demo", "feature_repo")

DAG_ID      = "ideko_mlflow_optuna_pipeline"
DESCRIPTION = "ML pipeline with MLflow tracking and Optuna hyperparameter optimization"

DEFAULT_ARGS = {
    "owner": "ideko-data-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    description=DESCRIPTION,
    schedule="@daily",
    start_date=datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["ml","manufacturing","anomaly-detection","multiclass","feast","dvc","mlflow","optuna"],
) as dag:

    # 0 ─────────────────────────────────────────────── MLflow server check
    check_mlflow_server = BashOperator(
        task_id="check_mlflow_server",
        bash_command="""
            echo '🔍 Checking MLflow server status...' && \
            if curl -f http://127.0.0.1:8081/health 2>/dev/null; then
                echo '✅ MLflow server is running'
            else
                echo '⚠️  MLflow server not running - please start with: mlflow server --host 127.0.0.1 --port 8081'
                exit 1
            fi
        """,
    )

    # 1 ─────────────────────────────────────────────── DVC pull
    dvc_pull_data = BashOperator(
        task_id="dvc_pull_data",
        cwd=MODEL_ROOT,
        bash_command="""
            echo '🔽 Pulling data from DVC remote …' && \
            dvc pull --allow-missing --force && \
            echo '✅ DVC pull done.'
        """,
    )

    # 2 ───────────────────────────────── data-directory validation
    validate_data_structure = BashOperator(
        task_id="validate_data_structure",
        cwd=MODEL_ROOT,
        bash_command="""
            echo '🔍 Validating data directory …' && \
            if [ ! -d data ]; then
                echo '⚠️  data/ missing – scaffolding minimal layout.' && \
                mkdir -p data/{electrical_anomalies,mechanical_anomalies,not_anomalous}
            fi && \
            for sub in electrical_anomalies mechanical_anomalies not_anomalous; do
                mkdir -p "data/$sub" && \
                echo "✅ $sub -> $(ls -1q data/$sub | wc -l) items";
            done && \
            echo '✅ Data layout OK.'
        """,
    )

    # 3 ───────────────────────────────── raw → Feast parquet conversion
    convert_to_feast = BashOperator(
        task_id="convert_to_feast_format",
        cwd=FEAST_REPO_PATH,
        bash_command="""
            echo '🔄 Converting data for Feast …' && \
            python convert_data.py && \
            echo '✅ Conversion finished.'
        """,
    )

    # 4a ───────────────────────────────── feast apply (CLI)
    feast_apply = BashOperator(
        task_id="feast_apply",
        cwd=FEAST_REPO_PATH,
        bash_command="""
            echo '🍽️  Applying Feast feature store…' && \
            feast apply && \
            echo '✅ Feast apply complete.'
        """,
    )

    # 4b ─────────────────────────────── feast validation (CLI)
    validate_feast = BashOperator(
        task_id="validate_feast_store",
        cwd=FEAST_REPO_PATH,
        bash_command="""
            echo '🔍 Listing feature-views…' && \
            feast feature-views list && \
            echo '✅ Feast validation OK.'
        """,
    )

    # 5 ────────────────────────────────────── MLflow experiment setup
    setup_mlflow_experiment = BashOperator(
        task_id="setup_mlflow_experiment",
        cwd=SRC_PATH,
        bash_command="""
            echo '🧪 Setting up MLflow experiment...' && \
            python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:8081')
experiment_name = 'ideko_manufacturing_anomaly_detection'
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f'Created experiment: {experiment_name} (ID: {experiment_id})')
    else:
        print(f'Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})')
        mlflow.set_experiment(experiment_name)
except Exception as e:
    print(f'Error setting up experiment: {e}')
    exit(1)
" && \
            echo '✅ MLflow experiment setup complete.'
        """,
    )

    # 6 ────────────────────────────────────── model training with MLflow + Optuna
    train_model_with_optuna = BashOperator(
        task_id="train_model_with_mlflow_optuna",
        cwd=SRC_PATH,
        bash_command="""
            echo '🤖 Training models with MLflow tracking and Optuna optimization…' && \
            python training_optuna.py && \
            echo '✅ Training with MLflow + Optuna complete.'
        """,
    )

    # 7 ───────────────────────────────────── MLflow metrics collection
    def collect_mlflow_metrics() -> None:
        """Collect and display MLflow experiment metrics"""
        import mlflow
        
        mlflow.set_tracking_uri("http://127.0.0.1:8081")
        
        try:
            # Get recent runs
            runs = mlflow.search_runs(
                experiment_names=["ideko_manufacturing_anomaly_detection"],
                max_results=10,
                order_by=["start_time DESC"]
            )
            
            if len(runs) > 0:
                print("📊 Recent MLflow runs:")
                for idx, run in runs.iterrows():
                    status = run.get('status', 'UNKNOWN')
                    run_id = run.get('run_id', 'N/A')[:8]
                    print(f"  - Run {run_id}: {status}")
                
                print(f"✅ Found {len(runs)} total runs in experiment")
            else:
                print("⚠️  No runs found in MLflow experiment")
                
        except Exception as e:
            print(f"❌ Error accessing MLflow: {e}")

    collect_mlflow_metrics_task = PythonOperator(
        task_id="collect_mlflow_metrics",
        python_callable=collect_mlflow_metrics,
    )

    # 8 ───────────────────────────────────── local artifact collection
    def collect_local_artifacts() -> None:
        """Collect local training artifacts"""
        out_dir = os.path.join(SRC_PATH, "output")
        if not os.path.isdir(out_dir):
            print("❌ output/ not found – training may have failed.")
            return
        
        files = os.listdir(out_dir)
        print("📦 Local artifacts:", files)
        
        # Count model files
        model_files = []
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                if file.endswith(".keras"):
                    model_files.append(os.path.join(root, file))
        
        if model_files:
            print(f"✅ Found {len(model_files)} model files:")
            for model_file in model_files:
                print(f"  - {model_file}")
        else:
            print("⚠️  No .keras model files found.")

    collect_local_artifacts_task = PythonOperator(
        task_id="collect_local_artifacts",
        python_callable=collect_local_artifacts,
    )

    # 9-10-11 ────────────────────────── package → dvc add → dvc push
    package_artifacts = BashOperator(
        task_id="package_artifacts",
        cwd=MODEL_ROOT,
        bash_command="""
            echo '📦 Packaging artifacts...' && \
            mkdir -p artifacts/{models,features,plots} && \
            find src/output -name "*.keras" -exec cp {} artifacts/models/ \\; 2>/dev/null || true && \
            find src/output -name "*.png" -exec cp {} artifacts/plots/ \\; 2>/dev/null || true && \
            cp -f src/feast_demo/feature_repo/data/offline/*.parquet artifacts/features/ 2>/dev/null || true && \
            echo "📊 Packaged artifacts:" && \
            find artifacts -type f | wc -l && \
            echo '✅ Artifacts packaged.'
        """,
    )

    dvc_add_artifacts = BashOperator(
        task_id="dvc_add_artifacts",
        cwd=MODEL_ROOT,
        bash_command="""
            echo '📝 Adding artifacts to DVC...' && \
            dvc add artifacts && \
            echo '✅ DVC add done.'
        """,
    )

    dvc_push = BashOperator(
        task_id="dvc_push",
        cwd=MODEL_ROOT,
        bash_command="""
            echo '⬆️  Pushing to DVC remote...' && \
            dvc push && \
            echo '✅ DVC push complete.'
        """,
    )

    # 12 ────────────────────────────────────── success summary
    def pipeline_summary() -> None:
        """Print pipeline completion summary"""
        print("🎉 MLflow + Optuna Pipeline Complete!")
        print("📊 Check MLflow UI at: http://127.0.0.1:8081")
        print("📈 View experiment: ideko_manufacturing_anomaly_detection")
        print("✅ All models trained with hyperparameter optimization")

    pipeline_summary_task = PythonOperator(
        task_id="pipeline_summary",
        python_callable=pipeline_summary,
    )

    done = EmptyOperator(task_id="pipeline_success")

    # ───────────────────────────────── dependency graph
    (
        check_mlflow_server
        >> dvc_pull_data
        >> validate_data_structure
        >> convert_to_feast
        >> feast_apply
        >> validate_feast
        >> setup_mlflow_experiment
        >> train_model_with_optuna
        >> [collect_mlflow_metrics_task, collect_local_artifacts_task]
        >> package_artifacts
        >> dvc_add_artifacts
        >> dvc_push
        >> pipeline_summary_task
        >> done
    )