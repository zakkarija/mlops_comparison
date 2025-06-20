"""
Ideko Manufacturing Multiclass‑Classification ML Pipeline
========================================================

A streamlined Airflow DAG that orchestrates the full ML workflow:

1. Pull data from the DVC remote.
2. Validate (or scaffold) the data directory layout.
3. Convert raw data into Feast‑ready parquet + apply the feature store.
4. Train an LSTM‑based multiclass model.
5. Package + version the resulting artifacts with DVC, then push remote.

Notes
-----
* No more inline `pyenv activate …` calls – the worker already runs
  inside your `ideko-airflow` environment.
* BashOperator is still used when we need shell utilities (DVC, Feast CLI).
  Pure‑Python work (metric collection) stays in PythonOperator.
* Feel free to convert more Bash steps to PythonOperator later.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/home/zak/PycharmProjects/mlops_comparison/ideko/MLFlow"
MODEL_ROOT = os.path.join(
    PROJECT_ROOT,
    "failure-prediction-in-manufacture-main",
    "multiclass-classification-model",
)
SRC_PATH = os.path.join(MODEL_ROOT, "src")
FEAST_REPO_PATH = os.path.join(SRC_PATH, "feast_demo", "feature_repo")

DAG_ID = "ideko_multiclass_ml_pipeline"
DESCRIPTION = "ML pipeline for Ideko manufacturing multiclass anomaly detection"

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
        start_date=datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0),
        catchup=False,
        default_args=DEFAULT_ARGS,
        max_active_runs=1,
        tags=[
            "ml",
            "manufacturing",
            "anomaly-detection",
            "multiclass",
            "feast",
            "dvc",
        ],
) as dag:
    # ---------------------------------------------------------------------
    # 1. Pull the latest data from DVC remote.
    # ---------------------------------------------------------------------
    dvc_pull_data = BashOperator(
        task_id="dvc_pull_data",
        cwd=MODEL_ROOT,
        bash_command="""
            echo '🔽 Pulling data from DVC remote …' && \
            dvc pull --allow-missing && \
            echo '✅ DVC pull done.'
        """,
    )

    # ---------------------------------------------------------------------
    # 2. Validate directory layout (or scaffold it for local testing).
    # ---------------------------------------------------------------------
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
                mkdir -p "data/$sub"  # ensure exists
                echo "✅ $sub -> $(ls -1q data/$sub | wc -l) items";
            done && \
            echo '✅ Data layout OK.'
        """,
    )

    # ---------------------------------------------------------------------
    # 3. Convert raw → Feast parquet, then 4. apply feature store.
    # ---------------------------------------------------------------------
    convert_to_feast = BashOperator(
        task_id="convert_to_feast_format",
        cwd=FEAST_REPO_PATH,
        bash_command="""
            echo '🔄 Converting data for Feast …' && \
            python convert_data.py && \
            echo '✅ Conversion finished.'
        """,
    )

    feast_apply = BashOperator(
        task_id="feast_apply",
        cwd=FEAST_REPO_PATH,
        bash_command="""
            echo '🍽️  feast apply …' && \
            feast apply && \
            feast feature-views list
        """,
    )

    validate_feast = BashOperator(
        task_id="validate_feast_store",
        cwd=FEAST_REPO_PATH,
        bash_command="""
            echo '🔍 Validating Feast store …' && \
            python test_feast.py
        """,
    )

    # ---------------------------------------------------------------------
    # 5. Train the model.
    # ---------------------------------------------------------------------
    train_model = BashOperator(
        task_id="train_multiclass_model",
        cwd=SRC_PATH,
        bash_command="""
            echo '🤖 Training model …' && \
            python main_poc.py && \
            echo '✅ Training done.'
        """,
    )


    # ---------------------------------------------------------------------
    # 6. Collect metrics (PythonOperator).
    # ---------------------------------------------------------------------
    def collect_metrics() -> None:
        out_dir = os.path.join(SRC_PATH, "output")
        if not os.path.isdir(out_dir):
            print("❌ output/ not found – training may have failed.")
            return

        files = os.listdir(out_dir)
        print("📊  artifacts:", files)
        model_files = [f for f in files if f.endswith(".keras")]
        if model_files:
            print("✅ model:", model_files[0])
        else:
            print("⚠️  no .keras model saved.")


    collect_metrics_task = PythonOperator(
        task_id="collect_model_metrics",
        python_callable=collect_metrics,
    )

    # ---------------------------------------------------------------------
    # 7‑8‑9. Package → dvc add → push.
    # ---------------------------------------------------------------------
    package_artifacts = BashOperator(
        task_id="package_artifacts",
        cwd=MODEL_ROOT,
        bash_command="""
            mkdir -p artifacts/{models,features,plots} && \
            cp -f src/output/*.keras artifacts/models/ 2>/dev/null || true && \
            cp -f src/output/*.png   artifacts/plots/  2>/dev/null || true && \
            cp -f src/feast_demo/feature_repo/data/offline/*.parquet artifacts/features/ 2>/dev/null || true && \
            echo '📦  artifacts packaged.'
        """,
    )

    dvc_add_artifacts = BashOperator(
        task_id="dvc_add_artifacts",
        cwd=MODEL_ROOT,
        bash_command="""
            dvc add artifacts && \
            echo '📝 dvc add done.'
        """,
    )

    dvc_push = BashOperator(
        task_id="dvc_push",
        cwd=MODEL_ROOT,
        bash_command="""
            dvc push && \
            echo '⬆️  dvc push complete.'
        """,
    )

    done = EmptyOperator(task_id="pipeline_success")

    # ------------------------------------------------------------------
    # Dependency graph
    # ------------------------------------------------------------------
    (
            dvc_pull_data
            >> validate_data_structure
            >> convert_to_feast
            >> feast_apply
            >> validate_feast
            >> train_model
            >> collect_metrics_task
            >> package_artifacts
            >> dvc_add_artifacts
            >> dvc_push
            >> done
    )
