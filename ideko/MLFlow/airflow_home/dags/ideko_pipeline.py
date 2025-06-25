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

DAG_ID      = "ideko_multiclass_ml_pipeline"
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
    start_date=datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["ml","manufacturing","anomaly-detection","multiclass","feast","dvc"],
) as dag:

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

    # 5 ────────────────────────────────────── model training
    train_model = BashOperator(
        task_id="train_multiclass_model",
        cwd=SRC_PATH,
        bash_command="""
            echo '🤖 Training model …' && \
            python main_poc.py && \
            echo '✅ Training done.'
        """,
    )

    # 6 ───────────────────────────────────── metric collection (Python)
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

    # 7-8-9 ────────────────────────── package → dvc add → dvc push
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

    # ───────────────────────────────── dependency graph
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
