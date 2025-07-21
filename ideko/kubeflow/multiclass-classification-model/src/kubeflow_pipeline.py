# ideko_pipeline.py  – Option B, fixed data placement for main_pipe.py
import kfp
from kfp import dsl
from kfp.dsl import component
from kfp.compiler import Compiler

# ------------------------------------------------------------------#
# Component 1: LakeFS → data_out (Dataset)
# ------------------------------------------------------------------#
@component(
    base_image="python:3.11-slim",
    packages_to_install=["lakefs==0.11.1"]
)
def lakefs_pull(data_out: dsl.Output[dsl.Dataset]) -> None:
    """Download first 10 objects from LakeFS into data_out.path"""
    import shutil
    from pathlib import Path
    import lakefs
    from lakefs.client import Client

    client = Client(
        host="http://lakefs.lakefs.svc.cluster.local:8000",
        username="user",
        password="pass"
    )
    repo   = lakefs.Repository("ideko", client=client)
    branch = repo.branch("main")

    tmp = Path("/tmp/lakefs_data")
    tmp.mkdir(parents=True, exist_ok=True)

    for i, obj_info in enumerate(branch.objects()):
        if i == 10:
            break
        with branch.object(obj_info.path).reader("rb") as reader:
            out_file = tmp / Path(obj_info.path).name
            out_file.write_bytes(reader.read())
        print("Downloaded", obj_info.path)

    shutil.copytree(tmp, data_out.path, dirs_exist_ok=True)


# ------------------------------------------------------------------#
# Component 2: Git clone → code_out (Dataset)
# ------------------------------------------------------------------#
@component(
    base_image="python:3.11",
    packages_to_install=["gitpython"]
)
def git_clone(code_out: dsl.Output[dsl.Dataset]) -> None:
    """Clone the repo into code_out.path"""
    import subprocess, shutil
    from pathlib import Path

    repo_dir = Path("/tmp/git_repo")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=True)

    subprocess.run([
        "git", "clone", "--branch", "kubeflow", "--depth", "1",
        "https://github.com/zakkarija/mlops_comparison.git",
        str(repo_dir)
    ], check=True)

    shutil.copytree(repo_dir, code_out.path, dirs_exist_ok=True)


# ------------------------------------------------------------------#
# Component 3: Run main_pipe.py (data one level up)
# ------------------------------------------------------------------#
@component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas", "numpy", "scikit-learn",
        "tensorflow==2.16.1", "ml-metadata==1.17.0",
        "matplotlib", "seaborn", "pyyaml"
    ]
)
def run_main_pipe(
    data_in: dsl.Input[dsl.Dataset],
    code_in: dsl.Input[dsl.Dataset]
) -> None:
    """Place data one level up, then execute main_pipe.py"""
    import os, sys, shutil, subprocess
    from pathlib import Path

    # Paths
    data_src = Path(data_in.path)
    code_root = Path(code_in.path)

    # Locate the script
    candidates = list(code_root.rglob("main_pipe.py"))
    if not candidates:
        raise FileNotFoundError("main_pipe.py not found in cloned repo")
    main_py = candidates[0]

    # We know main_pipe.py lives in .../src/kubeflow_pipeline
    pipeline_dir = main_py.parent
    project_src = pipeline_dir.parent      # this is .../src

    # Copy data into project_src/data (so main_pipe's default "../data" works)
    target_data = project_src / "data"
    if target_data.exists():
        shutil.rmtree(target_data)
    shutil.copytree(data_src, target_data)

    # Run from pipeline_dir
    os.chdir(pipeline_dir)
    sys.path.insert(0, str(pipeline_dir))
    sys.path.insert(0, str(code_root))

    # Execute the pipeline‑friendly script
    subprocess.run(
        [
            sys.executable,
            str(main_py),
            "--model_output_path",
            str(project_src / "mlmd_artifact")
        ],
        check=True
    )


# ------------------------------------------------------------------#
# Pipeline definition
# ------------------------------------------------------------------#
@dsl.pipeline(
    name="lakefs-git-run-mainpipe",
    description="LakeFS pull → Git clone → run main_pipe.py"
)
def lakefs_git_pipeline():
    data = lakefs_pull()
    code = git_clone()
    run_main_pipe(
        data_in = data.outputs["data_out"],
        code_in = code.outputs["code_out"]
    )


if __name__ == "__main__":
    Compiler().compile(
        pipeline_func=lakefs_git_pipeline,
        package_path="lakefs_git_run_main.yaml"
    )

    client = kfp.Client()

    run = client.create_run_from_pipeline_package(
        pipeline_file="lakefs_git_run_main.yaml",
        experiment_name="admin-experiment",
        run_name="lakefs-git-run-main-run",
        service_account="default-editor",
        arguments={}
    )
    print(f"Run submitted, ID: {run.run_id}")
