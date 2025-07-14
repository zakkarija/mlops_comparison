"""
Simple Kubeflow Pipeline for Ideko ML Training
3 components: git clone -> dvc pull -> python main.py
"""

import kfp
from kfp.dsl import component, pipeline, Input, Output, Artifact, Model


@component(
    base_image="python:3.11-slim",
    packages_to_install=["GitPython==3.1.40"]
)
def git_clone_component(
    repo_url: str,
    branch: str,
    code_output: Output[Artifact]
) -> None:
    """Clone git repository"""
    import git
    import os
    
    print(f"📥 Cloning {repo_url} (branch: {branch})")
    
    try:
        # Clone to output path
        git.Repo.clone_from(
            repo_url, 
            code_output.path,
            branch=branch,
            depth=1  # Shallow clone for speed
        )
        print("✅ Git clone completed successfully")
        
    except Exception as e:
        print(f"❌ Git clone failed: {e}")
        raise


@component(
    base_image="python:3.11-slim",
    packages_to_install=["dvc[s3]==3.48.4"]
)
def dvc_pull_component(
    code_input: Input[Artifact],
    project_output: Output[Artifact]
) -> None:
    """Pull data using DVC - data goes directly to project/data/ folder"""
    import subprocess
    import shutil
    import os
    
    print("📥 Running DVC pull...")
    
    try:
        # Copy entire project to output (so we have code + data together)
        shutil.copytree(code_input.path, project_output.path)
        
        # Change to the project directory (cloned repo root is mlops_comparison)
        project_path = os.path.join(project_output.path, "ideko/kubeflow/multiclass-classification-model")
        
        # Debug: Check what actually got cloned
        print("📂 Checking cloned repository structure:")
        for item in os.listdir(project_output.path):
            print(f"  - {item}")
        
        if not os.path.exists(project_path):
            print(f"⚠️ Expected path not found: {project_path}")
            print("Looking for DVC files in cloned repo...")
            
            # Search for .dvc files in the cloned repo
            for root, dirs, files in os.walk(project_output.path):
                for file in files:
                    if file.endswith('.dvc'):
                        found_dvc_dir = root
                        print(f"✅ Found DVC file at: {found_dvc_dir}")
                        project_path = found_dvc_dir
                        break
                if 'project_path' in locals():
                    break
        
        print(f"📍 Using project path: {project_path}")
        os.chdir(project_path)
        
        # Run DVC pull - this will automatically pull to data/ folder
        result = subprocess.run(
            ["dvc", "pull"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        print("✅ DVC pull completed successfully")
        print(result.stdout)
        
        # Verify data exists
        data_dir = os.path.join(project_path, "data")
        if os.path.exists(data_dir):
            print(f"✅ Data available at: {data_dir}")
            # List contents for verification
            for item in os.listdir(data_dir)[:5]:  # Show first 5 items
                print(f"  - {item}")
        else:
            print("⚠️ No data directory found after DVC pull")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ DVC pull failed: {e}")
        print(f"Stderr: {e.stderr}")
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "tensorflow==2.16.1",
        "scikit-learn==1.4.2", 
        "matplotlib==3.8.4",
        "pyyaml==6.0.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "keras==3.2.1"
    ]
)
def train_model_component(
    project_input: Input[Artifact],
    model_type: str,
    trained_model: Output[Model]
) -> None:
    """Simple training: just run python main.py"""
    import subprocess
    import os
    
    print(f"🚀 Running python main.py for model: {model_type}")
    
    # Change to the src directory where main.py is located
    src_path = os.path.join(project_input.path, "ideko/kubeflow/multiclass-classification-model/src")
    os.chdir(src_path)
    
    # Run main.py directly
    subprocess.run(["python", "main.py"], check=True)


@pipeline(
    name="ideko-ml-training",
    description="Simple ML training pipeline: git clone -> dvc pull -> train"
)
def ideko_training_pipeline(
    repo_url: str = "https://github.com/zakmic/mlops_comparison.git",
    branch: str = "kubeflow", 
    model_type: str = "NeuralNetwork"
):
    """
    Simple 3-step ML training pipeline
    
    Args:
        repo_url: Git repository URL
        branch: Git branch to clone  
        model_type: Model to train (NeuralNetwork, CNN, RNN, LSTM)
    """
    
    # Step 1: Clone repository
    clone_task = git_clone_component(
        repo_url=repo_url,
        branch=branch
    )
    
    # Step 2: Pull data with DVC (combines code + data)
    dvc_task = dvc_pull_component(
        code_input=clone_task.outputs["code_output"]
    )
    
    # Step 3: Train model (data is already in project structure)
    train_model_component(
        project_input=dvc_task.outputs["project_output"],
        model_type=model_type
    )


if __name__ == "__main__":
    # Compile the pipeline
    from kfp.compiler import Compiler
    
    compiler = Compiler()
    compiler.compile(
        pipeline_func=ideko_training_pipeline,
        package_path="ideko_training_pipeline.yaml"
    )
    
    print("✅ Pipeline compiled: ideko_training_pipeline.yaml")
    print("")
    print("🚀 Usage:")
    print("1. Update repo_url in the pipeline parameters")
    print("2. Upload ideko_training_pipeline.yaml to Kubeflow")
    print("3. Run with different model_type parameters:")
    print("   - NeuralNetwork")
    print("   - CNN") 
    print("   - RNN")
    print("   - LSTM")