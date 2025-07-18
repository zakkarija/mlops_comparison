#!/usr/bin/env python3
"""
Comprehensive MLMD Metadata Viewer for Kubeflow Pipeline
Accesses MLMD database directly via port-forward and extracts model metadata
"""

import json
import subprocess
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading
import signal

class MLMDMetadataViewer:
    def __init__(self):
        self.port_forward_process = None
        self.mysql_port = 3307  # Local port for port-forward
        self.target_run_id = "78f8a990-2701-4c42-81ef-c674e44fb28c"
        self.target_workflow = "lakefs-git-main-fresh-1752679843-2249-dfzgf"
        
    def setup_port_forward(self):
        """Setup port-forward to access MLMD MySQL database"""
        print("🔧 Setting up port-forward to MLMD database...")
        
        # Find the metadata-grpc service
        try:
            cmd = [
                "kubectl", "port-forward", 
                "-n", "kubeflow", 
                "service/metadata-grpc-service", 
                f"{self.mysql_port}:8080"
            ]
            
            self.port_forward_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait a bit for port-forward to establish
            time.sleep(3)
            
            # Check if port-forward is working
            if self.port_forward_process.poll() is None:
                print(f"✅ Port-forward established on localhost:{self.mysql_port}")
                return True
            else:
                print("❌ Port-forward failed to establish")
                return False
                
        except Exception as e:
            print(f"❌ Port-forward setup failed: {e}")
            return False
    
    def cleanup_port_forward(self):
        """Clean up port-forward process"""
        if self.port_forward_process:
            try:
                os.killpg(os.getpgid(self.port_forward_process.pid), signal.SIGTERM)
                self.port_forward_process.wait(timeout=5)
                print("🧹 Port-forward cleaned up")
            except:
                pass
    
    def query_mlmd_direct(self):
        """Query MLMD database directly"""
        print("🔍 Querying MLMD database directly...")
        
        try:
            from ml_metadata.proto import metadata_store_pb2
            from ml_metadata.metadata_store import metadata_store
            
            # Connect to MLMD via port-forward
            config = metadata_store_pb2.ConnectionConfig()
            config.fake_database.SetInParent()  # Use fake database for testing
            
            # Try to connect to actual MySQL if available
            try:
                config = metadata_store_pb2.ConnectionConfig()
                config.mysql.host = "localhost"
                config.mysql.port = self.mysql_port
                config.mysql.database = "metadb"
                config.mysql.user = "root"
                config.mysql.password = ""
                
                store = metadata_store.MetadataStore(config)
                print("✅ Connected to MLMD MySQL database")
                
                # Get all artifact types
                artifact_types = store.get_artifact_types()
                print(f"📦 Found {len(artifact_types)} artifact types:")
                
                for art_type in artifact_types:
                    print(f"   - {art_type.name} (ID: {art_type.id})")
                
                # Get all artifacts
                artifacts = store.get_artifacts()
                print(f"📁 Found {len(artifacts)} total artifacts")
                
                # Filter for model artifacts
                model_artifacts = []
                for artifact in artifacts:
                    if any(prop in artifact.name.lower() for prop in ['model', 'keras', 'tensorflow']):
                        model_artifacts.append(artifact)
                
                print(f"🤖 Found {len(model_artifacts)} model artifacts")
                
                # Extract metadata from model artifacts
                model_metadata = []
                for artifact in model_artifacts:
                    metadata = {
                        'id': artifact.id,
                        'name': artifact.name,
                        'uri': artifact.uri,
                        'type_id': artifact.type_id,
                        'create_time': artifact.create_time_since_epoch,
                        'properties': {}
                    }
                    
                    # Extract properties
                    for prop_name, prop_value in artifact.properties.items():
                        if prop_value.HasField('string_value'):
                            metadata['properties'][prop_name] = prop_value.string_value
                        elif prop_value.HasField('double_value'):
                            metadata['properties'][prop_name] = prop_value.double_value
                        elif prop_value.HasField('int_value'):
                            metadata['properties'][prop_name] = prop_value.int_value
                    
                    model_metadata.append(metadata)
                
                return {
                    'method': 'mlmd_direct',
                    'success': True,
                    'total_artifacts': len(artifacts),
                    'model_artifacts': len(model_artifacts),
                    'model_metadata': model_metadata,
                    'artifact_types': [{'name': at.name, 'id': at.id} for at in artifact_types]
                }
                
            except Exception as e:
                print(f"❌ MySQL connection failed: {e}")
                return {'method': 'mlmd_direct', 'error': str(e)}
                
        except ImportError:
            print("❌ ml-metadata package not available")
            return {'method': 'mlmd_direct', 'error': 'ml-metadata not installed'}
    
    def query_workflow_artifacts(self):
        """Query workflow artifacts via kubectl"""
        print("🔍 Querying workflow artifacts via kubectl...")
        
        try:
            # Get the specific workflow
            cmd = [
                "kubectl", "get", "workflow", 
                self.target_workflow, 
                "-n", "team-1", 
                "-o", "json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                workflow_data = json.loads(result.stdout)
                
                # Extract artifacts from workflow status
                artifacts = []
                nodes = workflow_data.get('status', {}).get('nodes', {})
                
                for node_id, node_data in nodes.items():
                    if 'outputs' in node_data and 'artifacts' in node_data['outputs']:
                        for artifact in node_data['outputs']['artifacts']:
                            artifact_info = {
                                'name': artifact.get('name', 'unknown'),
                                'path': artifact.get('path', ''),
                                'node': node_data.get('displayName', node_id),
                                's3': artifact.get('s3', {}),
                                'archive': artifact.get('archive', {}),
                                'raw': artifact  # Keep raw data for analysis
                            }
                            artifacts.append(artifact_info)
                
                return {
                    'method': 'workflow_artifacts',
                    'success': True,
                    'workflow_name': self.target_workflow,
                    'artifacts': artifacts,
                    'workflow_status': workflow_data.get('status', {}).get('phase', 'unknown'),
                    'workflow_data': workflow_data
                }
            else:
                return {
                    'method': 'workflow_artifacts',
                    'error': f"kubectl failed: {result.stderr}"
                }
                
        except Exception as e:
            return {
                'method': 'workflow_artifacts',
                'error': str(e)
            }
    
    def extract_model_metadata_from_artifacts(self, artifacts):
        """Extract model metadata from S3 artifacts"""
        print("📊 Extracting model metadata from artifacts...")
        
        model_metadata = []
        
        for artifact in artifacts:
            if 'model' in artifact['name'].lower() or 'trained' in artifact['name'].lower():
                print(f"🤖 Found model artifact: {artifact['name']}")
                
                # Try to read metadata from S3 path
                s3_info = artifact.get('s3', {})
                if s3_info:
                    try:
                        # Extract S3 path info
                        bucket = s3_info.get('bucket', '')
                        key = s3_info.get('key', '')
                        
                        if bucket and key:
                            # Try to read metadata file via MinIO
                            metadata_key = key.replace('model.keras', 'model_metadata.json')
                            metadata_path = f"s3://{bucket}/{metadata_key}"
                            
                            print(f"   📝 Looking for metadata at: {metadata_path}")
                            
                            # Use kubectl to access MinIO
                            metadata_content = self.read_s3_file(bucket, metadata_key)
                            if metadata_content:
                                try:
                                    metadata_json = json.loads(metadata_content)
                                    model_metadata.append({
                                        'artifact_name': artifact['name'],
                                        'node': artifact['node'],
                                        's3_path': f"s3://{bucket}/{key}",
                                        'metadata': metadata_json
                                    })
                                    print(f"   ✅ Successfully extracted metadata")
                                except:
                                    print(f"   ❌ Failed to parse metadata JSON")
                            else:
                                print(f"   ⚠️ No metadata file found")
                    except Exception as e:
                        print(f"   ❌ Error extracting metadata: {e}")
        
        return model_metadata
    
    def read_s3_file(self, bucket, key):
        """Read file from S3 via MinIO kubectl exec"""
        try:
            # Find MinIO pod
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", "kubeflow", 
                "-l", "app=minio", "-o", "jsonpath={.items[0].metadata.name}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                minio_pod = result.stdout.strip()
                
                # Read file from MinIO
                file_path = f"/data/{bucket}/{key}"
                result = subprocess.run([
                    "kubectl", "exec", "-n", "kubeflow", 
                    minio_pod, "--", "cat", file_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    return result.stdout
                else:
                    print(f"   ❌ Failed to read {file_path}: {result.stderr}")
                    return None
            else:
                print(f"   ❌ MinIO pod not found: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"   ❌ S3 read error: {e}")
            return None
    
    def export_to_local_file(self, all_results):
        """Export all results to local file"""
        print("💾 Exporting results to local file...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mlmd_metadata_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"✅ Results exported to: {filename}")
        
        # Also create a summary file
        summary_filename = f"mlmd_summary_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            f.write("MLMD Metadata Query Results\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Target Run ID: {self.target_run_id}\n")
            f.write(f"Target Workflow: {self.target_workflow}\n\n")
            
            # Workflow artifacts summary
            workflow_result = all_results.get('workflow_artifacts', {})
            if workflow_result.get('success'):
                artifacts = workflow_result.get('artifacts', [])
                f.write(f"Workflow Artifacts: {len(artifacts)}\n")
                for artifact in artifacts:
                    f.write(f"  - {artifact['name']} ({artifact['node']})\n")
                f.write("\n")
            
            # Model metadata summary
            model_metadata = all_results.get('model_metadata', [])
            if model_metadata:
                f.write(f"Model Metadata Found: {len(model_metadata)}\n")
                for model in model_metadata:
                    f.write(f"\n--- Model: {model['artifact_name']} ---\n")
                    metadata = model.get('metadata', {})
                    if 'train_accuracy' in metadata:
                        f.write(f"Training Accuracy: {metadata['train_accuracy']:.4f}\n")
                    if 'test_accuracy' in metadata:
                        f.write(f"Test Accuracy: {metadata['test_accuracy']:.4f}\n")
                    if 'n_classes' in metadata:
                        f.write(f"Number of Classes: {metadata['n_classes']}\n")
                    if 'class_names' in metadata:
                        f.write(f"Classes: {metadata['class_names']}\n")
                    if 'framework' in metadata:
                        f.write(f"Framework: {metadata['framework']}\n")
                    if 'mlmd_id' in metadata:
                        f.write(f"MLMD ID: {metadata['mlmd_id']}\n")
        
        print(f"✅ Summary exported to: {summary_filename}")
        return filename, summary_filename
    
    def run(self):
        """Main execution function"""
        print("🚀 Starting MLMD Metadata Viewer")
        print("=" * 60)
        
        all_results = {}
        
        try:
            # Method 1: Query workflow artifacts
            print("\n1️⃣ Querying workflow artifacts...")
            workflow_result = self.query_workflow_artifacts()
            all_results['workflow_artifacts'] = workflow_result
            
            # Method 2: Extract model metadata from artifacts
            if workflow_result.get('success'):
                print("\n2️⃣ Extracting model metadata...")
                artifacts = workflow_result.get('artifacts', [])
                model_metadata = self.extract_model_metadata_from_artifacts(artifacts)
                all_results['model_metadata'] = model_metadata
                
                # Print immediate results
                if model_metadata:
                    print(f"\n🎯 FOUND {len(model_metadata)} MODEL(S) WITH METADATA:")
                    for model in model_metadata:
                        print(f"\n📊 Model: {model['artifact_name']}")
                        metadata = model.get('metadata', {})
                        if 'train_accuracy' in metadata:
                            print(f"   Training Accuracy: {metadata['train_accuracy']:.4f}")
                        if 'test_accuracy' in metadata:
                            print(f"   Test Accuracy: {metadata['test_accuracy']:.4f}")
                        if 'n_classes' in metadata:
                            print(f"   Classes: {metadata['n_classes']}")
                        if 'framework' in metadata:
                            print(f"   Framework: {metadata['framework']}")
                        if 'mlmd_id' in metadata:
                            print(f"   MLMD ID: {metadata['mlmd_id']}")
                else:
                    print("⚠️ No model metadata found in artifacts")
            
            # Method 3: Try direct MLMD query (if port-forward works)
            print("\n3️⃣ Attempting direct MLMD query...")
            if self.setup_port_forward():
                mlmd_result = self.query_mlmd_direct()
                all_results['mlmd_direct'] = mlmd_result
                self.cleanup_port_forward()
            else:
                all_results['mlmd_direct'] = {'error': 'Port-forward failed'}
            
            # Export results
            print("\n4️⃣ Exporting results...")
            json_file, summary_file = self.export_to_local_file(all_results)
            
            print(f"\n🎯 FINAL RESULTS:")
            print(f"   JSON Data: {json_file}")
            print(f"   Summary: {summary_file}")
            
            return all_results
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            self.cleanup_port_forward()
            return all_results
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.cleanup_port_forward()
            return all_results

def main():
    """Main function"""
    viewer = MLMDMetadataViewer()
    
    # Handle cleanup on exit
    def cleanup_handler(signum, frame):
        print("\n🧹 Cleaning up...")
        viewer.cleanup_port_forward()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    try:
        results = viewer.run()
        return results
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        viewer.cleanup_port_forward()
        return {}

if __name__ == "__main__":
    main()