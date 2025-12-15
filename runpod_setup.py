#!/usr/bin/env python3
"""
RunPod Setup and Training Script
This script helps you deploy and run training on RunPod GPU instances
"""

import runpod
import time
import os

# Your RunPod API Key
RUNPOD_API_KEY = "rpa_29GS7ZOVYZRB5D2XFTZNBU9JQTOGNN3AEZC8KJR14ahqwp"

def setup_runpod():
    """Initialize RunPod with your API key"""
    runpod.api_key = RUNPOD_API_KEY
    print("✓ RunPod API key configured")

def list_gpu_types():
    """Show available GPU types and pricing"""
    print("\n=== Available GPU Types ===")
    try:
        gpu_types = runpod.get_gpus()
        if not gpu_types:
            print("No GPUs available at the moment")
            return []
        
        for idx, gpu in enumerate(gpu_types, 1):
            print(f"{idx}. {gpu['displayName']} (ID: {gpu['id']})")
            print(f"   Memory: {gpu.get('memoryInGb', 'N/A')} GB")
            print(f"   Price: ${gpu.get('lowestPrice', {}).get('minimumBidPrice', 'N/A')}/hour")
            print()
        return gpu_types
    except Exception as e:
        print(f"Error fetching GPUs: {e}")
        return []

def create_pod(gpu_type_id, disk_size=50):
    """
    Create a RunPod GPU instance
    
    Args:
        gpu_type_id: GPU type ID from RunPod (e.g., 'NVIDIA RTX 4090')
        disk_size: Storage size in GB
    """
    print(f"\n=== Creating Pod with GPU ID: {gpu_type_id} ===")
    
    try:
        pod = runpod.create_pod(
            name="heartbeat-training",
            image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            gpu_type_id=gpu_type_id,
            cloud_type="SECURE",  # or "COMMUNITY" for cheaper
            volume_in_gb=disk_size,
            ports="8888/http,6006/http",  # Jupyter and TensorBoard
        )
        
        print(f"✓ Pod created: {pod['id']}")
        print(f"  Status: {pod.get('desiredStatus', 'unknown')}")
        print(f"  Waiting for pod to be ready...")
        
        # Wait for pod to be ready
        for i in range(60):  # Wait up to 5 minutes
            time.sleep(5)
            try:
                pod_status = runpod.get_pod(pod['id'])
                
                if pod_status is None:
                    print(f"  Waiting for pod to initialize... ({i*5}s)")
                    continue
                
                runtime = pod_status.get('runtime', {})
                if runtime and runtime.get('uptimeInSeconds', 0) > 0:
                    print(f"✓ Pod is running!")
                    return pod_status
                
                print(f"  Still starting... ({i*5}s)")
            except Exception as e:
                print(f"  Checking status... ({i*5}s)")
                continue
        
        print("⚠ Pod creation timeout, but it may still be starting.")
        print("  Check RunPod console at: https://www.runpod.io/console/pods")
        print(f"  Pod ID: {pod['id']}")
        return pod
        
    except Exception as e:
        print(f"Error creating pod: {e}")
        raise

def get_pod_connection_info(pod_id):
    """Get SSH and connection details"""
    try:
        pod = runpod.get_pod(pod_id)
        
        print("\n=== Connection Information ===")
        print(f"Pod ID: {pod['id']}")
        print(f"Status: {pod.get('desiredStatus', 'unknown')}")
        
        runtime = pod.get('runtime', {})
        
        if 'ports' in runtime:
            print(f"\nWeb Access:")
            for port_info in runtime['ports']:
                print(f"  Port {port_info['privatePort']}: {port_info.get('publicUrl', 'Not available')}")
        
        # SSH info
        ssh_host = runtime.get('sshHost')
        ssh_port = runtime.get('sshPort')
        
        if ssh_host and ssh_port:
            print(f"\nSSH Command:")
            print(f"  ssh root@{ssh_host} -p {ssh_port}")
        else:
            print("\nSSH: Not yet available")
        
        return pod
    except Exception as e:
        print(f"Error getting pod info: {e}")
        return None

def print_setup_instructions(pod):
    """Print instructions for setting up the pod"""
    print("\n" + "="*60)
    print("POD CREATED SUCCESSFULLY!")
    print("="*60)
    
    if pod:
        print(f"\nPod ID: {pod.get('id', 'unknown')}")
        
        # Get connection info
        runtime = pod.get('runtime', {})
        ssh_host = runtime.get('sshHost')
        ssh_port = runtime.get('sshPort')
        
        if ssh_host and ssh_port:
            print(f"\nSSH Connection:")
            print(f"  ssh root@{ssh_host} -p {ssh_port}")
        else:
            print(f"\n⚠ SSH not yet ready. Check RunPod console:")
            print(f"  https://www.runpod.io/console/pods")
    
    print("\n" + "="*60)
    print("NEXT STEPS - Run these commands on the pod:")
    print("="*60)
    
    setup_commands = """
# 1. Clone your repository
git clone https://github.com/Tanziruz/HeartBeatIrregularity.git
cd HeartBeatIrregularity

# 2. Install dependencies
pip install -r requirements.txt
pip install protobuf==3.20.3

# 3. Upload your data
# Option A: SCP from your local machine
# scp -P <SSH_PORT> data/mel_128.h5 root@<POD_HOST>:~/HeartBeatIrregularity/data/

# Option B: Regenerate on pod (recommended - takes ~5 min)
python prepare_data.py

# 4. Run training
python train_fold_validation.py -c config/config_crnn.json

# 5. Monitor with TensorBoard (access via RunPod web interface)
tensorboard --logdir=saved/log --host=0.0.0.0 --port=6006
"""
    print(setup_commands)
    print("="*60)
    print("\n💡 TIP: Access your pod via RunPod Console for easier management:")
    print("   https://www.runpod.io/console/pods")
    print("="*60)

def main():
    """Main setup flow"""
    print("="*60)
    print("RunPod Training Setup for HeartBeat Irregularity")
    print("="*60)
    
    # Step 1: Setup API
    setup_runpod()
    
    # Step 2: Show available GPUs
    print("\nFetching available GPU types...")
    gpus = list_gpu_types()
    
    if not gpus:
        print("Cannot proceed without available GPUs. Please check your RunPod account.")
        return
    
    # Step 3: Interactive pod creation
    print("\n" + "="*60)
    print("Pod Creation Options:")
    print("="*60)
    print("1. Create new pod (recommended)")
    print("2. List existing pods")
    print("3. Get connection info for existing pod")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        # Let user select GPU from list
        print("\nAvailable GPUs:")
        for idx, gpu in enumerate(gpus, 1):
            print(f"{idx}. {gpu['displayName']} - ${gpu.get('lowestPrice', {}).get('minimumBidPrice', 'N/A')}/hr")
        
        gpu_idx = input(f"\nSelect GPU (1-{len(gpus)}) or press Enter for first available: ").strip()
        
        if gpu_idx and gpu_idx.isdigit() and 1 <= int(gpu_idx) <= len(gpus):
            selected_gpu = gpus[int(gpu_idx) - 1]
        else:
            selected_gpu = gpus[0]  # Default to first available
        
        print(f"\nSelected: {selected_gpu['displayName']} (ID: {selected_gpu['id']})")
        
        disk_size = input("Enter disk size in GB (default 50): ").strip()
        disk_size = int(disk_size) if disk_size else 50
        
        pod = create_pod(gpu_type_id=selected_gpu['id'], disk_size=disk_size)
        print_setup_instructions(pod)
        
    elif choice == "2":
        try:
            pods = runpod.get_pods()
            print("\n=== Your Pods ===")
            if not pods:
                print("No pods found.")
            else:
                for pod in pods:
                    runtime = pod.get('runtime', {})
                    print(f"ID: {pod['id']}")
                    print(f"  Name: {pod.get('name', 'N/A')}")
                    print(f"  Status: {pod.get('desiredStatus', 'unknown')}")
                    print(f"  GPU: {pod.get('gpuType', 'N/A')}")
                    print()
        except Exception as e:
            print(f"Error listing pods: {e}")
        
    elif choice == "3":
        pod_id = input("Enter Pod ID: ").strip()
        get_pod_connection_info(pod_id)
    
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
