"""
GPU information utility
Used to retrieve the GPU name and memory information of the machine.
"""

import subprocess
import platform
import json
from typing import Optional, Dict, Any

# class NodeInfo:
#     def __init__(self):
#         self.node_id: str
#         self.status: str
#         self.gpu_name: str
#         self.gpu_memory: int
# 
    # @classmethod
    # def create_with_detected_gpu(cls, node_id: str, status: str = "available"):
    #     """
    #     Create a NodeInfo instance and automatically detect GPU information.
    #     
    #     Args:
    #         node_id: Node ID
    #         status: Node status, default is "available"
    #         
    #     Returns:
    #         NodeInfo instance
    #     """
    #     gpu_info = get_gpu_name_and_memory()
    #     if not gpu_info:
    #         gpu_info = get_default_gpu_info()
    #     
    #     instance = cls()
    #     instance.node_id = node_id
    #     instance.status = status
    #     instance.gpu_name = gpu_info['gpu_name']
    #     instance.gpu_memory = gpu_info['gpu_memory'] or 8192  # Use default 8GB if None
    #     
    #     return instance

def get_gpu_name_and_memory() -> Optional[Dict[str, Any]]:
    """
    Get the GPU name and memory information of the machine.

    Returns:
        Dict containing 'gpu_name' and 'gpu_memory', or None if not available.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        return _get_macos_gpu_info()
    elif system in ["Linux", "Windows"]:
        return _get_nvidia_gpu_info()

    return None

def _get_nvidia_gpu_info() -> Optional[Dict[str, Any]]:
    """Get NVIDIA GPU information."""
    try:
        # Try to get GPU info using nvidia-smi
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=name,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        if result.stdout.strip():
            name, memory = result.stdout.strip().split(', ')
            return {
                'gpu_name': name.strip(),
                'gpu_memory': int(memory.strip())  # MB
            }
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try using pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get the first GPU
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory_mb = memory_info.total // (1024 * 1024)

            return {
                'gpu_name': name,
                'gpu_memory': total_memory_mb
            }
    except ImportError:
        pass
    except Exception:
        pass

    # Try using PyTorch
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            name = torch.cuda.get_device_name(0)
            memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

            return {
                'gpu_name': name,
                'gpu_memory': memory_mb
            }
    except ImportError:
        pass
    except Exception:
        pass

    return None

def _get_macos_gpu_info() -> Optional[Dict[str, Any]]:
    """Get macOS GPU information."""
    try:
        # Use system_profiler to get GPU info
        result = subprocess.run([
            'system_profiler', 'SPDisplaysDataType', '-json'
        ], capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)

        for display in data.get('SPDisplaysDataType', []):
            if 'sppci_model' in display:
                name = display['sppci_model']

                # Try to get GPU memory info
                gpu_memory = _get_macos_gpu_memory(name)

                return {
                    'gpu_name': name,
                    'gpu_memory': gpu_memory
                }
    except Exception:
        pass

    return None

def _get_macos_gpu_memory(gpu_name: str) -> Optional[int]:
    """Get macOS GPU memory size (MB)."""
    try:
        # Method 1: Try to get VRAM info from system_profiler
        result = subprocess.run([
            'system_profiler', 'SPDisplaysDataType', '-json'
        ], capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)

        for display in data.get('SPDisplaysDataType', []):
            if display.get('sppci_model') == gpu_name:
                # Look for VRAM info
                vram_info = display.get('spdisplays_vram', '')
                if vram_info and 'MB' in vram_info:
                    import re
                    match = re.search(r'(\d+)\s*MB', vram_info)
                    if match:
                        return int(match.group(1))

                # If no VRAM info, try other fields
                # For Apple Silicon, GPU memory is usually shared with system memory
                if 'Apple' in gpu_name or 'M1' in gpu_name or 'M2' in gpu_name or 'M3' in gpu_name:
                    return _get_apple_silicon_gpu_memory()

    except Exception:
        pass

    # Method 2: For Apple Silicon, try to get unified memory info
    if 'Apple' in gpu_name or 'M1' in gpu_name or 'M2' in gpu_name or 'M3' in gpu_name:
        return _get_apple_silicon_gpu_memory()

    return None

def _get_apple_silicon_gpu_memory() -> Optional[int]:
    """Get Apple Silicon GPU memory (unified memory)."""
    try:
        # Get total system memory
        result = subprocess.run([
            'sysctl', 'hw.memsize'
        ], capture_output=True, text=True, check=True)

        if result.stdout:
            memsize_bytes = int(result.stdout.split(':')[1].strip())
            memsize_gb = memsize_bytes // (1024 * 1024 * 1024)
            gpu_memory_mb = int(memsize_gb * 1024)
            return gpu_memory_mb

    except Exception:
        pass

    # Method 3: Use vm_stat to get memory info
    try:
        result = subprocess.run([
            'vm_stat'
        ], capture_output=True, text=True, check=True)

        if result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Pages free' in line:
                    pages_free = int(line.split(':')[1].strip().rstrip('.'))
                    # macOS page size is usually 4KB
                    free_memory_mb = (pages_free * 4096) // (1024 * 1024)
                    return int(free_memory_mb)

    except Exception:
        pass

    return None

def get_default_gpu_info() -> Dict[str, Any]:
    """
    Get default GPU information if detection fails.

    Returns:
        Dict containing default values for 'gpu_name' and 'gpu_memory'.
    """
    gpu_info = get_gpu_name_and_memory()

    if gpu_info:
        return gpu_info

    # Return default values
    return {
        'gpu_name': 'Unknown GPU',
        'gpu_memory': 8192  # Default 8GB
    }

# Example usage
if __name__ == "__main__":
    print("=== GPU Information Query ===")

    gpu_info = get_gpu_name_and_memory()
    if gpu_info:
        print(f"GPU Name: {gpu_info['gpu_name']}")
        print(f"GPU Memory: {gpu_info['gpu_memory']} MB" if gpu_info['gpu_memory'] else "GPU Memory: Unknown")
    else:
        print("Unable to retrieve GPU information")
        default_info = get_default_gpu_info()
        print(f"Using default - GPU Name: {default_info['gpu_name']}, GPU Memory: {default_info['gpu_memory']} MB")
