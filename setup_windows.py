#!/usr/bin/env python3
"""
Windows setup script for predictive jamming system
Handles dependency installation and GPU configuration for RTX 3060
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print("ERROR: Python 3.8+ required")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro} ✓")
    return True

def check_windows():
    """Check if running on Windows"""
    if platform.system() != "Windows":
        print("ERROR: This script is for Windows only")
        return False
    print(f"Windows {platform.release()} ✓")
    return True

def install_llama_cpp_gpu():
    """Install llama-cpp-python with CUDA support for RTX 3060"""
    print("Installing llama-cpp-python with CUDA support...")
    
    # Set environment variables for CUDA compilation
    env = os.environ.copy()
    env['CMAKE_ARGS'] = "-DLLAMA_CUBLAS=on"
    env['FORCE_CMAKE'] = "1"
    
    try:
        # Uninstall existing llama-cpp-python if present
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "llama-cpp-python", "-y"], 
                      check=False, capture_output=True)
        
        # Install with CUDA support
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "llama-cpp-python==0.2.90", "--force-reinstall", "--no-cache-dir"
        ], env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("llama-cpp-python with CUDA support installed ✓")
            return True
        else:
            print(f"Failed to install llama-cpp-python: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error installing llama-cpp-python: {e}")
        return False

def install_dependencies():
    """Install all dependencies"""
    print("Installing dependencies...")
    
    try:
        # Install basic dependencies first
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("Basic dependencies installed ✓")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "models/asr",
        "models/llm",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def download_models():
    """Provide instructions for downloading models"""
    print("\n" + "="*60)
    print("MODEL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nYou need to download the following models:")
    print("\n1. Vosk ASR Model:")
    print("   - Download: https://alphacephei.com/vosk/models")
    print("   - File: vosk-model-small-en-us")
    print("   - Extract to: models/asr/vosk-model-small-en-us/")
    print("\n2. Llama LLM Model:")
    print("   - Download: https://huggingface.co/TheBloke/Llama-3.2-1B-GGUF")
    print("   - File: llama-3.2-1b-q4_k_m.gguf")
    print("   - Place in: models/llm/llama-3.2-1b-q4_k_m.gguf")
    print("\nAfter downloading, run: python test_models.py")

def create_windows_config():
    """Create Windows-specific configuration"""
    print("Creating Windows-specific configuration...")
    
    # Create Windows audio config
    windows_audio_config = {
        'sample_rate': 16000,
        'frame_ms': 20,
        'out_buffer_ms': 20,
        'daf_delay_ms': 175,
        'hold_ms': 600,
        'consecutive_hits': 2,
        'input_device': None,   # Will be auto-detected
        'output_device': None,  # Will be auto-detected
        'input_gain': 1.0,
        'output_gain': 0.9,
        'limiter_ceiling_db': -3.0,
        'vad_enabled': True,
        'vad_rms_threshold': 0.015,
        'vad_silence_ms': 1000,
        'daf_latch_until_silence': True,
        'windows_audio_backend': 'wasapi',  # Windows-specific
        'gpu_acceleration': True  # Enable GPU for LLM
    }
    
    # Save to config file
    import yaml
    with open('config/audio_windows.yml', 'w') as f:
        yaml.dump(windows_audio_config, f, default_flow_style=False)
    
    print("Windows audio configuration created: config/audio_windows.yml")

def main():
    """Main setup function"""
    print("Windows Setup for Predictive Jamming System")
    print("=" * 50)
    
    # Check prerequisites
    if not check_windows():
        return 1
    
    if not check_python_version():
        return 1
    
    try:
        # Create directories
        create_directories()
        
        # Install dependencies
        if not install_dependencies():
            return 1
        
        # Install GPU-accelerated llama-cpp-python
        if not install_llama_cpp_gpu():
            print("Warning: GPU acceleration may not be available")
        
        # Create Windows config
        create_windows_config()
        
        # Provide model download instructions
        download_models()
        
        print("\n" + "="*50)
        print("SETUP COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nNext steps:")
        print("1. Download the required models (see instructions above)")
        print("2. Run: python test_audio_windows.py")
        print("3. Run: python src/main_windows.py")
        
        return 0
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
