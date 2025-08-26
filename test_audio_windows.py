#!/usr/bin/env python3
"""
Windows Audio Test Script
Tests audio components and GPU acceleration for the predictive jamming system
"""

import sys
import os
import time
import yaml
import platform

# Add src to path
sys.path.insert(0, 'src')

from daf_ring import DAFRing
from audio_io_windows import AudioIOWindows, list_windows_audio_devices
from state import DAFState
import utils


def test_windows_audio_devices():
    """Test Windows audio device detection"""
    print("Testing Windows Audio Device Detection")
    print("=" * 50)
    
    try:
        list_windows_audio_devices()
        print("✓ Windows audio device detection successful")
        return True
    except Exception as e:
        print(f"✗ Windows audio device detection failed: {e}")
        return False


def test_daf_ring():
    """Test DAF ring buffer functionality"""
    print("\nTesting DAF Ring Buffer")
    print("=" * 30)
    
    try:
        # Create DAF ring
        daf = DAFRing(sample_rate=16000, delay_ms=175)
        
        # Test frame size
        import numpy as np
        test_frame = np.ones(320, dtype=np.float32) * 0.5
        print(f"Frame size: {len(test_frame)} samples")
        
        # Test push/pull
        daf.push(test_frame)
        output_frame = daf.pull()
        print(f"Output frame size: {len(output_frame)} samples")
        print(f"DAF active: {daf.active}")
        
        # Test with DAF active
        daf.set_active(True)
        daf.push(test_frame)
        output_frame = daf.pull()
        print(f"DAF active output max: {np.max(np.abs(output_frame))}")
        
        print("✓ DAF ring buffer test passed")
        return True
        
    except Exception as e:
        print(f"✗ DAF ring buffer test failed: {e}")
        return False


def test_windows_audio_io():
    """Test Windows audio I/O functionality"""
    print("\nTesting Windows Audio I/O")
    print("=" * 30)
    
    try:
        # Load audio config
        config_path = "config/audio_windows.yml"
        if not os.path.exists(config_path):
            config_path = "config/audio.yml"
        
        with open(config_path, 'r') as f:
            audio_config = yaml.safe_load(f)
        
        # Create DAF ring
        daf = DAFRing(
            sample_rate=audio_config['sample_rate'],
            delay_ms=audio_config['daf_delay_ms']
        )
        daf.set_output_gain(audio_config['output_gain'])
        daf.set_limiter_ceiling_db(audio_config['limiter_ceiling_db'])
        
        # Create Windows audio I/O
        audio = AudioIOWindows(
            sample_rate=audio_config['sample_rate'],
            frame_ms=audio_config['frame_ms'],
            input_gain=audio_config['input_gain'],
            input_device=audio_config['input_device'],
            output_device=audio_config['output_device']
        )
        audio.set_daf_ring(daf)
        
        print(f"Audio I/O initialized: {audio.sample_rate}Hz, {audio.frame_ms}ms frames")
        print("✓ Windows audio I/O test setup complete")
        return True
        
    except Exception as e:
        print(f"✗ Windows audio I/O test failed: {e}")
        return False


def test_gpu_acceleration():
    """Test GPU acceleration for LLM"""
    print("\nTesting GPU Acceleration")
    print("=" * 30)
    
    try:
        from predictor_llamacpp_windows import check_gpu_availability
        
        gpu_info = check_gpu_availability()
        
        print(f"CUDA Available: {gpu_info['cuda_available']}")
        if gpu_info['gpu_name']:
            print(f"GPU Name: {gpu_info['gpu_name']}")
        if gpu_info['gpu_memory']:
            print(f"GPU Memory: {gpu_info['gpu_memory']:.1f} GB")
        print(f"Llama-cpp GPU Support: {gpu_info['llama_cpp_gpu']}")
        
        if gpu_info['cuda_available']:
            print("✓ GPU acceleration available")
            return True
        else:
            print("⚠ GPU acceleration not available, will use CPU")
            return True  # Not a failure, just a warning
            
    except Exception as e:
        print(f"✗ GPU acceleration test failed: {e}")
        return False


def test_daf_activation_windows():
    """Test DAF activation for 10 seconds on Windows"""
    print("\nTesting DAF Activation (10 seconds)")
    print("=" * 40)
    print("You should hear your voice with a 175ms delay.")
    print("Press Ctrl+C to stop early.")
    
    try:
        # Load audio config
        config_path = "config/audio_windows.yml"
        if not os.path.exists(config_path):
            config_path = "config/audio.yml"
        
        with open(config_path, 'r') as f:
            audio_config = yaml.safe_load(f)
        
        # Create DAF ring
        daf = DAFRing(
            sample_rate=audio_config['sample_rate'],
            delay_ms=audio_config['daf_delay_ms']
        )
        daf.set_output_gain(audio_config['output_gain'])
        daf.set_limiter_ceiling_db(audio_config['limiter_ceiling_db'])
        
        # Create Windows audio I/O
        audio = AudioIOWindows(
            sample_rate=audio_config['sample_rate'],
            frame_ms=audio_config['frame_ms'],
            input_gain=audio_config['input_gain'],
            input_device=audio_config['input_device'],
            output_device=audio_config['output_device']
        )
        audio.set_daf_ring(daf)
        
        # Start audio
        audio.start()
        
        # Activate DAF after 1 second
        time.sleep(1.0)
        print("Activating DAF...")
        daf.set_active(True)
        
        # Run for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10.0:
            time.sleep(0.1)
        
        # Deactivate DAF
        print("Deactivating DAF...")
        daf.set_active(False)
        
        # Run for 2 more seconds
        time.sleep(2.0)
        
        # Stop audio
        audio.stop()
        print("✓ Windows DAF activation test completed")
        return True
        
    except KeyboardInterrupt:
        print("\nStopped by user")
        if 'audio' in locals():
            audio.stop()
        return True
    except Exception as e:
        print(f"✗ Windows DAF activation test failed: {e}")
        if 'audio' in locals():
            audio.stop()
        return False


def test_model_loading():
    """Test model loading if available"""
    print("\nTesting Model Loading")
    print("=" * 25)
    
    # Check ASR model
    asr_model_path = "models/asr/vosk-model-small-en-us"
    if os.path.exists(asr_model_path):
        print(f"✓ ASR model found: {asr_model_path}")
        asr_available = True
    else:
        print(f"⚠ ASR model not found: {asr_model_path}")
        asr_available = False
    
    # Check LLM model
    llm_model_path = "models/llm/llama-3.2-1b-q4_k_m.gguf"
    if os.path.exists(llm_model_path):
        print(f"✓ LLM model found: {llm_model_path}")
        llm_available = True
    else:
        print(f"⚠ LLM model not found: {llm_model_path}")
        llm_available = False
    
    # Test LLM loading if available
    if llm_available:
        try:
            from predictor_llamacpp_windows import PredictorLlamaCPPWindows
            
            predictor = PredictorLlamaCPPWindows(
                model_path=llm_model_path,
                gpu_acceleration=True
            )
            
            # Test prediction
            test_text = "The weather is"
            prediction = predictor.test_prediction(test_text)
            print(f"Test prediction: '{test_text}' -> '{prediction}'")
            print("✓ LLM model loading and prediction successful")
            
        except Exception as e:
            print(f"✗ LLM model test failed: {e}")
            llm_available = False
    
    return asr_available, llm_available


def main():
    """Main test function"""
    print("Windows Audio and GPU Test Suite")
    print("=" * 40)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print()
    
    # Check if config files exist
    config_files = ["config/audio_windows.yml", "config/audio.yml"]
    config_found = False
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ Found config: {config_file}")
            config_found = True
            break
    
    if not config_found:
        print("✗ No audio config found")
        print("Please run 'python setup_windows.py' first")
        return 1
    
    # Run tests
    tests = [
        ("Windows Audio Devices", test_windows_audio_devices),
        ("DAF Ring Buffer", test_daf_ring),
        ("Windows Audio I/O", test_windows_audio_io),
        ("GPU Acceleration", test_gpu_acceleration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    # Test model loading
    print(f"\n{'='*60}")
    print("Running: Model Loading")
    print('='*60)
    asr_available, llm_available = test_model_loading()
    if asr_available and llm_available:
        passed += 1
    total += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✓ All tests passed! System is ready for use.")
        
        # Ask about DAF test
        print("\nDo you want to test DAF activation with audio? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                test_daf_activation_windows()
        except KeyboardInterrupt:
            print("\nSkipping audio test")
        
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Run 'python setup_windows.py' to install dependencies")
        print("2. Download required models (see setup instructions)")
        print("3. Check Windows audio device settings")
        print("4. Ensure CUDA drivers are installed for GPU acceleration")
        return 1


if __name__ == "__main__":
    exit(main())
