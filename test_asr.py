#!/usr/bin/env python3
"""
Test script for ASR functionality
"""

import sys
import time
import yaml

# Add src to path
sys.path.insert(0, 'src')

from asr_vosk import ASRVosk
import utils


def test_asr_basic():
    """Test basic ASR functionality"""
    print("Testing ASR Basic Functionality")
    print("===============================")
    
    try:
        # Initialize ASR
        asr = ASRVosk(sample_rate=16000)
        print("ASR initialized successfully")
        
        # Test partial detection
        print("\nTesting partial detection...")
        for i in range(10):
            partial = asr.get_current_partial()
            print(f"  Partial {i}: '{partial}'")
            time.sleep(0.1)
        
        # Test rolling text
        print(f"\nRolling text: '{asr.get_rolling_text()}'")
        
        print("\nASR basic test completed!")
        
    except Exception as e:
        print(f"ASR test failed: {e}")
        return False
    
    return True


def test_asr_partial_detection():
    """Test partial detection logic"""
    print("\nTesting ASR Partial Detection Logic")
    print("====================================")
    
    try:
        asr = ASRVosk(sample_rate=16000)
        
        print("Testing get_partial_if_new()...")
        for i in range(20):
            partial = asr.get_partial_if_new()
            if partial:
                print(f"  New partial {i}: '{partial}'")
            else:
                print(f"  No new partial {i}")
            time.sleep(0.1)
        
        print("\nASR partial detection test completed!")
        
    except Exception as e:
        print(f"ASR partial detection test failed: {e}")
        return False
    
    return True


def main():
    """Main test function"""
    print("ASR Functionality Test")
    print("======================")
    
    # Check if config files exist
    if not os.path.exists('config/audio.yml'):
        print("ERROR: config/audio.yml not found")
        return 1
    
    # Run tests
    success = True
    
    success &= test_asr_basic()
    success &= test_asr_partial_detection()
    
    if success:
        print("\nAll ASR tests passed! ✓")
        return 0
    else:
        print("\nSome ASR tests failed! ✗")
        return 1


if __name__ == "__main__":
    import os
    exit(main())
