#!/usr/bin/env python3
"""
Test script for DAF ring buffer and audio I/O
This tests the basic audio functionality without requiring ASR/LLM models.
"""

import sys
import os
import time
import yaml

# Add src to path
sys.path.insert(0, 'src')

from daf_ring import DAFRing
from audio_io import AudioIO
from state import DAFState
import utils


def test_daf_ring():
    """Test DAF ring buffer functionality"""
    print("Testing DAF ring buffer...")
    
    # Create DAF ring
    daf = DAFRing(sample_rate=16000, delay_ms=175)
    
    # Test frame size
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
    
    print("DAF ring buffer test passed ✓")


def test_audio_io():
    """Test audio I/O functionality"""
    print("\nTesting audio I/O...")
    
    # Load audio config
    with open('config/audio.yml', 'r') as f:
        audio_config = yaml.safe_load(f)
    
    # Create DAF ring
    daf = DAFRing(
        sample_rate=audio_config['sample_rate'],
        delay_ms=audio_config['daf_delay_ms']
    )
    daf.set_output_gain(audio_config['output_gain'])
    daf.set_limiter_ceiling_db(audio_config['limiter_ceiling_db'])
    
    # Create audio I/O
    audio = AudioIO(
        sample_rate=audio_config['sample_rate'],
        frame_ms=audio_config['frame_ms'],
        input_gain=audio_config['input_gain'],
        input_device=audio_config['input_device'],
        output_device=audio_config['output_device']
    )
    audio.set_daf_ring(daf)
    
    print(f"Audio I/O initialized: {audio.sample_rate}Hz, {audio.frame_ms}ms frames")
    print("Audio I/O test setup complete ✓")


def test_daf_activation():
    """Test DAF activation for 10 seconds"""
    print("\nTesting DAF activation for 10 seconds...")
    print("You should hear your voice with a 175ms delay.")
    print("Press Ctrl+C to stop early.")
    
    # Load audio config
    with open('config/audio.yml', 'r') as f:
        audio_config = yaml.safe_load(f)
    
    # Create DAF ring
    daf = DAFRing(
        sample_rate=audio_config['sample_rate'],
        delay_ms=audio_config['daf_delay_ms']
    )
    daf.set_output_gain(audio_config['output_gain'])
    daf.set_limiter_ceiling_db(audio_config['limiter_ceiling_db'])
    
    # Create audio I/O
    audio = AudioIO(
        sample_rate=audio_config['sample_rate'],
        frame_ms=audio_config['frame_ms'],
        input_gain=audio_config['input_gain'],
        input_device=audio_config['input_device'],
        output_device=audio_config['output_device']
    )
    audio.set_daf_ring(daf)
    
    try:
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
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        audio.stop()
        print("Audio test completed ✓")


def main():
    """Main test function"""
    print("DAF Ring Buffer and Audio I/O Test")
    print("===================================")
    
    # Check if config files exist
    if not os.path.exists('config/audio.yml'):
        print("ERROR: config/audio.yml not found")
        return 1
    
    try:
        # Test DAF ring buffer
        test_daf_ring()
        
        # Test audio I/O setup
        test_audio_io()
        
        # Ask user if they want to test audio
        print("\nDo you want to test audio I/O with DAF activation? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            test_daf_activation()
        
        print("\nAll tests completed successfully! ✓")
        return 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    import numpy as np
    exit(main())
