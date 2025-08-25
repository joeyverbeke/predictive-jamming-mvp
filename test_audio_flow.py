#!/usr/bin/env python3
"""
Test script for audio flow from microphone to ASR
"""

import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from audio_io import AudioIO
from asr_vosk import ASRVosk
import utils


def test_audio_to_asr():
    """Test audio flow from microphone to ASR"""
    print("Testing Audio Flow to ASR")
    print("==========================")
    
    try:
        # Initialize components
        audio = AudioIO(sample_rate=16000, frame_ms=20)
        asr = ASRVosk(sample_rate=16000)
        
        print("Components initialized successfully")
        
        # Start audio
        audio.start()
        print("Audio started - speak into microphone for 10 seconds...")
        
        # Monitor audio flow
        frame_count = 0
        partial_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 10:
            # Get frame from audio
            frame = audio.get_frame(timeout=0.1)
            if frame is not None:
                frame_count += 1
                
                # Convert to PCM bytes
                pcm_bytes = asr.convert_float32_to_int16_bytes(frame)
                
                # Feed to ASR
                asr.accept(pcm_bytes)
                
                # Check for partial
                partial = asr.get_partial_if_new()
                if partial:
                    partial_count += 1
                    print(f"  Partial {partial_count}: {partial}")
                
                # Log every 50 frames (about 1 second)
                if frame_count % 50 == 0:
                    queue_size = audio.get_queue_size()
                    print(f"  Processed {frame_count} frames, queue size: {queue_size}")
            
            time.sleep(0.01)
        
        # Stop audio
        audio.stop()
        
        print(f"\nTest completed:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total partials detected: {partial_count}")
        print(f"  Final partial: {asr.get_current_partial()}")
        
        return True
        
    except Exception as e:
        print(f"Audio flow test failed: {e}")
        return False


def main():
    """Main test function"""
    print("Audio Flow Test")
    print("===============")
    
    success = test_audio_to_asr()
    
    if success:
        print("\nAudio flow test passed! ✓")
        return 0
    else:
        print("\nAudio flow test failed! ✗")
        return 1


if __name__ == "__main__":
    exit(main())
