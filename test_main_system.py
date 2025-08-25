#!/usr/bin/env python3
"""
Test script for the main system
"""

import sys
import time

# Add src to path
sys.path.insert(0, 'src')

from main import SpeechMonitor
import utils


def test_main_system():
    """Test the main system for a short time"""
    print("Testing Main System")
    print("===================")
    
    try:
        # Create speech monitor
        monitor = SpeechMonitor()
        
        # Initialize system
        monitor.initialize()
        print("System initialized successfully")
        
        # Run for 30 seconds
        print("Running system for 30 seconds...")
        print("Speak into microphone to test ASR and keyword detection")
        print("Press Ctrl+C to stop early")
        
        monitor.run(duration_seconds=30)
        
        print("\nMain system test completed!")
        return True
        
    except Exception as e:
        print(f"Main system test failed: {e}")
        return False


def main():
    """Main test function"""
    print("Main System Test")
    print("================")
    
    success = test_main_system()
    
    if success:
        print("\nMain system test passed! ✓")
        return 0
    else:
        print("\nMain system test failed! ✗")
        return 1


if __name__ == "__main__":
    exit(main())
