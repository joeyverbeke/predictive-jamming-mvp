#!/usr/bin/env python3
"""
Test script for Bluetooth device detection and DAF latency adjustment
"""

import sys

# Add src to path
sys.path.insert(0, 'src')

from device_detector import DeviceDetector
from daf_ring import DAFRing
import utils


def test_device_detection():
    """Test Bluetooth device detection"""
    print("Bluetooth Device Detection Test")
    print("===============================")
    
    # Initialize device detector
    detector = DeviceDetector()
    
    # List all devices
    devices = detector.list_devices()
    
    print("\nAvailable Audio Devices:")
    print("-" * 50)
    for device in devices:
        bluetooth_status = "✓ Bluetooth" if device['is_bluetooth'] else "✗ Wired"
        latency_info = f" (~{device['estimated_latency']}ms)" if device.get('estimated_latency') else ""
        print(f"Device {device['id']}: {device['name']} {bluetooth_status}{latency_info}")
    
    # Get current device info
    current_info = detector.get_current_devices_info()
    
    print(f"\nCurrent Devices:")
    print("-" * 50)
    print(f"Input:  {current_info['input']['name']} ({'Bluetooth' if current_info['input']['is_bluetooth'] else 'Wired'})")
    print(f"Output: {current_info['output']['name']} ({'Bluetooth' if current_info['output']['is_bluetooth'] else 'Wired'})")
    print(f"DAF Adjustment: +{current_info['daf_adjustment']}ms")
    
    return current_info


def test_daf_adjustment():
    """Test DAF ring buffer with Bluetooth adjustment"""
    print("\nDAF Latency Adjustment Test")
    print("============================")
    
    # Get device info
    detector = DeviceDetector()
    current_info = detector.get_current_devices_info()
    bluetooth_adjustment = current_info['daf_adjustment']
    
    # Test different DAF configurations
    base_delay = 175  # Standard DAF delay
    
    print(f"\nTarget DAF delay (mouth-to-ear): {base_delay}ms")
    print(f"Estimated Bluetooth I/O latency: {bluetooth_adjustment}ms")
    ring_delay = max(0, base_delay - bluetooth_adjustment)
    print(f"Ring delay to apply (target - BT): {ring_delay}ms")
    print(f"Estimated total (ring + BT): {ring_delay + bluetooth_adjustment}ms ≈ target")
    
    # Create DAF ring with adjustment
    daf_ring = DAFRing(
        sample_rate=16000,
        delay_ms=base_delay,
        bluetooth_adjustment_ms=bluetooth_adjustment
    )
    
    print(f"\nDAF Ring Configuration:")
    print(f"  Sample rate: {daf_ring.sample_rate} Hz")
    print(f"  Target total delay: {daf_ring.delay_ms}ms")
    print(f"  Estimated BT latency: {daf_ring.bluetooth_adjustment_ms}ms")
    print(f"  Ring delay (applied): {daf_ring.total_delay_ms}ms")
    print(f"  Delay samples: {daf_ring.delay_samples}")
    
    # Test dynamic adjustment
    print(f"\nTesting dynamic adjustment...")
    
    # Simulate switching to wired headphones
    daf_ring.update_bluetooth_adjustment(0)
    print(f"  After switching to wired: {daf_ring.total_delay_ms}ms ring delay (no BT)")
    
    # Simulate switching back to Bluetooth
    daf_ring.update_bluetooth_adjustment(bluetooth_adjustment)
    print(f"  After switching to Bluetooth: {daf_ring.total_delay_ms}ms ring delay (with BT)")
    
    return True


def main():
    """Main test function"""
    print("Bluetooth Detection and DAF Adjustment Test")
    print("=" * 50)
    
    try:
        # Test device detection
        device_info = test_device_detection()
        
        # Test DAF adjustment
        test_daf_adjustment()
        
        print(f"\n✅ All tests passed!")
        print(f"\nSummary:")
        print(f"  - Input device: {device_info['input']['name']}")
        print(f"  - Output device: {device_info['output']['name']}")
        print(f"  - DAF adjustment: +{device_info['daf_adjustment']}ms")
        
        if device_info['daf_adjustment'] > 0:
            print(f"  - Bluetooth latency compensation: ENABLED")
        else:
            print(f"  - Bluetooth latency compensation: NOT NEEDED")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
