#!/usr/bin/env python3
"""
Test script for DAF state management
"""

import sys
import time

# Add src to path
sys.path.insert(0, 'src')

from state import DAFState


def test_daf_state():
    """Test DAF state management"""
    print("Testing DAF State Management")
    print("============================")
    
    # Create state with 600ms hold time
    state = DAFState(hold_ms=600)
    
    print(f"Initial state: active={state.is_active()}")
    
    # Test activation
    print("\nActivating DAF...")
    state.on()
    print(f"After activation: active={state.is_active()}")
    
    # Test update during hold
    print("\nTesting during hold period...")
    for i in range(5):
        remaining = state.get_remaining_hold_ms()
        print(f"  {i+1}s: remaining={remaining:.0f}ms, active={state.update()}")
        time.sleep(0.2)
    
    # Test after hold expires
    print("\nWaiting for hold to expire...")
    time.sleep(0.8)  # Wait for remaining hold time
    
    result = state.update()
    print(f"After hold expiry: active={state.is_active()}, update_result={result}")
    
    # Test deactivation
    print("\nTesting manual deactivation...")
    state.on()
    print(f"Reactivated: active={state.is_active()}")
    
    state.off()
    print(f"After manual deactivation: active={state.is_active()}")
    
    print("\nDAF state management test completed!")


if __name__ == "__main__":
    test_daf_state()
