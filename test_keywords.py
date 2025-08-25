#!/usr/bin/env python3
"""
Test script for keyword detector functionality
"""

import sys
import yaml

# Add src to path
sys.path.insert(0, 'src')

from detector_keywords import KeywordDetector


def test_keyword_detector():
    """Test keyword detector with various phrases"""
    print("Testing Keyword Detector")
    print("========================")
    
    # Load keywords config
    with open('config/keywords.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create detector
    detector = KeywordDetector.from_config(config['stems'])
    
    # Test phrases
    test_phrases = [
        ("normal conversation", False),
        ("my password is secret", True),
        ("credit card number", True),
        ("confidential information", True),
        ("social security number", True),
        ("bank routing number", True),
        ("pin number is 1234", True),
        ("hello world", False),
        ("the weather is nice", False),
        ("my secret is safe", True),
    ]
    
    print(f"Testing {len(test_phrases)} phrases...")
    print()
    
    for phrase, expected in test_phrases:
        result = detector.scan(phrase)
        matches = detector.scan_with_details(phrase)
        status = "✓" if result == expected else "✗"
        
        print(f"{status} '{phrase}' -> {result} (expected {expected})")
        if matches:
            print(f"    Matches: {matches}")
    
    print()
    print("Keyword detector test completed!")


if __name__ == "__main__":
    test_keyword_detector()
