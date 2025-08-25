#!/usr/bin/env python3
"""
Demo script for the speech monitoring system
This simulates the full system behavior without requiring actual ASR/LLM models.
"""

import sys
import time
import yaml
import random
from collections import deque

# Add src to path
sys.path.insert(0, 'src')

from daf_ring import DAFRing
from audio_io import AudioIO
from detector_keywords import KeywordDetector
from state import DAFState
import utils


class DemoSpeechMonitor:
    """Demo speech monitoring system with simulated ASR/LLM"""
    
    def __init__(self, config_dir="config"):
        """Initialize demo system"""
        self.config_dir = config_dir
        
        # Load configurations
        self.audio_config = self._load_config("audio.yml")
        self.keywords_config = self._load_config("keywords.yml")
        
        # Initialize components
        self.daf_ring = None
        self.audio_io = None
        self.detector = None
        self.state = None
        
        # Simulated state
        self.partial_text = ""
        self.horizon_text = ""
        self.last_partial_length = 0
        self.trigger_ring = deque(maxlen=6)
        self.last_llm_call_time = 0
        self.llm_call_interval_ms = 100
        
        # Demo phrases for simulation
        self.demo_phrases = [
            "hello world how are you today",
            "my password is secret123",
            "the weather is nice outside",
            "my credit card number is 1234",
            "let's go for a walk",
            "this is confidential information",
            "what time is it",
            "my social security number is",
            "nice to meet you",
            "the pin number is 5678",
        ]
        
        self.current_phrase_idx = 0
        self.phrase_progress = 0
        
    def _load_config(self, filename: str) -> dict:
        """Load YAML configuration file"""
        config_path = f"{self.config_dir}/{filename}"
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config {config_path}: {e}")
    
    def initialize(self):
        """Initialize demo system components"""
        utils.setup_logging()
        utils.log_audio_status("Initializing demo system...")
        
        try:
            # Initialize DAF ring buffer
            self.daf_ring = DAFRing(
                sample_rate=self.audio_config['sample_rate'],
                delay_ms=self.audio_config['daf_delay_ms']
            )
            self.daf_ring.set_output_gain(self.audio_config['output_gain'])
            self.daf_ring.set_limiter_ceiling_db(self.audio_config['limiter_ceiling_db'])
            
            # Initialize audio I/O
            self.audio_io = AudioIO(
                sample_rate=self.audio_config['sample_rate'],
                frame_ms=self.audio_config['frame_ms'],
                input_gain=self.audio_config['input_gain'],
                input_device=self.audio_config['input_device'],
                output_device=self.audio_config['output_device']
            )
            self.audio_io.set_daf_ring(self.daf_ring)
            
            # Initialize keyword detector
            self.detector = KeywordDetector.from_config(
                self.keywords_config['stems']
            )
            
            # Initialize DAF state
            self.state = DAFState(
                hold_ms=self.audio_config['hold_ms']
            )
            
            utils.log_audio_status("Demo system initialized successfully")
            
        except Exception as e:
            utils.log_error("Demo initialization failed", e)
            raise
    
    def simulate_asr_partial(self):
        """Simulate ASR partial text generation"""
        if self.current_phrase_idx >= len(self.demo_phrases):
            # Cycle through phrases
            self.current_phrase_idx = 0
            self.phrase_progress = 0
        
        phrase = self.demo_phrases[self.current_phrase_idx]
        words = phrase.split()
        
        # Progress through the phrase word by word
        if self.phrase_progress < len(words):
            self.partial_text = " ".join(words[:self.phrase_progress + 1])
            self.phrase_progress += 1
        else:
            # Move to next phrase
            self.current_phrase_idx += 1
            self.phrase_progress = 0
            self.partial_text = ""
        
        return self.partial_text
    
    def simulate_llm_prediction(self, partial_text):
        """Simulate LLM prediction based on partial text"""
        if not partial_text:
            return ""
        
        # Simple simulation: predict some continuation based on keywords
        if any(stem in partial_text.lower() for stem in self.keywords_config['stems']):
            # If keywords detected, predict risky continuation
            predictions = ["secret", "password", "number", "code", "key"]
            return random.choice(predictions)
        else:
            # Normal continuation
            predictions = ["is", "the", "a", "and", "but"]
            return random.choice(predictions)
    
    def _should_call_llm(self) -> bool:
        """Determine if LLM should be called"""
        current_time = utils.get_timestamp_ms()
        
        if current_time - self.last_llm_call_time < self.llm_call_interval_ms:
            return False
        
        if len(self.partial_text) - self.last_partial_length >= 4:
            self.last_llm_call_time = current_time
            self.last_partial_length = len(self.partial_text)
            return True
        
        return False
    
    def _get_partial_tail(self, words: int = 15) -> str:
        """Get last N words from partial text"""
        if not self.partial_text:
            return ""
        
        words_list = self.partial_text.split()
        if len(words_list) <= words:
            return self.partial_text
        
        return " ".join(words_list[-words:])
    
    def _update_trigger_ring(self, hit: bool):
        """Update trigger ring buffer"""
        self.trigger_ring.append(hit)
    
    def _check_trigger_condition(self) -> bool:
        """Check if trigger condition is met"""
        if len(self.trigger_ring) < self.audio_config['consecutive_hits']:
            return False
        
        recent_hits = list(self.trigger_ring)[-self.audio_config['consecutive_hits']:]
        return all(recent_hits)
    
    def update_cycle(self):
        """Main update cycle with simulated components"""
        try:
            # Simulate ASR partial
            new_partial = self.simulate_asr_partial()
            if new_partial and new_partial != self.partial_text:
                self.partial_text = new_partial
                utils.log_partial(new_partial)
            
            # Check if LLM should be called
            if self._should_call_llm():
                self.horizon_text = self.simulate_llm_prediction(self.partial_text)
                if self.horizon_text:
                    utils.log_horizon(self.horizon_text)
            
            # Get partial tail for scanning
            partial_tail = self._get_partial_tail()
            
            # Scan for keywords
            hit_asr = self.detector.scan(partial_tail)
            hit_llm = self.detector.scan(self.horizon_text) if self.horizon_text else False
            
            # Log hit detection
            utils.log_hit_detection(hit_asr, hit_llm, partial_tail, self.horizon_text)
            
            # Update trigger ring
            hit = hit_asr or hit_llm
            self._update_trigger_ring(hit)
            
            # Check trigger condition
            if self._check_trigger_condition() and not self.state.is_active():
                self.state.on()
                self.daf_ring.set_active(True)
                utils.log_daf_transition(True, "Trigger condition met")
            
            # Update DAF state
            if not self.state.update() and self.daf_ring.active:
                self.daf_ring.set_active(False)
                utils.log_daf_transition(False, "Hold timer expired")
            
        except Exception as e:
            utils.log_error("Demo update cycle error", e)
    
    def run(self, duration_seconds: int = 60):
        """Run the demo system"""
        utils.log_audio_status(f"Starting demo for {duration_seconds} seconds")
        
        try:
            # Start audio I/O
            self.audio_io.start()
            
            # Main loop
            start_time = utils.get_timestamp_ms()
            cycle_interval_ms = 40  # 40ms update cycle
            last_cycle_time = start_time
            
            while True:
                current_time = utils.get_timestamp_ms()
                elapsed_time = current_time - start_time
                
                # Check if duration exceeded
                if elapsed_time > duration_seconds * 1000:
                    break
                
                # Run update cycle every 40ms
                if current_time - last_cycle_time >= cycle_interval_ms:
                    self.update_cycle()
                    last_cycle_time = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
            
        except KeyboardInterrupt:
            utils.log_audio_status("Demo interrupted by user")
        except Exception as e:
            utils.log_error("Demo runtime error", e)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up demo system"""
        utils.log_audio_status("Cleaning up demo...")
        
        if self.audio_io:
            self.audio_io.stop()
        
        utils.log_audio_status("Demo completed")


def main():
    """Main demo function"""
    print("Demo Speech Monitoring System")
    print("=============================")
    print("This demo simulates the full system behavior without requiring ASR/LLM models.")
    print("It will cycle through test phrases and demonstrate DAF triggering.")
    print()
    
    # Check if config files exist
    if not os.path.exists('config/audio.yml') or not os.path.exists('config/keywords.yml'):
        print("ERROR: Configuration files not found")
        return 1
    
    # Create and run demo
    demo = DemoSpeechMonitor()
    
    try:
        demo.initialize()
        demo.run(duration_seconds=60)  # 1 minute demo
        return 0
    except Exception as e:
        utils.log_error("Demo failed", e)
        return 1


if __name__ == "__main__":
    import os
    exit(main())
