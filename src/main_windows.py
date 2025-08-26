#!/usr/bin/env python3
"""
MVP Speech Monitoring Prototype for Windows
Main orchestrator that coordinates audio, ASR, LLM prediction, and DAF triggering.
Optimized for Windows with GPU acceleration support.
"""

import yaml
import time
import threading
import queue
import numpy as np
import os
import platform
from collections import deque
from typing import Optional
from pathlib import Path

# Local imports
from daf_ring import DAFRing
from audio_io_windows import AudioIOWindows
from asr_vosk import ASRVosk
from predictor_llamacpp_windows import PredictorLlamaCPPWindows
from detector_keywords import KeywordDetector
from state import DAFState
from device_detector import DeviceDetector
import utils


class SpeechMonitorWindows:
    """Windows-optimized speech monitoring system"""
    
    def __init__(self, config_dir="config"):
        """Initialize Windows speech monitoring system"""
        self.config_dir = config_dir
        
        # Load configurations
        self.audio_config = self._load_config("audio_windows.yml")
        self.keywords_config = self._load_config("keywords.yml")
        
        # Initialize components
        self.daf_ring = None
        self.audio_io = None
        self.asr = None
        self.predictor = None
        self.detector = None
        self.state = None
        
        # ASR worker thread
        self.asr_thread = None
        self.asr_running = False
        self.asr_queue = queue.Queue(maxsize=200)
        
        # State tracking
        self.partial_text = ""
        self.horizon_text = ""
        self.last_partial_length = 0
        self.trigger_ring = deque(maxlen=6)  # Last 6 trigger results
        self.last_llm_call_time = 0
        self.llm_call_interval_ms = 100
        
        # ASR monitoring
        self.last_partial_update_time = 0
        self.asr_stuck_threshold_ms = 5000  # 5 seconds without updates
        
        # Performance tracking
        self.start_time = None
        self.frame_count = 0
        
        # Debug mode
        self.debug_mode = True  # Set to True for verbose logging
        
        # Windows-specific settings
        self.gpu_acceleration = self.audio_config.get('gpu_acceleration', True)
        
    def _load_config(self, filename: str) -> dict:
        """Load YAML configuration file with Windows path handling"""
        config_path = Path(self.config_dir) / filename
        
        # Try Windows-specific config first, fall back to original
        if not config_path.exists() and filename == "audio_windows.yml":
            config_path = Path(self.config_dir) / "audio.yml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config {config_path}: {e}")
    
    def initialize(self):
        """Initialize all system components with Windows optimizations"""
        utils.setup_logging()
        utils.log_audio_status("Initializing Windows speech monitoring system...")
        
        try:
            # Check Windows-specific requirements
            self._check_windows_requirements()
            
            # Initialize device detector
            self.device_detector = DeviceDetector()
            
            # Get Bluetooth adjustment
            bluetooth_adjustment = self.device_detector.get_daf_adjustment(
                self.audio_config.get('input_device'),
                self.audio_config.get('output_device')
            )
            
            # Log device information
            device_info = self.device_detector.get_current_devices_info()
            utils.log_audio_status(f"Input device: {device_info['input']['name']} (Bluetooth: {device_info['input']['is_bluetooth']})")
            utils.log_audio_status(f"Output device: {device_info['output']['name']} (Bluetooth: {device_info['output']['is_bluetooth']})")
            utils.log_audio_status(f"DAF compensation: -{bluetooth_adjustment}ms for Bluetooth latency")
            
            # Initialize DAF ring buffer
            self.daf_ring = DAFRing(
                sample_rate=self.audio_config['sample_rate'],
                delay_ms=self.audio_config['daf_delay_ms'],
                bluetooth_adjustment_ms=bluetooth_adjustment
            )
            self.daf_ring.set_output_gain(self.audio_config['output_gain'])
            self.daf_ring.set_limiter_ceiling_db(self.audio_config['limiter_ceiling_db'])
            utils.log_audio_status(
                f"DAF target: {self.audio_config['daf_delay_ms']}ms, BT est: {bluetooth_adjustment}ms, ring delay applied: {self.daf_ring.total_delay_ms}ms"
            )
            
            # Initialize Windows audio I/O
            self.audio_io = AudioIOWindows(
                sample_rate=self.audio_config['sample_rate'],
                frame_ms=self.audio_config['frame_ms'],
                input_gain=self.audio_config['input_gain'],
                input_device=self.audio_config['input_device'],
                output_device=self.audio_config['output_device'],
                                 vad_enabled=self.audio_config.get('vad_enabled', True),
                 vad_rms_threshold=self.audio_config.get('vad_rms_threshold', 0.015)
            )
            self.audio_io.set_daf_ring(self.daf_ring)
            
            # Initialize ASR
            self.asr = ASRVosk(
                sample_rate=self.audio_config['sample_rate']
            )
            
            # Initialize Windows LLM predictor with GPU support
            model_path = "models/llm/llama-3.2-1b-q4_k_m.gguf"
            self.predictor = PredictorLlamaCPPWindows(
                model_path=model_path,
                context_tokens=self.keywords_config['context_tokens'],
                gpu_acceleration=self.gpu_acceleration,
                gpu_layers=32  # Optimize for RTX 3060
            )
            
            # Initialize keyword detector
            self.detector = KeywordDetector.from_config(
                self.keywords_config['stems']
            )
            
            # Initialize DAF state
            self.state = DAFState(
                hold_ms=self.audio_config['hold_ms']
            )
            
            utils.log_audio_status("All Windows components initialized successfully")
            
        except Exception as e:
            utils.log_error("Windows initialization failed", e)
            raise
    
    def _check_windows_requirements(self):
        """Check Windows-specific requirements"""
        utils.log_audio_status("Checking Windows requirements...")
        
        # Check if running on Windows
        if platform.system() != "Windows":
            utils.log_audio_status("Warning: Not running on Windows, some optimizations may not work")
        
        # Check GPU availability if GPU acceleration is enabled
        if self.gpu_acceleration:
            try:
                from predictor_llamacpp_windows import check_gpu_availability
                gpu_info = check_gpu_availability()
                
                if gpu_info['cuda_available']:
                    utils.log_audio_status(f"GPU acceleration available: {gpu_info['gpu_name']}")
                else:
                    utils.log_audio_status("GPU acceleration requested but CUDA not available, falling back to CPU")
                    self.gpu_acceleration = False
                    
            except Exception as e:
                utils.log_audio_status(f"GPU check failed: {e}, falling back to CPU")
                self.gpu_acceleration = False
        
        # Check audio backend
        utils.log_audio_status("Windows audio backend will be auto-selected")
    
    def start_asr_worker(self):
        """Start ASR worker thread"""
        if self.asr_thread and self.asr_thread.is_alive():
            return
        
        self.asr_running = True
        self.asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
        self.asr_thread.start()
        utils.log_audio_status("ASR worker thread started")
    
    def _asr_worker(self):
        """ASR worker thread function"""
        frame_count = 0
        while self.asr_running:
            try:
                # Get frame from audio I/O
                frame = self.audio_io.get_frame(timeout=0.1)
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Convert to PCM bytes for Vosk
                pcm_bytes = self.asr.convert_float32_to_int16_bytes(frame)
                
                # Feed to ASR
                self.asr.accept(pcm_bytes)
                
                # Check for new partial
                new_partial = self.asr.get_partial_if_new()
                if new_partial:
                    self.partial_text = new_partial
                    utils.log_partial(new_partial)
                
                # Log frame processing every 1000 frames (about 20 seconds)
                if frame_count % 1000 == 0:
                    queue_size = self.audio_io.get_queue_size()
                    utils.log_audio_status(f"ASR worker processed {frame_count} frames, queue size: {queue_size}")
                
            except Exception as e:
                utils.log_error("ASR worker error", e)
                time.sleep(0.01)
    
    def _should_call_llm(self) -> bool:
        """Determine if LLM should be called based on partial growth"""
        current_time = utils.get_timestamp_ms()
        
        # Check time interval
        if current_time - self.last_llm_call_time < self.llm_call_interval_ms:
            return False
        
        # Check if partial has grown significantly
        if len(self.partial_text) - self.last_partial_length >= 4:
            self.last_llm_call_time = current_time
            self.last_partial_length = len(self.partial_text)
            return True
        
        return False
    
    def _get_partial_tail(self, words: int = 15) -> str:
        """Get last N words from partial text"""
        if not self.partial_text:
            return ""
        # When DAF is active and we latch until silence, don't truncate the partial
        if self.state and self.state.is_active() and self.audio_config.get('daf_latch_until_silence', True):
            return self.partial_text
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
        
        # Check for consecutive hits
        recent_hits = list(self.trigger_ring)[-self.audio_config['consecutive_hits']:]
        return all(recent_hits)
    
    def update_cycle(self):
        """Main update cycle - called every 40ms"""
        try:
            # Get current partial text (always available)
            current_partial = self.asr.get_current_partial()
            if current_partial != self.partial_text:
                self.partial_text = current_partial
                self.last_partial_update_time = utils.get_timestamp_ms()
                # Log when partial text changes (only non-empty for signal)
                if current_partial.strip():
                    utils.log_partial(current_partial)
                else:
                    # Clear stale horizon when partial becomes empty
                    self.horizon_text = ""
            
            # Debug logging
            if self.debug_mode and self.frame_count % 100 == 0:  # Every 4 seconds
                utils.log_audio_status(f"Debug - Partial: '{self.partial_text}', Rolling: '{self.asr.get_rolling_text()[:30]}...'")
            
            # Check if ASR is stuck (no updates for a while)
            current_time = utils.get_timestamp_ms()
            if (self.last_partial_update_time > 0 and 
                current_time - self.last_partial_update_time > self.asr_stuck_threshold_ms):
                utils.log_error("ASR appears stuck, resetting...")
                self.asr.reset()
                self.last_partial_update_time = current_time
            
            # Check if LLM should be called
            if self._should_call_llm():
                # Get rolling text for context
                rolling_text = self.asr.get_rolling_text()
                if rolling_text:
                    # Call LLM predictor
                    params = {
                        'min_pred_tokens': self.keywords_config['min_pred_tokens'],
                        'max_pred_tokens': self.keywords_config['max_pred_tokens'],
                        'top_k': self.keywords_config['top_k']
                    }
                    
                    self.horizon_text = self.predictor.predict_horizon(rolling_text, params)
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
            
            # VAD-based deactivation: if silence for configured window, close DAF early
            if self.state.is_active() and self.audio_config.get('vad_enabled', True):
                last_voice_ms = self.audio_io.get_last_voice_time_ms()
                silence_ms = current_time - last_voice_ms if last_voice_ms > 0 else float('inf')
                silence_window_ms = self.audio_config.get('vad_silence_ms', 1000)
                if silence_ms >= silence_window_ms:
                    self.state.off()
                    if self.daf_ring.active:
                        self.daf_ring.set_active(False)
                    # Reset trigger ring to avoid immediate re-trigger from stale hits
                    try:
                        self.trigger_ring.clear()
                    except Exception:
                        self.trigger_ring = deque(maxlen=6)
                    # Clear horizon text to avoid stale LLM hits
                    self.horizon_text = ""
                    utils.log_daf_transition(False, f"VAD silence ({int(silence_ms)}ms >= {silence_window_ms}ms)")
            
            # Update DAF state (hold timer) unless latching until silence
            latch_until_silence = self.audio_config.get('daf_latch_until_silence', True) and self.audio_config.get('vad_enabled', True)
            if not latch_until_silence:
                if not self.state.update() and self.daf_ring.active:
                    self.daf_ring.set_active(False)
                    utils.log_daf_transition(False, "Hold timer expired")
            
        except Exception as e:
            utils.log_error("Update cycle error", e)
    
    def run(self, duration_seconds: int = 120):
        """Run the Windows speech monitoring system"""
        utils.log_audio_status(f"Starting Windows speech monitoring for {duration_seconds} seconds")
        
        try:
            # Start audio I/O
            self.audio_io.start()
            
            # Start ASR worker
            self.start_asr_worker()
            
            # Main loop
            self.start_time = utils.get_timestamp_ms()
            cycle_interval_ms = 40  # 40ms update cycle
            last_cycle_time = self.start_time
            
            while True:
                current_time = utils.get_timestamp_ms()
                elapsed_time = current_time - self.start_time
                
                # Check if duration exceeded
                if elapsed_time > duration_seconds * 1000:
                    break
                
                # Run update cycle every 40ms
                if current_time - last_cycle_time >= cycle_interval_ms:
                    self.update_cycle()
                    last_cycle_time = current_time
                    self.frame_count += 1
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
            
        except KeyboardInterrupt:
            utils.log_audio_status("Interrupted by user")
        except Exception as e:
            utils.log_error("Runtime error", e)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up system resources"""
        utils.log_audio_status("Cleaning up Windows system...")
        
        # Stop ASR worker
        self.asr_running = False
        if self.asr_thread and self.asr_thread.is_alive():
            self.asr_thread.join(timeout=1.0)
        
        # Stop audio I/O
        if self.audio_io:
            self.audio_io.stop()
        
        # Log final statistics
        if self.start_time:
            total_time = utils.get_timestamp_ms() - self.start_time
            utils.log_audio_status(f"Windows run completed: {utils.format_duration_ms(total_time)}, {self.frame_count} cycles")


def main():
    """Main entry point for Windows"""
    print("MVP Speech Monitoring Prototype for Windows")
    print("============================================")
    
    # Check for required model files
    required_files = [
        "models/asr/vosk-model-small-en-us",
        "models/llm/llama-3.2-1b-q4_k_m.gguf"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: Missing required model files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease download and place the required model files in the models/ directory.")
        print("Run 'python setup_windows.py' for setup instructions.")
        return 1
    
    # Create and run Windows speech monitor
    monitor = SpeechMonitorWindows()
    
    try:
        monitor.initialize()
        monitor.run(duration_seconds=120)  # 2 minutes
        return 0
    except Exception as e:
        utils.log_error("Fatal Windows error", e)
        return 1


if __name__ == "__main__":
    exit(main())
