import numpy as np
import sounddevice as sd
import queue
import threading
import time
import platform
from typing import Optional, Tuple


class AudioIOWindows:
    """Windows-specific audio input/output with duplex stream"""
    
    def __init__(self, sample_rate=16000, frame_ms=20, input_gain=1.0,
                 input_device=None, output_device=None,
                 vad_enabled: bool = True, vad_rms_threshold: float = 0.015):
        """
        Initialize Windows audio I/O
        
                 Args:
             sample_rate: Audio sample rate in Hz
             frame_ms: Frame duration in milliseconds
             input_gain: Input gain multiplier
             input_device: Input device index (None for default)
             output_device: Output device index (None for default)
             vad_enabled: Enable voice activity detection
             vad_rms_threshold: RMS threshold for VAD
        """
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.samples_per_frame = int(frame_ms * sample_rate / 1000)
        self.input_gain = input_gain
        self.input_device = input_device
        self.output_device = output_device

        
        # Simple RMS-based VAD parameters
        self.vad_enabled = vad_enabled
        self.vad_rms_threshold = vad_rms_threshold
        self._last_voice_time_ms = 0.0
        self._vad_lock = threading.Lock()
        
        # Audio stream
        self.stream = None
        self.is_running = False
        
        # Frame queue for ASR worker
        self.frame_queue = queue.Queue(maxsize=100)
        
        # DAF ring buffer reference (set externally)
        self.daf_ring = None
        
        # Pre-allocate temp arrays for performance
        self.temp_input = np.zeros(self.samples_per_frame, dtype=np.float32)
        self.temp_output = np.zeros(self.samples_per_frame, dtype=np.float32)
        
        # Windows-specific audio settings
        self._setup_windows_audio()
    
    def _setup_windows_audio(self):
        """Setup Windows-specific audio settings"""
        # Note: sounddevice doesn't support setting backend directly on Windows
        # The backend is automatically selected based on available drivers
        if platform.system() == "Windows":
            print(f"Windows audio backend will be auto-selected (WASAPI preferred)")
            # Log available backends for debugging
            try:
                backends = sd.query_hostapis()
                print(f"Available audio backends: {[b['name'] for b in backends]}")
            except Exception as e:
                print(f"Could not query audio backends: {e}")
    
    def _find_best_devices(self) -> Tuple[Optional[int], Optional[int]]:
        """Find the best input and output devices for Windows"""
        try:
            devices = sd.query_devices()
            
            # Find default devices
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            
            # If specific devices not specified, use defaults
            if self.input_device is None:
                self.input_device = default_input
            if self.output_device is None:
                self.output_device = default_output
            
            # Validate devices
            if self.input_device >= len(devices):
                print(f"Warning: Invalid input device {self.input_device}, using default")
                self.input_device = default_input
            
            if self.output_device >= len(devices):
                print(f"Warning: Invalid output device {self.output_device}, using default")
                self.output_device = default_output
            
            # Log device information
            input_device_info = devices[self.input_device]
            output_device_info = devices[self.output_device]
            
            print(f"Input device: {input_device_info.get('name', 'Unknown')} (ID: {self.input_device})")
            print(f"Output device: {output_device_info.get('name', 'Unknown')} (ID: {self.output_device})")
            
            return self.input_device, self.output_device
            
        except Exception as e:
            print(f"Error finding devices: {e}")
            return None, None
    
    def set_daf_ring(self, daf_ring):
        """Set DAF ring buffer reference"""
        self.daf_ring = daf_ring
    
    def audio_callback(self, indata, outdata, frames, time, status):
        """Light audio callback for real-time processing"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Handle different input formats
        if indata.ndim > 1:
            input_frame = indata[:, 0].astype(np.float32)  # Mono input
        else:
            input_frame = indata.astype(np.float32)
        
        # Ensure we have the right number of samples
        if len(input_frame) != self.samples_per_frame:
            # Pad or truncate if necessary
            if len(input_frame) > self.samples_per_frame:
                input_frame = input_frame[:self.samples_per_frame]
            else:
                input_frame = np.pad(input_frame, (0, self.samples_per_frame - len(input_frame)))
        
        # Apply input gain
        for i in range(self.samples_per_frame):
            self.temp_input[i] = input_frame[i] * self.input_gain
        
        # Simple RMS-based VAD (keep extremely light)
        if self.vad_enabled:
            # Vectorized RMS over the frame
            rms = float(np.sqrt(np.mean(self.temp_input * self.temp_input)))
            if rms >= self.vad_rms_threshold:
                with self._vad_lock:
                    self._last_voice_time_ms = time_now_ms()

        # Push to DAF ring
        if self.daf_ring:
            self.daf_ring.push(self.temp_input)
        
        # Add to ASR queue (non-blocking)
        try:
            self.frame_queue.put_nowait(self.temp_input.copy())
        except queue.Full:
            pass  # Drop frame if queue is full
        
        # Get output frame from DAF ring
        if self.daf_ring:
            output_frame = self.daf_ring.pull()
        else:
            output_frame = np.zeros(self.samples_per_frame, dtype=np.float32)
        
        # Handle different output formats
        if outdata.ndim > 1:
            # Stereo output
            for i in range(self.samples_per_frame):
                outdata[i, 0] = output_frame[i]  # Left channel
                if outdata.shape[1] > 1:
                    outdata[i, 1] = output_frame[i]  # Right channel (same as left)
        else:
            # Mono output
            for i in range(self.samples_per_frame):
                outdata[i] = output_frame[i]

    def get_last_voice_time_ms(self) -> float:
        """Get timestamp (ms) of the last time frame exceeded VAD threshold"""
        with self._vad_lock:
            return self._last_voice_time_ms
    
    def start(self):
        """Start audio stream with Windows-specific handling"""
        if self.is_running:
            return
        
        try:
            # Find best devices
            input_device, output_device = self._find_best_devices()
            
            # For duplex stream, device can be a pair (input_device, output_device)
            device = None
            if input_device is not None or output_device is not None:
                device = (input_device, output_device)
            
            # Windows-specific stream configuration
            stream_kwargs = {
                'samplerate': self.sample_rate,
                'blocksize': self.samples_per_frame,
                'channels': 1,  # Mono
                'dtype': np.float32,
                'device': device,
                'callback': self.audio_callback
            }
            
            # Add Windows-specific settings
            if platform.system() == "Windows":
                stream_kwargs.update({
                    'latency': 'low',  # Request low latency
                })
            
            self.stream = sd.Stream(**stream_kwargs)
            
            self.stream.start()
            self.is_running = True
            print(f"Windows audio stream started: {self.sample_rate}Hz, {self.frame_ms}ms frames")
            
        except Exception as e:
            print(f"Failed to start Windows audio stream: {e}")
            # Try fallback configuration
            try:
                print("Trying fallback audio configuration...")
                self.stream = sd.Stream(
                    samplerate=self.sample_rate,
                    blocksize=self.samples_per_frame,
                    channels=1,
                    dtype=np.float32,
                    callback=self.audio_callback
                )
                self.stream.start()
                self.is_running = True
                print("Fallback audio stream started ✓")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise
    
    def stop(self):
        """Stop audio stream"""
        if not self.is_running:
            return
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Warning: Error stopping stream: {e}")
            finally:
                self.stream = None
        
        self.is_running = False
        print("Windows audio stream stopped")
    
    def get_frame(self, timeout=0.1):
        """
        Get a frame from the ASR queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Frame as float32 array or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_size(self):
        """Get current queue size for debugging"""
        return self.frame_queue.qsize()
    
    def clear_queue(self):
        """Clear the frame queue"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break


def time_now_ms() -> float:
    """Monotonic-ish wall time in milliseconds (sufficient for VAD timing)."""
    return time.time() * 1000.0


def list_windows_audio_devices():
    """List all available Windows audio devices"""
    try:
        devices = sd.query_devices()
        print("\nAvailable Windows Audio Devices:")
        print("=" * 50)
        
        for i, device in enumerate(devices):
            device_type = []
            if device.get('max_inputs', 0) > 0:
                device_type.append("Input")
            if device.get('max_outputs', 0) > 0:
                device_type.append("Output")
            
            device_type_str = "/".join(device_type) if device_type else "Unknown"
            
            print(f"{i:2d}: {device.get('name', 'Unknown')} ({device_type_str})")
            print(f"     Sample rate: {device.get('default_samplerate', 'Unknown')}Hz")
            
            # Check if this is the default
            if i == sd.default.device[0]:
                print("     [DEFAULT INPUT]")
            if i == sd.default.device[1]:
                print("     [DEFAULT OUTPUT]")
            print()
            
    except Exception as e:
        print(f"Error listing devices: {e}")


if __name__ == "__main__":
    # Test device listing
    list_windows_audio_devices()
