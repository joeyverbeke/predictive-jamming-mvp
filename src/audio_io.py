import numpy as np
import sounddevice as sd
import queue
import threading
from typing import Optional


class AudioIO:
    """Audio input/output with duplex stream"""
    
    def __init__(self, sample_rate=16000, frame_ms=20, input_gain=1.0, 
                 input_device=None, output_device=None):
        """
        Initialize audio I/O
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_ms: Frame duration in milliseconds
            input_gain: Input gain multiplier
            input_device: Input device index (None for default)
            output_device: Output device index (None for default)
        """
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.samples_per_frame = int(frame_ms * sample_rate / 1000)
        self.input_gain = input_gain
        self.input_device = input_device
        self.output_device = output_device
        
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
    
    def set_daf_ring(self, daf_ring):
        """Set DAF ring buffer reference"""
        self.daf_ring = daf_ring
    
    def audio_callback(self, indata, outdata, frames, time, status):
        """Light audio callback for real-time processing"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Read input frame and apply gain
        input_frame = indata[:, 0].astype(np.float32)  # Mono input
        for i in range(self.samples_per_frame):
            self.temp_input[i] = input_frame[i] * self.input_gain
        
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
        
        # Write to output
        for i in range(self.samples_per_frame):
            outdata[i, 0] = output_frame[i]  # Mono output
    
    def start(self):
        """Start audio stream"""
        if self.is_running:
            return
        
        try:
            # For duplex stream, device can be a pair (input_device, output_device)
            device = None
            if self.input_device is not None or self.output_device is not None:
                device = (self.input_device, self.output_device)
            
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.samples_per_frame,
                channels=1,  # Mono
                dtype=np.float32,
                device=device,
                callback=self.audio_callback
            )
            
            self.stream.start()
            self.is_running = True
            print(f"Audio stream started: {self.sample_rate}Hz, {self.frame_ms}ms frames")
            
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            raise
    
    def stop(self):
        """Stop audio stream"""
        if not self.is_running:
            return
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.is_running = False
        print("Audio stream stopped")
    
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
