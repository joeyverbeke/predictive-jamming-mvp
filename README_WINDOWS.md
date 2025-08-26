# Predictive Jamming System - Windows Edition

A Windows-optimized speech monitoring system that listens to speech, predicts upcoming words, and triggers delayed auditory feedback (DAF) when risky content is expected. Optimized for Windows with GPU acceleration support for RTX 3060.

## Features

- **Real-time Speech Recognition**: Uses Vosk for streaming ASR
- **GPU-Accelerated LLM**: Uses Llama 3.2 1B with CUDA support for RTX 3060
- **Keyword Detection**: Aho-Corasick automaton for efficient keyword matching
- **Delayed Auditory Feedback**: 175ms delay with configurable hold time
- **Windows Audio Optimization**: WASAPI backend with low-latency audio
- **Bluetooth Device Support**: Automatic latency compensation for wireless devices

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.8 or higher
- **GPU**: NVIDIA RTX 3060 (or compatible CUDA GPU)
- **Audio**: Microphone and headphones/speakers
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for models

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd predictive-jamming-mvp

# Run Windows setup script with virtual environment
python setup_windows_venv.py
```

The setup script will:
- Create a virtual environment (.venv)
- Install all required dependencies in the virtual environment
- Configure GPU acceleration for RTX 3060
- Create Windows-specific configuration files
- Set up necessary directories
- Create activation scripts

### 2. Download Models

You need to download two models:

#### ASR Model (Vosk)
- **Download**: [Vosk Models](https://alphacephei.com/vosk/models)
- **File**: `vosk-model-small-en-us`
- **Extract to**: `models/asr/vosk-model-small-en-us/`

#### LLM Model (Llama)
- **Download**: [Hugging Face](https://huggingface.co/TheBloke/Llama-3.2-1B-GGUF)
- **File**: `llama-3.2-1b-q4_k_m.gguf`
- **Place in**: `models/llm/llama-3.2-1b-q4_k_m.gguf`

### 3. Activate Virtual Environment and Test

```bash
# Activate the virtual environment
activate_venv.bat

# Test audio and GPU components
python test_audio_windows.py

# Run the full system
python src/main_windows.py
```

**Important**: Always activate the virtual environment before running the system!

## Windows-Specific Features

### GPU Acceleration
The system automatically detects and uses your RTX 3060 for LLM inference:
- 32 layers offloaded to GPU for optimal performance
- Automatic fallback to CPU if CUDA unavailable
- Performance monitoring and benchmarking

### Audio Optimization
- **WASAPI Backend**: Low-latency Windows audio
- **Device Auto-Detection**: Automatically finds best input/output devices
- **Bluetooth Compensation**: Adjusts DAF delay for wireless devices
- **Fallback Support**: Multiple audio backends for compatibility

### Configuration
Windows-specific configuration in `config/audio_windows.yml`:
```yaml
sample_rate: 16000
frame_ms: 20
daf_delay_ms: 175
hold_ms: 600
consecutive_hits: 2
input_device: null   # Auto-detect
output_device: null  # Auto-detect
input_gain: 1.0
output_gain: 0.9
limiter_ceiling_db: -3.0
vad_enabled: true
vad_rms_threshold: 0.015
vad_silence_ms: 1000
daf_latch_until_silence: true
windows_audio_backend: 'wasapi'  # Windows-specific
gpu_acceleration: true           # Enable GPU for RTX 3060
```

## Troubleshooting

### Audio Issues

**Problem**: No audio input/output
```bash
# Activate virtual environment first
activate_venv.bat

# Check audio devices
python -c "from src.audio_io_windows import list_windows_audio_devices; list_windows_audio_devices()"

# Test audio components
python test_audio_windows.py
```

**Problem**: Audio feedback/echo
- Use headphones instead of speakers
- Reduce `output_gain` to 0.7 in config
- Check microphone positioning

**Problem**: Audio clicks/pops
- Ensure no other applications are using audio
- Try different `windows_audio_backend` (wasapi, directsound, mme)
- Increase `frame_ms` to 30ms for stability

### GPU Issues

**Problem**: CUDA not detected
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA Toolkit if needed
# Download from: https://developer.nvidia.com/cuda-downloads
```

**Problem**: LLM runs slowly
- Verify GPU acceleration is enabled
- Check GPU memory usage
- Reduce `gpu_layers` to 16 if needed

### Model Issues

**Problem**: Models not found
```bash
# Activate virtual environment first
activate_venv.bat

# Check model paths
ls models/asr/
ls models/llm/

# Download missing models (see Quick Start section)
```

**Problem**: LLM prediction errors
- Verify model file integrity
- Check available GPU memory
- Try CPU-only mode by setting `gpu_acceleration: false`

## Performance Optimization

### For RTX 3060
- **GPU Layers**: 32 (optimal for 12GB VRAM)
- **Batch Size**: 512 (optimized for RTX 3060)
- **Context Tokens**: 96 (balanced performance/accuracy)

### Audio Latency
- **Target**: 175ms DAF delay
- **Achievable**: 150-200ms with good hardware
- **Bluetooth**: +50-150ms compensation applied automatically

### System Resources
- **CPU**: 2-4 cores for ASR processing
- **GPU**: 4-8GB VRAM for LLM
- **RAM**: 4-8GB for system operation

## Advanced Configuration

### Custom Keywords
Edit `config/keywords.yml`:
```yaml
stems:
  - passw
  - credit card
  - secret
  - confid
  # Add your custom keywords here
```

### Performance Tuning
For different hardware configurations:

**High-End (RTX 4090)**
```yaml
gpu_layers: 48
n_batch: 1024
context_tokens: 128
```

**Mid-Range (RTX 3060)**
```yaml
gpu_layers: 32
n_batch: 512
context_tokens: 96
```

**Low-End (CPU only)**
```yaml
gpu_acceleration: false
context_tokens: 64
max_pred_tokens: 3
```

## Development

### Project Structure
```
project/
├── config/
│   ├── audio_windows.yml    # Windows audio config
│   └── keywords.yml         # Keyword configuration
├── models/
│   ├── asr/                 # Vosk ASR models
│   └── llm/                 # Llama LLM models
├── src/
│   ├── main_windows.py      # Windows main system
│   ├── audio_io_windows.py  # Windows audio I/O
│   ├── predictor_llamacpp_windows.py # GPU-accelerated LLM
│   └── ...                  # Other components
├── setup_windows.py         # Windows setup script
├── test_audio_windows.py    # Windows audio tests
└── README_WINDOWS.md        # This file
```

### Testing
```bash
# Activate virtual environment first
activate_venv.bat

# Run all tests
python test_audio_windows.py

# Test specific components
python src/predictor_llamacpp_windows.py  # GPU test
python src/audio_io_windows.py            # Audio device test
```

### Debugging
Enable debug mode in `src/main_windows.py`:
```python
self.debug_mode = True  # Set to True for verbose logging
```

## License

This is an MVP prototype for research purposes.

## Support

For Windows-specific issues:
1. Check the troubleshooting section above
2. Activate virtual environment: `activate_venv.bat`
3. Run `python test_audio_windows.py` for diagnostics
4. Verify CUDA installation with `nvidia-smi`
5. Check Windows audio device settings
