# MVP Speech Monitoring Prototype for macOS

A simple, reliable system that listens to speech, predicts upcoming words, and triggers delayed auditory feedback (DAF) when risky content is expected.

## Features

- **Real-time Speech Recognition**: Uses Vosk for streaming ASR
- **Predictive Analysis**: Uses Llama 3.2 1B for next-token prediction
- **Keyword Detection**: Aho-Corasick automaton for efficient keyword matching
- **Delayed Auditory Feedback**: 175ms delay with configurable hold time
- **Minimal and Robust**: Designed for macOS with minimal dependencies

## Requirements

- macOS with Python 3.11
- Microphone and headphones (to prevent acoustic feedback)
- Microphone access granted to terminal/IDE

## Installation

1. **Clone and setup environment**:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install numpy==1.26.4 sounddevice==0.4.7 soundfile==0.12.1 vosk==0.3.44 pyahocorasick==2.1.0 llama-cpp-python==0.2.90 onnxruntime==1.18.1 pyyaml==6.0.2 loguru==0.7.2
   ```

2. **Download required models**:
   - **Vosk ASR Model**: Download `vosk-model-small-en-us` from [Vosk Models](https://alphacephei.com/vosk/models)
     - Extract to `models/asr/vosk-model-small-en-us/`
   - **Llama Model**: Download `llama-3.2-1b-q4_k_m.gguf` from [Hugging Face](https://huggingface.co/TheBloke/Llama-3.2-1B-GGUF)
     - Place in `models/llm/llama-3.2-1b-q4_k_m.gguf`

3. **Grant microphone permissions**:
   - Go to System Settings > Privacy & Security > Microphone
   - Enable access for Terminal or your IDE

## Project Structure

```
project/
├── config/
│   ├── audio.yml          # Audio configuration
│   └── keywords.yml       # Keyword stems and LLM parameters
├── models/
│   ├── asr/
│   │   └── vosk-model-small-en-us/  # Vosk ASR model
│   └── llm/
│       └── llama-3.2-1b-q4_k_m.gguf # Llama LLM model
├── src/
│   ├── main.py            # Main orchestrator
│   ├── daf_ring.py        # DAF ring buffer
│   ├── audio_io.py        # Audio I/O with duplex stream
│   ├── asr_vosk.py        # Vosk ASR integration
│   ├── predictor_llamacpp.py # Llama LLM predictor
│   ├── detector_keywords.py  # Keyword detector
│   ├── state.py           # DAF state management
│   └── utils.py           # Logging utilities
├── test_daf.py            # DAF and audio test script
└── README.md
```

## Configuration

### Audio Configuration (`config/audio.yml`)

```yaml
sample_rate: 16000          # Audio sample rate
frame_ms: 20               # Frame duration
daf_delay_ms: 175          # DAF delay in milliseconds
hold_ms: 600               # Hold duration after trigger
consecutive_hits: 2        # Required consecutive hits to trigger
input_gain: 1.0            # Input gain multiplier
output_gain: 0.9           # Output gain multiplier
limiter_ceiling_db: -3.0   # Hard limiter ceiling
```

### Keywords Configuration (`config/keywords.yml`)

```yaml
stems:                     # Keyword stems to detect
  - passw
  - credit card
  - secret
  - confid
  - ssn
  - social secur
  - bank rout
  - pin number
min_pred_tokens: 3         # Minimum LLM prediction tokens
max_pred_tokens: 5         # Maximum LLM prediction tokens
top_k: 20                  # LLM top-k sampling
context_tokens: 96         # LLM context window
require_two_sources: true  # Require both ASR and LLM hits
```

## Usage

### Testing Audio Setup

Before running the full system, test the audio components:

```bash
source .venv/bin/activate
python test_daf.py
```

This will test the DAF ring buffer and optionally test audio I/O with DAF activation.

### Running the Full System

```bash
source .venv/bin/activate
python src/main.py
```

The system will:
1. Initialize all components
2. Start audio I/O with duplex stream
3. Begin ASR processing in a worker thread
4. Run LLM predictions when partial text grows
5. Monitor for keyword matches in both ASR and LLM output
6. Trigger DAF when consecutive hits are detected
7. Run for 2 minutes by default

## How It Works

1. **Audio Processing**: 20ms frames at 16kHz are processed through a duplex stream
2. **ASR**: Vosk processes audio frames and produces partial text results
3. **LLM Prediction**: When partial text grows by 4+ characters, Llama predicts next tokens
4. **Keyword Detection**: Aho-Corasick automaton scans both ASR partials and LLM predictions
5. **Trigger Logic**: DAF activates when 2 consecutive hits are detected
6. **DAF**: 175ms delayed audio feedback with 600ms hold time

## Troubleshooting

### Audio Issues
- **Feedback**: Use headphones and reduce `output_gain` to 0.7
- **Clicks**: Ensure no allocations in audio callback
- **Device Issues**: Check `sounddevice.query_devices()` and set device indices in config

### Performance Issues
- **Slow LLM**: Reduce `context_tokens` to 64 and `max_pred_tokens` to 3
- **ASR Stalls**: Verify 16kHz sample rate and 320-sample frames
- **High Latency**: Reduce LLM call frequency by increasing partial increment threshold

### Model Issues
- **Vosk Model**: Ensure `vosk-model-small-en-us` is extracted to correct directory
- **Llama Model**: Verify `llama-3.2-1b-q4_k_m.gguf` is in `models/llm/`

## Acceptance Tests

1. **Neutral Speech**: 60 seconds of normal conversation should produce zero unintended DAF activations
2. **Risky Phrases**: Saying "my password is", "my secret is", etc. should trigger DAF within 200-400ms
3. **Rapid Speech**: 30 seconds of fast speech should not produce clicks or crackles

## Limitations

- **macOS Only**: Designed specifically for macOS audio stack
- **Single Speaker**: Optimized for single speaker scenarios
- **English Only**: Uses English ASR and LLM models
- **Limited Context**: 96-token context window for LLM predictions

## Development

The system is designed to be minimal and robust. Key design principles:

- **Light Audio Callback**: No heavy processing in audio thread
- **Worker Threads**: ASR processing in separate thread
- **Pre-allocated Buffers**: Avoid allocations in real-time paths
- **Configurable Parameters**: Easy tuning without code changes
- **Comprehensive Logging**: Detailed logging for debugging

## License

This is an MVP prototype for research purposes.
