#!/usr/bin/env python3
"""
Stress test Vosk recognizer access under concurrent feed/reads/resets.

This simulates two threads:
  - feeder: feeds 20 ms silent frames at ~50 FPS (AcceptWaveform)
  - reader: reads partials frequently and triggers periodic resets

Goal: catch thread-safety issues that could trigger Kaldi lattice assertions.

Run:
  source .venv/bin/activate
  python test_vosk_stress.py
"""

import sys
import time
import threading
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from asr_vosk import ASRVosk


def main(duration_s: float = 5.0):
    asr = ASRVosk(model_path='models/asr/vosk-model-small-en-us', sample_rate=16000)

    stop = threading.Event()

    def feeder():
        frame = np.zeros(320, dtype=np.float32)  # 20 ms @ 16 kHz
        while not stop.is_set():
            pcm = asr.convert_float32_to_int16_bytes(frame)
            asr.accept(pcm)
            time.sleep(0.02)

    def reader():
        i = 0
        while not stop.is_set():
            _ = asr.get_partial_if_new()
            i += 1
            if i % 50 == 0:
                asr.reset()
            time.sleep(0.01)

    threads = [threading.Thread(target=feeder, daemon=True), threading.Thread(target=reader, daemon=True)]
    for t in threads:
        t.start()

    print(f"Running stress test for ~{duration_s} seconds...")
    try:
        time.sleep(duration_s)
        print("Done, stopping...")
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=1.0)
    print("Completed without crash.")


if __name__ == '__main__':
    exit(0 if main() is None else 0)

