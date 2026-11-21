"""Audio preprocessing for HearMe AI."""
import numpy as np
import librosa

TARGET_SR = 16000
DURATION = 5.0
TARGET_LEN = int(TARGET_SR * DURATION)  # 80,000 samples


def preprocess_audio(y, sr):
    """Convert any audio into a standard format:
    mono, 16 kHz, exactly 5 seconds, peak-normalized to [-1, 1]."""
    # 1. Mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # 2. Resample
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # 3. Fixed length
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]

    # 4. Peak normalize
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y.astype(np.float32)
