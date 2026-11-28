"""YAMNet embedding features for HearMe AI.
Recipe (verified against training data):
    2048 features = [mean over windows (1024), max over windows (1024)]
"""
import numpy as np
import tensorflow_hub as hub
from audio_processing import preprocess_audio

_yamnet = None

def load_yamnet():
    global _yamnet
    if _yamnet is None:
        _yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    return _yamnet

def yamnet_features(y, sr):
    """Raw audio (any rate) -> 2048-number feature vector."""
    wav = preprocess_audio(y, sr)
    _, emb, _ = load_yamnet()(wav)
    emb = emb.numpy()
    return np.concatenate([emb.mean(axis=0), emb.max(axis=0)])
