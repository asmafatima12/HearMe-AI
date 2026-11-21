"""MFCC feature extraction for HearMe AI."""
import numpy as np
import librosa
from audio_processing import TARGET_SR

N_MFCC = 20
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64


def mfcc_features(y_std):
    """Turn a standardized 5-second clip into a 40-number vector
    (20 MFCC means + 20 MFCC standard deviations)."""
    mfcc = librosa.feature.mfcc(
        y=y_std, sr=TARGET_SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
