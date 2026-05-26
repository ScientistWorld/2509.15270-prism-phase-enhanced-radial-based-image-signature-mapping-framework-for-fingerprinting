"""PRISM radial DFT feature extraction.

Implements the paper's core feature computation: per-channel FFT shift, log
magnitude, phase, radial annular averaging, and cosine-transformed circular
mean phase. With 64 radial bins and RGB input this yields 384 features/image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _radial_bins(height: int, width: int, n_bins: int) -> np.ndarray:
    y, x = np.indices((height, width))
    center_y = height / 2.0
    center_x = width / 2.0
    radii = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    edges = np.linspace(0.0, radii.max() + 1e-9, n_bins + 1)
    bins = np.digitize(radii.ravel(), edges, right=False) - 1
    return np.clip(bins, 0, n_bins - 1)


def _channel_features(channel: np.ndarray, bins: np.ndarray, n_bins: int) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(channel.astype(np.float32)))
    log_magnitude = np.log(np.abs(spectrum) + 1.0).ravel()
    phase = np.angle(spectrum).ravel()

    counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
    mag_sum = np.bincount(bins, weights=log_magnitude, minlength=n_bins)
    mag_profile = mag_sum / np.maximum(counts, 1.0)

    complex_phase = np.exp(1j * phase)
    real_sum = np.bincount(bins, weights=complex_phase.real, minlength=n_bins)
    imag_sum = np.bincount(bins, weights=complex_phase.imag, minlength=n_bins)
    circular_mean_angle = np.angle((real_sum + 1j * imag_sum) / np.maximum(counts, 1.0))
    phase_profile = np.cos(circular_mean_angle)

    return np.concatenate([mag_profile, phase_profile]).astype(np.float32)


def extract_prism_features(image: Image.Image, n_bins: int = 64) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    height, width, _ = rgb.shape
    bins = _radial_bins(height, width, n_bins)
    channel_vectors = [
        _channel_features(rgb[:, :, channel], bins, n_bins) for channel in range(3)
    ]
    return np.concatenate(channel_vectors).astype(np.float32)
