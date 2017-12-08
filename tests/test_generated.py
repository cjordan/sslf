#!/usr/bin/env python

import os
import numpy as np
from scipy.signal import gaussian

from sslf.sslf import Spectrum


TEST_DIR = '/'.join(os.path.realpath(__file__).split('/')[0:-1])


def generate_spectrum(channels=1000, noise_std=1,
                      signal_pos=0, signal_intensity=2, signal_width=20):
    spectrum = np.random.normal(scale=noise_std, size=channels)
    signal = signal_intensity * gaussian(channels, signal_width)
    spectrum += np.roll(signal, signal_pos)
    return spectrum


def test_simple_spectrum():
    np.random.seed(1000)

    spectrum = generate_spectrum()
    s = Spectrum(spectrum)
    s.find_cwt_peaks(scales=np.arange(1, 40, 2), snr=5)
    assert s.channel_peaks == [504]


def test_pathological_edge_case():
    np.random.seed(1000)

    spectrum = generate_spectrum(signal_pos=500)
    s = Spectrum(spectrum)
    s.find_cwt_peaks(scales=np.arange(1, 40, 2), snr=5)
    s.subtract_bandpass()
    assert len(s.channel_peaks) == 2
    assert np.all(np.isclose([987, 15], s.channel_peaks, atol=3))


def test_pathological_edge_case2():
    np.random.seed(1000)

    spectrum = generate_spectrum(signal_intensity=10,
                                 signal_pos=450)
    s = Spectrum(spectrum)
    s.find_cwt_peaks(scales=np.arange(1, 40, 2), snr=5)
    s.subtract_bandpass()
    assert len(s.channel_peaks) == 1
    assert np.all(np.isclose([950], s.channel_peaks, atol=3))


def test_pathological_edge_ripple_case():
    np.random.seed(1000)

    spectrum = generate_spectrum(signal_pos=500,
                                 noise_std=0.1)
    spectrum += 0.22 * np.sin(np.linspace(0, 50, 1000))
    s = Spectrum(spectrum)
    s.find_cwt_peaks(scales=np.arange(20, 40, 2), snr=5)
    s.subtract_bandpass(window_length=49)
    assert len(s.channel_peaks) == 2
    assert np.all(np.isclose([15, 988], s.channel_peaks, atol=3))


def test_pathological_edge_ripple_case2():
    np.random.seed(1000)

    spectrum = generate_spectrum(signal_pos=450,
                                 noise_std=0.1)
    spectrum += 0.35 * np.sin(np.linspace(0, 50, 1000))
    s = Spectrum(spectrum)
    s.find_cwt_peaks(scales=np.arange(20, 40, 2), snr=5)
    s.subtract_bandpass(window_length=49)
    assert len(s.channel_peaks) == 1
    assert np.all(np.isclose([942], s.channel_peaks, atol=3))


if __name__ == "__main__":
    # Introspect and run all the functions starting with "test".
    for f in dir():
        if f.startswith("test"):
            print(f)
            exec(f+"()")
