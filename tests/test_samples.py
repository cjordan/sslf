#!/usr/bin/env python

import os
import numpy as np

from sslf.sslf import Spectrum


TEST_DIR = '/'.join(os.path.realpath(__file__).split('/')[0:-1])


def test_standard_spectrum():
    # Standard operation.
    with open(TEST_DIR + "/spectrum_4.txt", 'r') as f:
        x, y = np.loadtxt(f, unpack=True)
    s = Spectrum(y, vel=x)
    s.find_cwt_peaks(scales=np.arange(1, 50, 2), snr=5)
    s.subtract_bandpass()
    assert abs(s.vel_peaks[0] + 54) < 1


def test_list_inputs():
    # Lists for inputs.
    with open(TEST_DIR + "/spectrum_5.txt", 'r') as f:
        x, y = np.loadtxt(f, unpack=True)
    s = Spectrum(list(y), vel=list(x))
    s.find_cwt_peaks(scales=range(1, 50, 2), snr=5)
    s.subtract_bandpass()
    assert abs(s.vel_peaks[0] + 51.5) < 1


def test_nan_handling():
    # Replace a couple of values with NaN.
    with open(TEST_DIR + "/spectrum_1.txt", 'r') as f:
        x, y = np.loadtxt(f, unpack=True)
    y[100] = np.nan
    y[500] = np.nan
    s = Spectrum(y, vel=x)
    s.find_cwt_peaks(scales=np.arange(1, 50, 2), snr=5)
    s.subtract_bandpass()
    assert abs(s.vel_peaks[0] + 41.8e3) < 0.5e3

    # Handle another case.
    with open(TEST_DIR + "/spectrum_2.txt", 'r') as f:
        _, y = np.loadtxt(f, unpack=True)
    y[200] = np.nan
    y[300] = np.nan
    s = Spectrum(y)
    s.find_cwt_peaks(scales=np.arange(1, 20, 2), snr=5)
    s.subtract_bandpass()
    assert abs(s.channel_peaks[0] - 289) < 5


def test_vel_peaks2chan_peaks():
    with open(TEST_DIR + "/spectrum_3.txt", 'r') as f:
        x, y = np.loadtxt(f, unpack=True)
    s = Spectrum(y, vel=x)
    s.find_cwt_peaks(scales=np.arange(1, 20, 2), snr=5)
    sslf_channel_peaks = s.channel_peaks

    # Supply the velocity peaks manually and compare.
    s.vel_peaks = [-53.4, -59.8]
    s.vel_peaks2chan_peaks()

    assert np.all(np.isclose(sslf_channel_peaks, s.channel_peaks, atol=8))


if __name__ == "__main__":
    # Introspect and run all the functions starting with "test".
    for f in dir():
        if f.startswith("test"):
            print(f)
            exec(f+"()")
