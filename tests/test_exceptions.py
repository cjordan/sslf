#!/usr/bin/env python

import logging
import pytest
import numpy as np

from sslf.sslf import Spectrum


def test_rms_0():
    with pytest.raises(ValueError):
        Spectrum(np.zeros(100))


def test_lowest_logging_level():
    logging.basicConfig(level=logging.DEBUG)
    s = Spectrum(np.random.normal(size=1000))
    s.find_cwt_peaks(scales=np.arange(1, 20), snr=5)
    s.subtract_bandpass()


if __name__ == "__main__":
    # Introspect and run all the functions starting with "test".
    for f in dir():
        if f.startswith("test"):
            print(f)
            exec(f+"()")
