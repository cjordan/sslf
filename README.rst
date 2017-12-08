sslf
====
.. image:: https://travis-ci.org/cjordan/sslf.svg?branch=master
   :target: https://travis-ci.org/cjordan/sslf
.. image:: https://coveralls.io/repos/github/cjordan/sslf/badge.svg?branch=master
   :target: https://coveralls.io/github/cjordan/sslf?branch=master
.. image:: https://api.codacy.com/project/badge/Grade/84eaced156f94ccfa9e18ee4c1a28c85
   :target: https://www.codacy.com/app/cjordan/sslf?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cjordan/sslf&amp;utm_campaign=Badge_Grade

Simple Spectral Line Finder

``sslf`` is designed to be a dead-simple, effective and useful spectral line finder for 1D data. It utilises the continuous wavelet transform from scipy, which is a productive way to find even weak spectral lines.

Because there are many definitions for a spectral line, this package can not conceivably work in all cases. Here, I have designed ``sslf`` for Gaussian-profiled spectral lines. A big part of my work focuses on recovering weak signals in the noise, so ``sslf`` aims to identify lines, flag them, and subtract the non-zero bandpass around them.

Usage
-----
With a 1D spectrum (``spectral_data``) and a range of scales to test (e.g. 1 through 20), statistically significant spectral peaks can be found with::

    import numpy as np
    from sslf.sslf import Spectrum

    s = Spectrum(spectral_data)
    s.find_cwt_peaks(scales=np.arange(1, 20), snr=6.5)
    s.subtract_bandpass()

The flattened spectrum is then contained in ``s.modified``, and peak locations at ``s.channel_peaks``.

``find_cwt_peaks`` can optionally take the wavelet to be used in the wavelet transformation (Ricker wavelet by default). ``subtract_bandpass`` is a little harder to explain; see more complicated examples of usage can be found in the ``notebooks`` directory.

Limitations / Future work
-------------------------
I don't know how to optimally select a wavelet for a particular spectral line. The Ricker wavelet appears to work well for Gaussian-profiled lines, though. I don't know how well this extends to something like a Lorentzian, but you can spectify the wavelet to ``find_cwt_peaks``. I believe the wavelet transformation is the most expensive part of this software. Rather than using ``scipy``, ``pyWavelets`` may be faster, but I need time to try it out.

Contributions and Suggestions
-----------------------------
I created this software because I could not find anything in the community that functioned similarly. So, I would welcome contributions to make this software more generally flexible, to be suitable for a wide audience. It's really simple, just look at the code!

Bugs? Inconsistencies?
----------------------
Absolutely! If you find something odd, let me know and I'll attempt to fix it ASAP.

Contact
-------
christopherjordan87 -at- gmail.com

Installation
------------
::

    pip install sslf

or

- clone this repo (``git clone https://github.com/cjordan/sslf.git``)
- within the directory, run: ``python setup.py install --optimize=1``

Dependencies
------------
- python 2.7.x or 3.x
- numpy 1.8+
- scipy
- future
