"""
The main module of sslf (simple spectral line finder).
The provided Spectrum class is intended to facilitate all functionality.
"""

# Python 2 and 3 compatibility
from builtins import range
from future.utils import raise_with_traceback

import copy
import logging

import numpy as np
import numpy.ma as ma
from scipy import signal


logger = logging.getLogger(__name__)


def find_background_rms(array, num_chunks=5, use_chunks=3):
    """
    Break the input array into evenly sized chunks, then find the three
    with the smallest RMS. Return the average of these as the true RMS.
    """
    chunks = np.array_split(array, num_chunks)
    np.seterr(under="warn")
    sorted_by_rms = sorted([np.std(x) for x in chunks])
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("rms chunks = %s" % sorted_by_rms)
    mean = np.mean(sorted_by_rms[:use_chunks])
    if mean == 0:
        raise_with_traceback(ValueError("RMS is 0, which cannot be used."))
    return mean


def _blank_spectrum_part(spectrum, point, radius, value=0):
    lower = max([0, point+int(-radius)])
    upper = min([len(spectrum), point+int(radius)])
    spectrum[lower:upper] = value


class _Peak(object):
    def __init__(self, channel, snr, width):
        self.channel = channel
        self.snr = snr
        self.width = width


class Spectrum(object):
    def __init__(self, spectrum, vel=None):
        """
        Provide a spectrum to find lines on, and/or remove the bandpass from.
        The optional vel parameter essentially provides an "x-axis" to the
        data, such that peak positions can be determined in terms of this
        axis rather than merely the channel position in the spectrum.
        Note that any NaN values in the spectrum are filtered.
        """
        # Make sure the data are in numpy arrays.
        if isinstance(spectrum, list):
            spectrum = np.array(spectrum)
        if isinstance(vel, list):
            spectrum = np.array(vel)

        # Filter any NaNs, scipy doesn't like them.
        if np.any(np.isnan(spectrum)):
            nan_indices = np.isnan(spectrum)
            self.original = spectrum[~nan_indices]
            if vel is not None:
                self.vel = vel[~nan_indices]
            else:
                self.vel = None
        else:
            self.original = spectrum
            self.vel = vel

        self.rms = find_background_rms(spectrum)


    def find_cwt_peaks(self, scales=[], snr=6.5, wavelet=signal.ricker):
        """
        From the input spectrum (and a range of scales to search):
        - perform a CWT
        - find a significant peak in the CWT matrix
        - mask this peak in wavelet space, for all scales
        - loop from step 2, until no significant peaks remain
        - return the list of peaks

        In my experience, an SNR of 6.5 is a good compromise for reducing the number
        of false positives found while reliably finding real, significant peaks.

        It may be worthwhile to smooth the spectrum before performing the CWT.
        """

        assert len(scales) > 0, "No scales supplied!"

        peaks = []
        cwt_mat = signal.cwt(self.original, wavelet, scales)
        cwt_mat = ma.array(cwt_mat)
        spectrum_length = cwt_mat.shape[1]

        while True:
            peak_pixel = cwt_mat.argmax()
            i, peak_channel = np.unravel_index(peak_pixel, cwt_mat.shape)
            peak = cwt_mat[i, peak_channel]
            rms = find_background_rms(cwt_mat[i])
            sig = peak/rms
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Biggest peak at channel %s, scale %s" % (peak_channel, scales[i]))
            if logger.isEnabledFor(logging.NOTSET):
                logger.notset("rms = %s" % rms)

            # If this maximum is not significant, we're done.
            if sig < snr:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Peak is not significant (%s < %s), finishing" % (sig, snr))
                break
            # Otherwise, blank this line across all scales.
            else:
                for k in range(len(scales)):
                    # If the line is too close to the edge,
                    # cap the mask at the edge.
                    lower = max([0, peak_channel - 2*scales[k]])
                    upper = min([spectrum_length, peak_channel + 2*scales[k]])
                    if logger.isEnabledFor(logging.NOTSET):
                        logger.notset("lower = %s, upper = %s" % (lower, upper))
                    cwt_mat[k, lower:upper] = ma.masked
                peaks.append(_Peak(peak_channel, sig, scales[i]))

        self.channel_peaks = [p.channel for p in peaks]
        self.peak_snrs = [p.snr for p in peaks]
        self.peak_widths = [p.width for p in peaks]
        if self.vel is not None:
            self.vel_peaks = [self.vel[p.channel] for p in peaks]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Channel peaks: %s" % self.channel_peaks)
            logger.debug("Peak SNRs: %s" % self.peak_snrs)
            logger.debug("Peak widths: %s" % self.peak_widths)
            if self.vel is not None:
                logger.debug("Velocity peaks: %s" % self.vel_peaks)


    def vel_peaks2chan_peaks(self):
        """
        This function is useful for when you know the velocities of the spectral lines,
        and need to determine the relevant channels before subtracting the bandpass.
        """
        self.channel_peaks = []
        for vp in self.vel_peaks:
            self.channel_peaks.append(np.abs(self.vel-vp).argmin())
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Channel peaks: %s" % self.channel_peaks)


    def subtract_bandpass(self, window_length=151, poly_order=1, blank_spectrum_width=1.4, allowable_peak_gap=10):
        """
        Once we have the locations of the lines, flag them, and subtract the non-zero
        bandpass everywhere else. Provide the flattened spectrum in self.modified.

        window_length and poly_order feed directly into scipy's savgol_filter, which
        forms the bulk of the work in this function. See that function's documentation
        for more information.

        window_length is probably the most important parameter, and will need to be
        tuned for different spectra. In a sense, it specifies how far to look ahead
        and behind every channel when considering bandpass shape. If this is too small,
        it will behave more like a low-pass filter than a high-pass filter, when we
        want it to be more on the high-pass side.

        poly_order...

        allowable_peak_gap...

        blank_spectrum_width...
        """
        mask = np.zeros(len(self.original))

        for i, p in enumerate(self.channel_peaks):
            width = self.peak_widths[i] * blank_spectrum_width

            # Blank the lines, fitting the bandpass around them.
            _blank_spectrum_part(mask, p, radius=width, value=1)

        self.filtered = copy.copy(self.original)

        # Interpolate between gaps in the spectrum.
        edges = np.where(np.diff(mask))[0]
        for i in range(len(edges)//2):
            e1, e2 = edges[2*i], edges[2*i+1]
            if logger.isEnabledFor(logging.NOTSET):
                logger.notset("Interpolation edges: %s, %s" % (e1, e2))

            if e1 < allowable_peak_gap or e2 > len(self.original) - allowable_peak_gap:
                if logger.isEnabledFor(logging.NOTSET):
                    logger.notset("Interpolation edges are too close")
                continue
            # Need a check for e2 being too close to the next e1.

            range_1 = np.arange(e1-allowable_peak_gap, e1)
            range_2 = np.arange(e2, e2+allowable_peak_gap)
            interp_range = np.concatenate((range_1, range_2))
            poly_fit = np.poly1d(np.polyfit(interp_range, self.filtered[interp_range], poly_order))
            self.filtered[e1:e2] = poly_fit(np.arange(e1, e2))

        self.bandpass = signal.savgol_filter(self.filtered, window_length=window_length, polyorder=poly_order)
        self.modified = self.original - self.bandpass
