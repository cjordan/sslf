#!/usr/bin/env python2

import copy

import numpy as np
import numpy.ma as ma
from scipy import signal


def find_background_rms(array, num_chunks=5, use_chunks=3):
    """
    Break the input array into evenly sized chunks, then find the three
    with the smallest RMS. Return the average of these as the true RMS.
    """
    chunks = np.array_split(array, num_chunks)
    np.seterr(under="warn")
    sorted_by_rms = sorted([np.std(x) for x in chunks])
    return np.mean(sorted_by_rms[:use_chunks])


def blank_spectrum_part(spectrum, point, radius, value=0):
    lower = max([0, point+int(-radius)])
    upper = min([len(spectrum), point+int(radius)])
    spectrum[lower:upper] = value


class Peak(object):
    def __init__(self, channel, snr, width):
        self.channel = channel
        self.snr = snr
        self.width = width


class Spectrum(object):
    def __init__(self, spectrum, vel=None):
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


    def find_cwt_peaks(self, scales=[], snr=6.5, min_space=5, wavelet=signal.ricker):
        """
        From the input spectrum (and a range of scales to search):
        - perform a CWT
        - 
        - return the list of peaks

        An SNR of 6.5 is a good compromise for reducing the number of false positives found
        while reliably finding real, significant peaks.

        It may be worthwhile to be in some smoothing for every element of cwt_mat.
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

            # If this maximum is not significant, we're done.
            if sig < snr:
                break
            # Otherwise, blank this line across all scales.
            else:
                for k in xrange(len(scales)):
                    # If the line is too close to the edge,
                    # cap the mask at the edge.
                    lower = max([0, peak_channel - 2*scales[k]])
                    upper = min([spectrum_length, peak_channel + 2*scales[k]])
                    cwt_mat[k, lower:upper] = ma.masked
                peaks.append(Peak(peak_channel, sig, scales[i]))

        self.channel_peaks = [p.channel for p in peaks]
        self.peak_snrs = [p.snr for p in peaks]
        self.peak_widths = [p.width for p in peaks]
        if self.vel is not None:
            self.vel_peaks = [self.vel[p.channel] for p in peaks]


    def vel_peaks2chan_peaks(self):
        """
        This function is useful for when you know the velocities of the spectral lines,
        and need to determine the relevant channels before subtracting the bandpass.
        """
        self.channel_peaks = []
        for vp in self.vel_peaks:
            self.channel_peaks.append(np.abs(self.vel-vp).argmin())


    def subtract_bandpass(self, window_length=151, poly_order=1, p0s=None, allowable_peak_gap=10):
        """
        """
        mask = np.zeros(len(self.original))

        for i, p in enumerate(self.channel_peaks):
            width = self.peak_widths[i] * 1.2

            ## Blank the lines, fitting the bandpass around them.
            blank_spectrum_part(mask, p, radius=width, value=1)

        self.filtered = copy.copy(self.original)

        # Interpolate between gaps in the spectrum.
        edges = np.where(np.diff(mask))[0]
        for i in xrange(len(edges)/2):
            e1, e2 = edges[2*i], edges[2*i+1]

            if e1 < allowable_peak_gap or e2 > len(self.original) - allowable_peak_gap:
                continue
            # Need a check for e2 being too close to the next e1.

            range_1 = np.arange(e1-allowable_peak_gap, e1)
            range_2 = np.arange(e2, e2+allowable_peak_gap)
            interp_range = np.concatenate((range_1, range_2))
            poly_fit = np.poly1d(np.polyfit(interp_range, self.filtered[interp_range], poly_order))
            self.filtered[e1:e2] = poly_fit(np.arange(e1, e2))

        self.bandpass = signal.savgol_filter(self.filtered, window_length=window_length, polyorder=poly_order)
        self.modified = self.original - self.bandpass
