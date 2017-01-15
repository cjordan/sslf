#!/usr/bin/env python2

import copy

import numpy as np
from scipy import optimize, signal


def find_nearest(np_array, value):
    return np.abs(np_array - value).argmin()


def find_background_rms(array, num_chunks=5, use_chunks=3):
    """
    Break the input array into evenly sized chunks, then find the three
    with the smallest RMS. Return the average of these as the true RMS.
    """
    chunks = np.array_split(array, num_chunks)
    sorted_by_rms = sorted([np.std(x) for x in chunks])
    return np.mean(sorted_by_rms[:use_chunks])


def gaussian(x, a, b, c):
    """
    The general form of a Gaussian (without a DC term),
    for curve-fitting purposes.
    """
    return a * np.exp(-(x - b)**2 / (2 * c**2))


def mexican_hat(x, a, b, c):
    """
    A convenience function to fit the Ricker wavelet ("mexican hat") shape
    independent of scipy's built-in wavelet.
    """
    hat = 2/(np.sqrt(3*c) * np.pi**(1/4.)) * (1 - (x-b)**2/c**2) * np.exp(-(x-b)**2/(2*c**2))
    return a * hat/max(hat)


def find_cwt_peak(cwt_spec, scale, snr=6.5, rms=None):
    """
    From a wavelet-domain spectrum (and an expected scale), find a peak
    and return its fitted parameters.
    """
    y1 = max(cwt_spec)
    if rms is None:
        rms = find_background_rms(cwt_spec)

    # Is this maximum significant?
    if y1/rms > snr:
        x = np.arange(0, len(cwt_spec))
        x1 = find_nearest(cwt_spec, y1)
        popt, _ = optimize.curve_fit(mexican_hat, x, cwt_spec, p0=(y1, x1, 3*scale),
                                     bounds=([0.8*y1, x1-5, scale], [1.2*y1, x1+5, 4*scale]))
        return popt, y1/rms

    return None, None


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


    def find_cwt_peaks(self, scales=[], snr=6.5, min_space=30, wavelet=signal.ricker):
        """
        From the input spectrum (and a range of scales to search):
        - perform a CWT
        - for each specified scale:
        -- find a significant peak
        -- subtract its contribution in the wavelet domain, continue searching for peaks
        - return the list of peaks

        An SNR of 6.5 is a good compromise for reducing the number of false positives found
        while reliably finding real, significant peaks.

        May be worthwhile to be in some smoothing for every element of cwt_mat.
        """

        assert len(scales) > 0, "No scales supplied!"

        peaks, snrs, fits = [], [], []
        cwt_mat = signal.cwt(self.original, wavelet, scales)
        x = np.arange(cwt_mat.shape[1])

        for i, s in enumerate(scales):
            y = cwt_mat[i]
            rms = find_background_rms(y)

            # Fit any existing peaks.
            for p in peaks:
                if not y[p] < 0 and np.mean(y[p-10:p+10])/rms > 3:
                    try:
                        popt, _ = optimize.curve_fit(mexican_hat, x, y, p0=(y[p], p, s),
                                                     bounds=([0.8*y[p], p-10, s], [1.2*y[p], p+10, 4*s]))
                        if np.abs(popt[1]-p) < 5:
                            y -= mexican_hat(x, *popt)
                    except RuntimeError:
                        continue

            while True:
                # Find peaks.
                # There should be checks on the returned parameters,
                # e.g. against the current i (scale length)
                popt, sig = find_cwt_peak(y, s, snr=snr, rms=rms)
                if popt is not None:
                    y -= mexican_hat(x, *popt)
                else:
                    break

                # Check if this peak has already been found.
                if not np.any(np.isclose([popt[1]], peaks, atol=min_space, rtol=0)):
                    # Round the peak positions to integers.
                    peaks.append(np.rint(popt[1]).astype(int))
                    snrs.append(sig)
                    fits.append(popt)

        self.channel_peaks = peaks
        self.peak_snrs = snrs
        self.peak_fits = fits
        if self.vel is not None:
            self.vel_peaks = self.vel[peaks]


    def vel_peaks2chan_peaks(self):
        """
        This function is useful for when you know the velocities of the spectral lines,
        and need to determined the relevant channels before subtracting the bandpass.
        """
        self.channel_peaks = []
        for vp in self.vel_peaks:
            self.channel_peaks.append(np.abs(self.vel-vp).argmin())


    def subtract_bandpass(self, window_length=101, poly_order=1, p0s=None, width_factor=3, allowable_peak_gap=10):
        """
        Fit Gaussians to the specified lines, then subtract the coarse
        bandpass. Initial guesses for the Gaussian fits can (and should)
        be supplied; otherwise, you perform a Hail Mary to scipy.
        """
        self.bandpass = copy.copy(self.original)

        mask = np.zeros(len(self.original))

        for i, p in enumerate(self.channel_peaks):
            if p0s is None:
                p0 = (self.original[p], p, 5)
            else:
                p0 = p0s[i]
            width = p0[2]

            ## Naive implementation - assumes that the spectral line sits on a bandpass of 0.
            # popt, _ = optimize.curve_fit(gaussian, x, self.original, p0=p0)
            # self.bandpass -= gaussian(x, *popt)

            # ## Isolate the relevant chunk of the spectrum before fitting.
            # width = p0[2]
            # chunk = self.original[p-int(width_factor*width):p+int(width_factor*width)]
            # median = np.median(chunk)
            # popt, _ = optimize.curve_fit(gaussian, np.arange(len(chunk)), chunk-median, p0=p0)
            # popt[1] -= int(width_factor*width) + p
            # self.bandpass -= gaussian(x, *popt)

            ## Blank the lines, fitting the bandpass around them.
            mask[p+int(-width*width_factor):p+int(width*width_factor)] = 1

        self.filtered = copy.copy(self.original)

        # Interpolate between gaps in the spectrum.
        edges = np.where(np.diff(mask))[0]
        for i in xrange(len(edges)/2):
            e1, e2 = edges[2*i], edges[2*i+1]

            if e1 < allowable_peak_gap or e2 > len(self.original) - allowable_peak_gap:
                print "Peak found within the edge of the spectrum - too lazy to account for this now, pull requests welcome."
                # exit()
                continue
            # Need a check for e2 being too close to the next e1.

            range_1 = np.arange(e1-allowable_peak_gap, e1)
            range_2 = np.arange(e2, e2+allowable_peak_gap)
            interp_range = np.concatenate((range_1, range_2))
            poly_fit = np.poly1d(np.polyfit(interp_range, self.filtered[interp_range], poly_order))
            self.filtered[e1:e2] = poly_fit(np.arange(e1, e2))

        self.bandpass = signal.savgol_filter(self.filtered, window_length=window_length, polyorder=poly_order)
        self.modified = self.original - self.bandpass
