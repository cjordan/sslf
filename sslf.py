#!/usr/bin/env python2

import copy

import numpy as np
from scipy import optimize, signal

# import matplotlib.pyplot as plt


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


def find_cwt_peak(cwt_spec, snr=6.5, rms=None):
    """
    From a wavelet-domain spectrum (and an expected scale), find a peak
    and return its fitted parameters.
    """
    y_peak = max(cwt_spec)
    if rms is None:
        rms = find_background_rms(cwt_spec)

    # Is this maximum significant?
    if y_peak/rms > snr:
        x_peak = find_nearest(cwt_spec, y_peak)
        return x_peak, y_peak/rms

    return None, None


def blank_spectrum_part(spectrum, point, radius, value=0):
    lower = point+int(-radius)
    upper = point+int(radius)
    if lower < 0:
        lower = 0
    if upper > len(spectrum):
        upper = len(spectrum)
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
        - for each specified scale:
        -- find a significant peak
        -- subtract its contribution in the wavelet domain, continue searching for peaks
        - return the list of peaks

        An SNR of 6.5 is a good compromise for reducing the number of false positives found
        while reliably finding real, significant peaks.

        It may be worthwhile to be in some smoothing for every element of cwt_mat.
        """

        assert len(scales) > 0, "No scales supplied!"

        peaks = []
        cwt_mat = signal.cwt(self.original, wavelet, scales)

        for i, scale in enumerate(scales):
            y = cwt_mat[i]
            x = np.arange(len(y)).astype(np.float)
            rms = find_background_rms(y)

            # Blank any existing peaks.
            # fig, ax = plt.subplots(1, figsize=(20, 1))
            for p in peaks:
                lower = p.channel+int(-scale*2)
                upper = p.channel+int(scale*2)
                if lower < 0:
                    lower = 0
                if upper > len(y):
                    upper = len(y)
                peak = max(y[lower:upper])
                peak_pos = find_nearest(y, peak)

                # peak_snr = peak/rms
                # if peak_snr > p.snr:
                #     print "Updating peak", p.channel, "scale from", p.width, "to", scale, "snr from", p.snr, "to", peak_snr
                #     print ""
                #     # The peak found in this CWT is more significant than that found before. Update.
                #     x_peak = find_nearest(y, peak)
                #     p.channel = x_peak
                #     p.snr = peak_snr
                #     p.width = scale

                # Replace the part of the spectrum containing the peak with zeros.
                # ax.plot(y, c='b')
                y -= mexican_hat(x, peak, peak_pos, scale)
                # ax.plot(y, c='g')
                # blank_spectrum_part(y, p.channel, radius=1.3*scale)

            rms = find_background_rms(y)
            while True:
                # Find peaks.
                # There should be checks on the returned parameters,
                # e.g. against the current i (scale length)
                peak, sig = find_cwt_peak(y, snr=snr, rms=rms)
                if peak is not None:
                    # ax.plot(y, c='r')
                    # ax.plot(mexican_hat(x, y[peak], peak, scale))
                    y -= mexican_hat(x, y[peak], peak, scale)
                    # ax.plot(y)
                    # blank_spectrum_part(y, peak, radius=1.3*scale)
                else:
                    break

                # Check if this peak has already been found.
                # if not np.any(np.isclose([peak], [p.channel for p in peaks],
                #                          atol=min_space, rtol=0)):
                if not np.any(np.isclose([peak], [p.channel for p in peaks] + [0, len(y)],
                                         atol=min_space, rtol=0)):
                    # Round the peak positions to integers.
                    peaks.append(Peak(np.rint(peak).astype(int), sig, scale))
            # ax.plot(y, c='k')
            # ax.set_xlim(0, len(y))
            # plt.show()

        self.channel_peaks = [p.channel for p in peaks]
        self.peak_snrs = [p.snr for p in peaks]
        self.peak_widths = [p.width for p in peaks]
        if self.vel is not None:
            self.vel_peaks = [self.vel[p.channel] for p in peaks]


    def vel_peaks2chan_peaks(self):
        """
        This function is useful for when you know the velocities of the spectral lines,
        and need to determined the relevant channels before subtracting the bandpass.
        """
        self.channel_peaks = []
        for vp in self.vel_peaks:
            self.channel_peaks.append(np.abs(self.vel-vp).argmin())


    def subtract_bandpass(self, window_length=151, poly_order=1, p0s=None, allowable_peak_gap=10):
        """
        Fit Gaussians to the specified lines, then subtract the coarse
        bandpass. Initial guesses for the Gaussian fits can (and should)
        be supplied; otherwise, you perform a Hail Mary to scipy.
        """
        self.bandpass = copy.copy(self.original)

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
                # import matplotlib.pyplot as plt
                # plt.plot(self.original)
                # plt.plot(mask)
                # print self.channel_peaks
                # plt.plot(filtered)
                # plt.show()
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
