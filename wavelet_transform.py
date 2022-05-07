# Cauchy Wavelet for EXAFS, adopted from
# matlab code from Munoz, Argoul, and Farges:
#
# CONTINUOUS CAUCHY WAVELET TRANSFORM OF EXAFS SIGNAL
# code freely downloaded from http://www.univ-mlv.fr/~farges/waw
# (c) 2000, Univ. Marne la Vallee, France
#
#  please cite us of this code with:
#   Munoz M., Argoul P. and Farges F.
#   Continuous Cauchy wavelet transform astyses of
#   EXAFS spectra: a qualitative approach.
#   American Mineralogist 88, pp. 694-700 (2003).
#
# version history:
# 1999 Hans-Argoul : core wavelet algorithm
# 1999-2002 Argoul-Munoz : EXAFS adapation
# 2002 Farges : graphical and user interface
# 2003 Munoz : CPU optimizations and graphical updates
# 2003 Farges-Munoz : various fixes and web version
#
# 2014-Apr M Newville : translated to Python for Larch
#
# 2022-May S Tetef : translated to pure Python compatability

import numpy as np


def cauchy_wavelet(k, chi=None, rmax_out=10, kweight=0, nfft=2048):
    """
    Wavelet transform.

    Cauchy Wavelet Transform for XAFS, following work of
    Munoz, Argoul, and Farges.

    Parameters:
    -----------
      k:        1-d array of photo-electron wavenumber in Ang^-1
      chi:      1-d array of chi
      rmax_out: highest R for output data (10 Ang)
      kweight:  exponent for weighting spectra by k**kweight
      nfft:     value to use for N_fft (2048).

      Returns:
    ---------
      out:       complex cauchy wavelet(k, R)

    """
    kstep = np.round(1000. * (k[1] - k[0])) / 1000.0
    rstep = (np.pi / 2048) / kstep
    rmin = 1.e-7
    rmax = rmax_out
    nrpts = int(np.round((rmax - rmin) / rstep))
    nkout = len(k)
    if kweight != 0:
        chi = chi * k**kweight

    # extend EXAFS to 1024 data points...
    nft = int(nfft / 2)
    if len(k) < nft:
        knew = np.arange(nft) * kstep
        xnew = np.zeros(nft) * kstep
        xnew[:len(k)] = chi
    else:
        knew = k[:nft]
        xnew = chi[:nft]

    # FT parameters
    freq = (1.0 / kstep) * np.arange(nfft) / (2 * nfft)
    omega = 2 * np.pi * freq

    # simple FT calculation
    tff = np.fft.fft(xnew, n=2 * nfft)

    # scale parameter
    r = np.linspace(0, rmax, nrpts)
    r[0] = 1.e-19
    a = nrpts / (2 * r)

    # Characteristic values for Cauchy wavelet:
    cauchy_sum = np.log(2 * np.pi) - np.log(1.0 + np.arange(nrpts)).sum()

    # Main calculation:
    out = np.zeros(nkout * nrpts,
                   dtype='complex128').reshape(nrpts, nkout)
    for i in range(nrpts):
        aom = a[i] * omega
        aom[np.where(aom == 0)] = 1.e-19
        filt = cauchy_sum + nrpts * np.log(aom) - aom
        tmp = np.conj(np.exp(filt)) * tff[:nfft]
        out[i, :] = np.fft.ifft(tmp, 2 * nfft)[:nkout]

    return out
