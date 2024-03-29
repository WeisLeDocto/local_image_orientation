# coding: utf-8

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib
from matplotlib import pyplot as plt
from itertools import product
from functools import partial
from math import prod
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm
import sys
import cv2
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
matplotlib.use("TkAgg")


def periodic_gaussian(x, sigma, a, b, mu):
  return b + a * np.exp(
    - (((x + np.pi / 2 - mu) % np.pi - np.pi / 2) / sigma) ** 2)


def periodic_gaussian_2(x,
                        sigma_1, a_1, b_1,
                        sigma_2, a_2, b_2, mu_1, mu_2):
  return (periodic_gaussian(x, sigma_1, mu_1, a_1, b_1) +
          periodic_gaussian(x, sigma_2, mu_2, a_2, b_2))


def periodic_gaussian_3(x,
                        sigma_1, a_1, b_1,
                        sigma_2, a_2, b_2,
                        sigma_3, a_3, b_3, mu_1, mu_2, mu_3):
  return (periodic_gaussian(x, sigma_1, mu_1, a_1, b_1) +
          periodic_gaussian(x, sigma_2, mu_2, a_2, b_2) +
          periodic_gaussian(x, sigma_3, mu_3, a_3, b_3))


def search_maxima(input_aray):
  ret = np.full((*input_aray.shape[:2], 3), -1, dtype=np.int16)
  for i, (x, y) in tqdm(enumerate(product(*map(range, input_aray.shape[:2]))),
                        total=prod(input_aray.shape[:2]),
                        desc='Peak detection',
                        file=sys.stdout,
                        colour='green'):
    min_index = np.argmin(input_aray[x, y])
    to_search = np.append(input_aray[x, y][min_index:],
                          input_aray[x, y][:min_index])
    peak_index, props = find_peaks(
      to_search,
      prominence=0.05 * (np.max(input_aray[x, y]) - np.min(input_aray[x, y])))
    peak_index = (peak_index + min_index) % len(res[x, y])
    proms = props['prominences']
    proms, peak_index = zip(*sorted(zip(proms, peak_index), reverse=True))
    peak_index = peak_index[:3]
    ret[x, y, :len(peak_index)] = peak_index
  return ret


def curve_fit_wrapper(args):
  i, (meth, maxima, x_data, y_data, max_fev, p0, bounds) = args
  n_peaks = np.count_nonzero(np.invert(np.isnan(maxima)))
  if n_peaks == 1:
    meth = partial(meth, mu=maxima[0])
  elif n_peaks == 2:
    meth = partial(meth, mu_1=maxima[0], mu_2=maxima[1])
  elif n_peaks == 3:
    meth = partial(meth, mu_1=maxima[0], mu_2=maxima[1], mu_3=maxima[2])
  return i, curve_fit(meth,
                      x_data,
                      y_data,
                      maxfev=max_fev,
                      p0=p0,
                      bounds=bounds)[0]


def fit_curve(maxima, gabor, ang_steps):

  low_bound = (0, 0, 0)
  up_bound = (np.inf, np.inf, np.inf)
  guess = (0.5, 0.5, 0)

  ret = np.zeros((*gabor.shape[:2], 9))
  ret[:, :, ::3] = np.inf

  n_peak = np.count_nonzero(np.invert(np.isnan(maxima)), axis=2)
  fit_meth = np.empty_like(n_peak, dtype='O')
  fit_meth[n_peak == 1] = periodic_gaussian
  fit_meth[n_peak == 2] = periodic_gaussian_2
  fit_meth[n_peak == 3] = periodic_gaussian_3

  pool_iterables = enumerate(zip(
    fit_meth.flatten(),
    maxima.reshape(-1, *maxima.shape[2:]) * np.pi / 180,
    repeat(ang_steps * np.pi / 180),
    gabor.reshape(-1, *gabor.shape[2:]),
    repeat(50000),
    (n_peak[x, y] * guess for x, y in
     product(*map(range, gabor.shape[:2]))),
    ((n_peak[x, y] * low_bound, n_peak[x, y] * up_bound)
     for x, y in product(*map(range, gabor.shape[:2])))))

  with ProcessPoolExecutor(max_workers=8) as executor:
    for i, vals in tqdm(executor.map(curve_fit_wrapper, pool_iterables),
                        total=prod(gabor.shape[:2]),
                        desc='Gaussian interpolation',
                        file=sys.stdout,
                        colour='green'):
      # vals = list(chain(*sorted(batched(vals, 3), key=sort_key,
      #                           reverse=True)))
      ret[*np.unravel_index(i, gabor.shape[:2]), :len(vals)] = vals

  return ret


if __name__ == '__main__':

  res = np.load('./result.npy')[350:550, 350:550]
  img = cv2.imread('/home/weis/Downloads/Transmission/2/capture_5-'
                   'capture_320-Weights= Triangular - Response curve= '
                   'Linear - Model= Debevec.exr',
                   cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  img = img[750: 1600, 1750: 2750, 1][350:550, 350:550]
  nb_ang = 45
  x, y = 72, 50
  ang = np.linspace(0, 180, nb_ang)

  peak_idx = search_maxima(res)
  peaks = ang[peak_idx]
  peaks[peak_idx == -1] = np.nan

  print()
  for i, n in enumerate(
      np.histogram(np.count_nonzero(np.invert(np.isnan(peaks)), axis=2),
                   bins=range(1, 5))[0],
      start=1):
    print(f"{i} peaks: {n} pixels")
  print()

  if False:
    fit = fit_curve(peaks, res, ang)
    np.save('./gaussian_fit.npy', fit)
  else:
    fit = np.load('./gaussian_fit.npy')

  plt.figure()
  plt.subplot(1, 3, 1)
  plt.imshow(peaks[:, :, 0], cmap='twilight', clim=(0, 180))
  plt.subplot(1, 3, 2)
  plt.imshow(peaks[:, :, 0], cmap='twilight', alpha=0.5, clim=(0, 180))
  plt.imshow(peaks[:, :, 1], cmap='twilight', clim=(0, 180))
  plt.subplot(1, 3, 3)
  plt.imshow(peaks[:, :, 0], cmap='twilight', alpha=0.5, clim=(0, 180))
  plt.imshow(peaks[:, :, 1], cmap='twilight', alpha=0.5, clim=(0, 180))
  plt.imshow(peaks[:, :, 2], cmap='twilight', clim=(0, 180))
  plt.show(block=False)

  plt.figure()
  peaks_sup = peaks.copy()
  peaks_sup[peaks_sup < 90] = np.nan
  plt.subplot(2, 3, 1)
  plt.imshow(peaks_sup[:, :, 0], cmap='twilight', clim=(0, 180))
  plt.subplot(2, 3, 2)
  plt.imshow(peaks_sup[:, :, 0], cmap='twilight', alpha=0.5, clim=(0, 180))
  plt.imshow(peaks_sup[:, :, 1], cmap='twilight', clim=(0, 180))
  plt.subplot(2, 3, 3)
  plt.imshow(peaks_sup[:, :, 0], cmap='twilight', alpha=0.5, clim=(0, 180))
  plt.imshow(peaks_sup[:, :, 1], cmap='twilight', alpha=0.5, clim=(0, 180))
  plt.imshow(peaks_sup[:, :, 2], cmap='twilight', clim=(0, 180))
  peaks_inf = peaks.copy()
  peaks_inf[peaks_inf > 90] = np.nan
  plt.subplot(2, 3, 4)
  plt.imshow(peaks_inf[:, :, 0], cmap='twilight', clim=(0, 180))
  plt.subplot(2, 3, 5)
  plt.imshow(peaks_inf[:, :, 0], cmap='twilight', alpha=0.5, clim=(0, 180))
  plt.imshow(peaks_inf[:, :, 1], cmap='twilight', clim=(0, 180))
  plt.subplot(2, 3, 6)
  plt.imshow(peaks_inf[:, :, 0], cmap='twilight', alpha=0.5, clim=(0, 180))
  plt.imshow(peaks_inf[:, :, 1], cmap='twilight', alpha=0.5, clim=(0, 180))
  plt.imshow(peaks_inf[:, :, 2], cmap='twilight', clim=(0, 180))
  plt.show(block=False)

  plt.figure()
  plt.subplot(1, 2, 1)
  plt.imshow(np.average(res, axis=2), cmap='plasma')
  plt.subplot(1, 2, 2)
  plt.imshow(ang[np.argmax(res, axis=2)], cmap='twilight', clim=(0, 180))
  plt.show(block=False)

  plt.figure()
  plt.subplot(2, 2, 1)
  plt.imshow(img, cmap='plasma')
  plt.subplot(2, 2, 2)
  plt.imshow(fit[:, :, 0], clim=(np.percentile(fit[:, :, 0], 1),
                                 np.percentile(fit[:, :, 0], 99)),
             cmap='plasma')
  plt.subplot(2, 2, 3)
  plt.imshow(fit[:, :, 1], clim=(np.percentile(fit[:, :, 1], 1),
                                 np.percentile(fit[:, :, 1], 99)),
             cmap='plasma')
  plt.subplot(2, 2, 4)
  plt.imshow(fit[:, :, 2], clim=(np.percentile(fit[:, :, 2], 1),
                                 np.percentile(fit[:, :, 2], 99)),
             cmap='plasma')
  plt.show()
