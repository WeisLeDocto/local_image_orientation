# coding: utf-8

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from itertools import product
from math import prod
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm
import sys


def periodic_gaussian(x, sigma, mu, a, b):
  return b + a * np.exp(
    -0.5 * (((x + np.pi / 2 - mu) % np.pi - np.pi / 2) / sigma) ** 2)


def periodic_gaussian_2(x,
                        sigma_1, mu_1, a_1, b_1,
                        sigma_2, mu_2, a_2, b_2):
  return (periodic_gaussian(x, sigma_1, mu_1, a_1, b_1) +
          periodic_gaussian(x, sigma_2, mu_2, a_2, b_2))


def periodic_gaussian_3(x,
                        sigma_1, mu_1, a_1, b_1,
                        sigma_2, mu_2, a_2, b_2,
                        sigma_3, mu_3, a_3, b_3):
  return (periodic_gaussian(x, sigma_1, mu_1, a_1, b_1) +
          periodic_gaussian(x, sigma_2, mu_2, a_2, b_2) +
          periodic_gaussian(x, sigma_3, mu_3, a_3, b_3))


def search_maxima(input_aray):
  ret = np.empty(input_aray.shape[:2], dtype=np.uint8)
  print()
  for i, (x, y) in tqdm(enumerate(product(*map(range, input_aray.shape[:2]))),
                        total=prod(input_aray.shape[:2]),
                        desc='Peak detection',
                        file=sys.stdout,
                        colour='green'):
    min_index = np.argmin(input_aray[x, y])
    to_search = np.append(input_aray[x, y][min_index:],
                          input_aray[x, y][:min_index])
    peaks, *_ = find_peaks(
      to_search,
      prominence=0.05 * (np.max(input_aray[x, y]) - np.min(input_aray[x, y])))
    peaks = (peaks + min_index) % len(res[x, y])
    ret[x, y] = len(peaks)

  return ret


def curve_fit_wrapper(args):
  i, (meth, x_data, y_data, max_fev, p0, bounds) = args
  return i, curve_fit(meth,
                      x_data,
                      y_data,
                      maxfev=max_fev,
                      p0=p0,
                      bounds=bounds)[0]


def fit_curve(maxima, gabor, ang_steps, parallel: bool = False):

  low_bound = (0, -np.inf, 0, 0)
  up_bound = (np.inf, np.inf, np.inf, np.inf)
  guess = (0.5, 0, 0.5, 0)

  ret = np.zeros((*gabor.shape[:2], 12))
  ret[:, :, ::4] = np.inf

  n_peak = np.minimum(maxima, 3)
  fit_meth = np.empty_like(n_peak, dtype='O')
  fit_meth[n_peak == 1] = periodic_gaussian
  fit_meth[n_peak == 2] = periodic_gaussian_2
  fit_meth[n_peak == 3] = periodic_gaussian_3

  if not parallel:
    for i, (x, y) in tqdm(enumerate(product(*map(range, gabor.shape[:2]))),
                          total=prod(gabor.shape[:2]),
                          desc='Gaussian interpolation',
                          file=sys.stdout,
                          colour='green'):

      try:
        ret[x, y, :4 * n_peak[x, y]], *_ = curve_fit(
          fit_meth[x, y],
          ang_steps * np.pi / 180,
          gabor[x, y],
          maxfev=50000,
          p0=n_peak[x, y] * guess,
          bounds=(n_peak[x, y] * low_bound,
                  n_peak[x, y] * up_bound))

      except RuntimeError:
        pass

  else:
    pool_iterables = enumerate(zip(
      fit_meth.flatten(),
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
        ret[*np.unravel_index(i, gabor.shape[:2]), :len(vals)] = vals

  return ret


if __name__ == '__main__':

  res = np.load('./result_1.npy')[400:500, 400:500]
  nb_ang = 45
  x, y = 99, 99
  ang = np.linspace(0, 180, nb_ang)

  nb_peaks = search_maxima(res)

  print()
  for i, n in enumerate(
      np.histogram(nb_peaks, bins=range(nb_peaks.min(),
                                        nb_peaks.max() + 1))[0],
      start=1):
    print(f"{i} peaks: {n} pixels")
  print()

  if False:
    fit = fit_curve(nb_peaks, res, ang, parallel=True)
    np.save('./gaussian_fit_1.npy', fit)
  else:
    fit = np.load('./gaussian_fit_1.npy')

  plt.figure()
  plt.plot(ang, res[x, y])
  plt.plot(ang, periodic_gaussian_3(ang * np.pi / 180, *fit[x, y]))
  plt.show()
