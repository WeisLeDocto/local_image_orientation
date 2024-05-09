# coding: utf-8

from pathlib import Path
import numpy as np
from re import search
from scipy.signal import convolve2d
from skimage.filters import gabor_kernel
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
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
import numba as nb
import cupyx.scipy.signal as gpu_signal
import cupy as cp
import cucim.skimage.filters as gpu_filters
from time import time

matplotlib.use('TkAgg')


roi_x = slice(750, 1600, 1)
roi_y = slice(1750, 2750, 1)
color = 1

path = Path('/home/weis/Downloads/Transmission/2/')
nb_pix = 15
nb_ang = 45

window_x = slice(350, 550, 1)
window_y = slice(350, 550, 1)


def convolve_gabor(args):
  """"""

  n_pix, ang, image = args
  kernel = gabor_kernel(frequency=1 / n_pix,
                        theta=np.pi / 2 - ang,
                        n_stds=3,
                        offset=0,
                        bandwidth=1,
                        dtype=np.complex64,
                        sigma_x=4,
                        sigma_y=7.5)
  conv = convolve2d(image,
                    kernel,
                    mode='same',
                    boundary='symm').astype(np.complex64)
  return np.sqrt(conv.real ** 2 + conv.imag ** 2)


def process_gabor(image, n_ang, n_pix):
  """"""

  res = np.zeros(shape=(*image.shape, n_ang), dtype='float64')

  print()
  pool_iterables = zip(repeat(n_pix),
                       np.linspace(0, np.pi, n_ang),
                       repeat(image))
  with ProcessPoolExecutor(max_workers=8) as executor:
    for i, gab in tqdm(enumerate(executor.map(convolve_gabor,
                                              pool_iterables)),
                       total=n_ang,
                       desc='Gabor kernel convolution',
                       file=sys.stdout,
                       colour='green'):
      res[:, :, i] = gab

  return res


def process_gabor_gpu(image, n_ang, n_pix):
  """"""

  img_gpu = cp.asarray(image, dtype='float32')
  res = cp.zeros(shape=(*image.shape, n_ang), dtype='float32')
  for i, ang in tqdm(enumerate(np.linspace(0, np.pi, n_ang)),
                     total=n_ang,
                     desc='Gabor kernel convolution',
                     file=sys.stdout,
                     colour='green',
                     mininterval=0.01,
                     maxinterval=0.1):
    kernel = gpu_filters.gabor_kernel(frequency=1 / n_pix,
                                      theta=np.pi / 2 - ang,
                                      n_stds=3,
                                      offset=0,
                                      bandwidth=1,
                                      dtype=cp.complex64,
                                      sigma_x=4,
                                      sigma_y=7.5)
    conv = gpu_signal.convolve2d(img_gpu,
                                 kernel,
                                 mode='same',
                                 boundary='symm').astype(cp.complex64)
    res[:, :, i] = cp.sqrt(conv.real ** 2 + conv.imag ** 2)

  return cp.asnumpy(res)


def periodic_gaussian(x, sigma, a, b, mu):
  """"""

  return b + a * np.exp(
    - (((x + np.pi / 2 - mu) % np.pi - np.pi / 2) / sigma) ** 2)


def periodic_gaussian_2(x, sigma_1, a_1, sigma_2, a_2, b, mu_1, mu_2):
  """"""

  return (periodic_gaussian(x, sigma_1, a_1, 0, mu_1) +
          periodic_gaussian(x, sigma_2, a_2, b, mu_2))


def periodic_gaussian_3(x, sigma_1, a_1, sigma_2, a_2, sigma_3, a_3,
                        b, mu_1, mu_2, mu_3):
  """"""

  return (periodic_gaussian(x, sigma_1, a_1, 0, mu_1) +
          periodic_gaussian(x, sigma_2, a_2, 0, mu_2) +
          periodic_gaussian(x, sigma_3, a_3, b, mu_3))


@nb.jit(nb.types.float64[:](
  nb.types.float64[:], nb.types.float64, nb.types.float64, nb.types.float64,
  nb.types.float64, nb.types.float64, nb.types.float64, nb.types.float64,
  nb.types.float64, nb.types.float64, nb.types.float64, nb.types.int64),
  nopython=True, nogil=True)
def periodic_gaussian_jit(x, sigma_1, a_1, sigma_2, a_2, sigma_3, a_3,
                          b, mu_1, mu_2, mu_3, n):
  """"""

  if n == 1:
    return (b + a_1 * np.exp(-(((x + np.pi / 2 - mu_1) % np.pi - np.pi / 2)
                               / sigma_1) ** 2))
  elif n == 2:
    return (b + a_1 * np.exp(-(((x + np.pi / 2 - mu_1) % np.pi - np.pi / 2)
                               / sigma_1) ** 2) +
            a_2 * np.exp(-(((x + np.pi / 2 - mu_2) % np.pi - np.pi / 2)
                           / sigma_2) ** 2))
  else:
    return (b + a_1 * np.exp(-(((x + np.pi / 2 - mu_1) % np.pi - np.pi / 2)
                               / sigma_1) ** 2) +
            a_2 * np.exp(-(((x + np.pi / 2 - mu_2) % np.pi - np.pi / 2)
                           / sigma_2) ** 2) +
            a_3 * np.exp(-(((x + np.pi / 2 - mu_3) % np.pi - np.pi / 2)
                           / sigma_3) ** 2))


@nb.jit(nb.types.float64[:](
  nb.types.float64[:], nb.types.float64[:], nb.types.float64, nb.types.float64,
  nb.types.float64, nb.types.float64, nb.types.float64, nb.types.float64,
  nb.types.float64, nb.types.float64, nb.types.float64, nb.types.float64,
  nb.types.int64), nopython=True, nogil=True)
def periodic_gaussian_derivative(x, y, sigma_1, a_1, sigma_2, a_2, sigma_3,
                                 a_3, b, mu_1, mu_2, mu_3, n):
  """"""

  diff = y - periodic_gaussian_jit(x, sigma_1, a_1, sigma_2, a_2, sigma_3,
                                   a_3, b, mu_1, mu_2, mu_3, n)
  exp_1 = np.exp(-((x - mu_1) / sigma_1) ** 2)

  if n == 1:
    return np.array((-4 * a_1 * np.sum(diff * exp_1 * (x - mu_1) ** 2 /
                                       (sigma_1 ** 3)),
                     -2 * np.sum(diff * exp_1),
                     0,
                     0,
                     0,
                     0,
                     -2 * np.sum(diff)))
  elif n == 2:
    exp_2 = np.exp(-((x - mu_2) / sigma_2) ** 2)
    return np.array((-4 * a_1 * np.sum(diff * exp_1 *
                                       (x - mu_1) ** 2 / (sigma_1 ** 3)),
                     -2 * np.sum(diff * exp_1),
                     -4 * a_2 * np.sum(diff * exp_2 *
                                       (x - mu_2) ** 2 / (sigma_2 ** 3)),
                     -2 * np.sum(diff * exp_2),
                     0,
                     0,
                     -2 * np.sum(diff)))
  else:
    exp_2 = np.exp(-((x - mu_2) / sigma_2) ** 2)
    exp_3 = np.exp(-((x - mu_3) / sigma_3) ** 2)
    return np.array((-4 * a_1 * np.sum(diff * exp_1 *
                                       (x - mu_1) ** 2 / (sigma_1 ** 3)),
                     -2 * np.sum(diff * exp_1),
                     -4 * a_2 * np.sum(diff * exp_2 *
                                       (x - mu_2) ** 2 / (sigma_2 ** 3)),
                     -2 * np.sum(diff * exp_2),
                     -4 * a_3 * np.sum(diff * exp_3 *
                                       (x - mu_3) ** 2 / (sigma_3 ** 3)),
                     -2 * np.sum(diff * exp_3),
                     -2 * np.sum(diff)))


def search_maxima(input_aray, angle_step):
  """"""

  ret_idx = np.full((*input_aray.shape[:2], 3), -1, dtype=np.int16)
  ret_amp = np.zeros((*input_aray.shape[:2], 3), dtype=np.float32)
  ret_sigma = np.zeros((*input_aray.shape[:2], 3), dtype=np.float32)
  ret_offset = np.zeros((*input_aray.shape[:2],), dtype=np.float32)
  for i, (x, y) in tqdm(enumerate(product(*map(range, input_aray.shape[:2]))),
                        total=prod(input_aray.shape[:2]),
                        desc='Peak detection',
                        file=sys.stdout,
                        colour='green'):
    min_index = np.argmin(input_aray[x, y])
    to_search = np.append(input_aray[x, y][min_index:],
                          input_aray[x, y][:min_index])
    min_val = np.min(input_aray[x, y])

    peak_index, props = find_peaks(
      to_search,
      prominence=0.05 * (np.max(input_aray[x, y]) - min_val),
      width=(None, None), height=(None, None))

    widths, width_heights, *_ = peak_widths(to_search, peak_index,
                                            rel_height=0.5)
    widths *= angle_step * np.pi / 180
    peak_index = (peak_index + min_index) % len(input_aray[x, y])

    proms = props['prominences']
    heights = props['peak_heights']

    proms, peak_index, widths, width_heights, heights = zip(*sorted(zip(
      proms, peak_index, widths, width_heights, heights), reverse=True))

    peak_index = peak_index[:3]
    widths = np.array(widths[:3])
    width_heights = np.array(width_heights[:3]) - min_val
    heights = np.array(heights[:3]) - min_val

    deviation = widths / (2 * np.sqrt(np.log(heights / width_heights)))

    ret_idx[x, y, :len(peak_index)] = peak_index
    ret_amp[x, y, :len(heights)] = heights
    ret_sigma[x, y, :len(deviation)] = deviation
    ret_offset[x, y] = min_val

  return ret_idx, ret_amp, ret_sigma, ret_offset


def curve_fit_wrapper(args):
  """"""

  i, (meth, maxima, x_data, y_data, max_fev, p0, bounds) = args
  n_peaks = np.count_nonzero(np.invert(np.isnan(maxima)))
  p0 = np.append(p0[:2 * n_peaks], p0[-1])
  bounds = np.append(bounds[:, :2 * n_peaks], bounds[:, -1][:, np.newaxis],
                     axis=1)
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
                      bounds=bounds,
                      full_output=True)


def fit_curve(maxima, gabor, ang_steps, guess_amp, guess_sigma, guess_offset):
  """"""

  ret = np.zeros((*gabor.shape[:2], 7))
  residuals = np.zeros(gabor.shape[:2])
  ret[:, :, 0] = np.inf
  ret[:, :, 2] = np.inf
  ret[:, :, 4] = np.inf

  guess = np.zeros_like(ret)
  guess[:, :, 0:6:2] = guess_sigma
  guess[:, :, 1:6:2] = guess_amp
  guess[:, :, -1] = guess_offset

  bounds = np.zeros((*gabor.shape[:2], 2, 7))
  bounds[:, :, 1] = np.inf

  n_peak = np.count_nonzero(np.invert(np.isnan(maxima)), axis=2)
  fit_meth = np.empty_like(n_peak, dtype='O')
  fit_meth[n_peak == 1] = periodic_gaussian
  fit_meth[n_peak == 2] = periodic_gaussian_2
  fit_meth[n_peak == 3] = periodic_gaussian_3

  pool_iterables = enumerate(zip(
    fit_meth.flatten(),
    maxima.reshape(-1, *maxima.shape[2:]),
    repeat(ang_steps),
    gabor.reshape(-1, *gabor.shape[2:]),
    repeat(50000),
    guess.reshape(-1, *guess.shape[2:]),
    bounds.reshape(-1, *bounds.shape[2:])))

  with ProcessPoolExecutor(max_workers=8) as executor:
    for i, (vals, _, info_dict, *_) in tqdm(executor.map(curve_fit_wrapper,
                                                         pool_iterables),
                                            total=prod(gabor.shape[:2]),
                                            desc='Gaussian interpolation',
                                            file=sys.stdout,
                                            colour='green'):
      ret[*np.unravel_index(i, gabor.shape[:2]), :len(vals) - 1] = vals[:-1]
      ret[*np.unravel_index(i, gabor.shape[:2]), -1] = vals[-1]
      residuals[*np.unravel_index(i, gabor.shape[:2])] = np.sum(
        info_dict['fvec']**2)

  return ret, residuals


@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64))(
  nb.types.int64, nb.types.float64[:], nb.types.float64[:],
  nb.types.float64[:], nb.types.float64[:], nb.types.float64, nb.types.int64),
  nopython=True)
def optimized(n_peak, x_data, y_data, params, mu, thresh, max_iter):
  """"""

  weight = 0.001
  residuals = np.sum((y_data -
                      periodic_gaussian_jit(x_data, params[0], params[1],
                                            params[2], params[3], params[4],
                                            params[5], params[6],
                                            mu[0], mu[1], mu[2], n_peak)) ** 2)
  params -= weight * periodic_gaussian_derivative(
    x_data, y_data, params[0], params[1], params[2], params[3], params[4],
    params[5], params[6], mu[0], mu[1], mu[2], n_peak)
  new_residuals = np.sum((y_data -
                          periodic_gaussian_jit(
                            x_data, params[0], params[1], params[2], params[3],
                            params[4], params[5], params[6], mu[0], mu[1],
                            mu[2], n_peak)) ** 2)

  params0 = params.copy()

  n = 0
  while True:
    if ((new_residuals < residuals and residuals - new_residuals < thresh)
        or n > max_iter):
      break
    n += 1

    if (np.any(np.isnan(params)) or np.any(np.isinf(params)) or
        np.isnan(new_residuals) or np.isinf(new_residuals)):
      weight = 0.001
      n = 0
      params = params0.copy()
      new_residuals = np.sum((y_data - periodic_gaussian_jit(
                                x_data, params[0], params[1], params[2],
                                params[3],
                                params[4], params[5], params[6], mu[0], mu[1],
                                mu[2], n_peak)) ** 2)

      continue

    weight = 1.2 * weight if new_residuals < residuals else 0.5 * weight
    params -= weight * periodic_gaussian_derivative(
      x_data, y_data, params[0], params[1], params[2], params[3], params[4],
      params[5], params[6], mu[0], mu[1], mu[2], n_peak)
    residuals = new_residuals
    new_residuals = np.sum((y_data - periodic_gaussian_jit(
      x_data, params[0], params[1], params[2], params[3], params[4], params[5],
      params[6], mu[0], mu[1], mu[2], n_peak)) ** 2)

  return params, new_residuals


def newton_fit_map(args):
  """"""

  i, (n_peak, maxima, x_data, y_data, params, thresh, max_iter) = args
  n_peaks = np.count_nonzero(np.invert(np.isnan(maxima)))

  mu = np.zeros((3,))

  if n_peaks == 1:
    mu[0] = maxima[0]
  elif n_peaks == 2:
    mu[0] = maxima[0]
    mu[1] = maxima[1]
  elif n_peaks == 3:
    mu[0] = maxima[0]
    mu[1] = maxima[1]
    mu[2] = maxima[2]

  return i, optimized(n_peak, x_data, y_data, params, mu, thresh, max_iter)


def fit_curve_newton(maxima, gabor, ang_steps, guess_amp, guess_sigma,
                     guess_offset, thresh, max_iter):
  """"""

  ret = np.zeros((*gabor.shape[:2], 7))
  residuals = np.zeros(gabor.shape[:2])
  ret[:, :, 0] = np.inf
  ret[:, :, 2] = np.inf
  ret[:, :, 4] = np.inf

  guess = np.zeros_like(ret)
  guess[:, :, 0:6:2] = guess_sigma
  guess[:, :, 1:6:2] = guess_amp
  guess[:, :, -1] = guess_offset

  n_peak = np.count_nonzero(np.invert(np.isnan(maxima)), axis=2)

  pool_iterables = enumerate(zip(
    n_peak.flatten(),
    maxima.reshape(-1, *maxima.shape[2:]),
    repeat(ang_steps),
    gabor.reshape(-1, *gabor.shape[2:]),
    guess.reshape(-1, *guess.shape[2:]),
    repeat(thresh),
    repeat(max_iter)))

  with ProcessPoolExecutor(max_workers=8) as executor:
    for i, (vals, res) in tqdm(executor.map(newton_fit_map,
                                            pool_iterables),
                               total=prod(gabor.shape[:2]),
                               desc='Gaussian interpolation',
                               file=sys.stdout,
                               colour='green'):
      ret[*np.unravel_index(i, gabor.shape[:2])] = vals
      residuals[*np.unravel_index(i, gabor.shape[:2])] = res

  return ret, residuals


if __name__ == '__main__':

  # Loading the images and the shutter speed
  images_names = path.glob('*.tiff')
  images = (cv2.imread(str(image)) for image in images_names)
  exposure_time = (1 / int(search(r'\d+', image.name)[0])
                   for image in images_names)

  print('\nCreating HDR image')
  t = time()
  hdr = cv2.createMergeMertens(
    contrast_weight=0,
    saturation_weight=1,
    exposure_weight=1).process(tuple(images), tuple(exposure_time))
  hdr = hdr[roi_x, roi_y, color]
  print(f'Created HDR image, took {time() - t:.2f}s\n')

  del images

  img = ((hdr - hdr.min()) / (hdr.max() - hdr.min())).astype('float64')

  del hdr

  print('\nApplying Gabor filter')
  t = time()
  res = process_gabor(img, nb_ang, nb_pix)
  intensity = np.max(res, axis=2) / img

  intensity[intensity < np.percentile(intensity, 5)] = np.percentile(
    intensity, 5)
  intensity[intensity > np.percentile(intensity, 95)] = np.percentile(
    intensity, 95)

  intensity = ((intensity - intensity.min()) /
               (intensity.max() - intensity.min())).astype('float64')

  res = process_gabor(intensity, nb_ang, nb_pix)
  print(f'Applied Gabor filter, took {time() - t:.2f}s\n')

  del intensity

  res = res[window_x, window_y]
  ang = np.linspace(0, 180, nb_ang)
  peak_idx, amp, sigma, offset = search_maxima(res, ang[1] - ang[0])
  peaks = ang[peak_idx]
  peaks[peak_idx == -1] = np.nan

  print()
  for i, n in enumerate(
      np.histogram(np.count_nonzero(np.invert(np.isnan(peaks)), axis=2),
                   bins=range(1, 5))[0],
      start=1):
    print(f"{i} peaks: {n} pixels")
  print()

  fit, residual = fit_curve(peaks * np.pi / 180, res, ang * np.pi / 180, amp,
                            sigma, offset)

  plt.figure()
  plt.imshow(residual)
  plt.show(block=False)

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
  plt.imshow(img[window_x, window_y], cmap='plasma')
  plt.subplot(2, 2, 2)
  plt.imshow(fit[:, :, 0], clim=(np.percentile(fit[:, :, 0], 1),
                                 np.percentile(fit[:, :, 0], 99)),
             cmap='plasma')
  plt.subplot(2, 2, 3)
  plt.imshow(fit[:, :, 1], clim=(np.percentile(fit[:, :, 1], 1),
                                 np.percentile(fit[:, :, 1], 99)),
             cmap='plasma')
  plt.subplot(2, 2, 4)
  plt.imshow(fit[:, :, -1], clim=(np.percentile(fit[:, :, -1], 1),
                                  np.percentile(fit[:, :, -1], 99)),
             cmap='plasma')
  plt.show()
