# coding: utf-8

from pathlib import Path
import numpy as np
from re import search
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from skimage.filters import gabor_kernel
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


def periodic_gaussian(x, sigma, a, b, mu):
  """"""

  return b + a * np.exp(
    - (((x + np.pi / 2 - mu) % np.pi - np.pi / 2) / sigma) ** 2)


def periodic_gaussian_2(x,
                        sigma_1, a_1, b_1,
                        sigma_2, a_2, b_2, mu_1, mu_2):
  """"""

  return (periodic_gaussian(x, sigma_1, mu_1, a_1, b_1) +
          periodic_gaussian(x, sigma_2, mu_2, a_2, b_2))


def periodic_gaussian_3(x,
                        sigma_1, a_1, b_1,
                        sigma_2, a_2, b_2,
                        sigma_3, a_3, b_3, mu_1, mu_2, mu_3):
  """"""

  return (periodic_gaussian(x, sigma_1, mu_1, a_1, b_1) +
          periodic_gaussian(x, sigma_2, mu_2, a_2, b_2) +
          periodic_gaussian(x, sigma_3, mu_3, a_3, b_3))


def search_maxima(input_aray):
  """"""

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
  """"""

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
  """"""

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

  # Loading the images and the shutter speed
  images_names = path.glob('*.tiff')
  images = (cv2.imread(str(image)) for image in images_names)
  exposure_time = (1 / int(search(r'\d+', image.name)[0])
                   for image in images_names)

  hdr = cv2.createMergeMertens(
    contrast_weight=0,
    saturation_weight=1,
    exposure_weight=1).process(tuple(images), tuple(exposure_time))
  hdr = hdr[roi_x, roi_y, color]
  hdr = (hdr - hdr.min()) / (hdr.max() - hdr.min())

  del images

  img = gaussian_filter(hdr, sigma=2, order=0,
                        mode='nearest', truncate=4.0, axes=None)
  img = ((img - img.min()) / (img.max() - img.min())).astype('float64')

  del hdr

  res = process_gabor(img, nb_ang, nb_pix)
  intensity = np.max(res, axis=2) / img

  intensity[intensity < np.percentile(intensity, 5)] = np.percentile(
    intensity, 5)
  intensity[intensity > np.percentile(intensity, 95)] = np.percentile(
    intensity, 95)

  intensity = ((intensity - intensity.min()) /
               (intensity.max() - intensity.min())).astype('float64')

  res = process_gabor(intensity, nb_ang, nb_pix)

  del intensity

  res = res[window_x, window_y]
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

  fit = fit_curve(peaks, res, ang)

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
