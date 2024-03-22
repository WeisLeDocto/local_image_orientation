# coding: utf-8

from __future__ import annotations
import os
import cv2
import matplotlib
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from skimage.filters import gabor_kernel
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
matplotlib.use("TkAgg")
parallel = True


def convolve_gabor(args):
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


def plot_histogram(histogram: np.ndarray,
                   save_path: Path = None) -> None:
  """Plots the 2D histogram of the anisotropy distribution and optionally saves
  it.

  Args:
    histogram: The histogram to plot, as an ndarray.
    save_path: If not None, saves the histogram to this file.
  """

  plt.figure()
  plt.bar(np.arange(0, 180), histogram)
  if save_path is not None:
    plt.savefig(str(save_path), format='svg')
  plt.show()


if __name__ == '__main__':

  if True:
    img = cv2.imread('/home/weis/Downloads/Transmission/2/capture_5-'
                     'capture_320-Weights= Triangular - Response curve= '
                     'Linear - Model= Debevec.exr',
                     cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = img[750: 1600, 1750: 2750, 1]
  else:
    img = np.load('./../artificial/small_wave.npy')

  img = gaussian_filter(img, sigma=2, order=0,
                        mode='nearest', truncate=4.0, axes=None)

  img = ((img - img.min()) / (img.max() - img.min())).astype('float64')

  nb_pix = 15
  gab_kernel = gabor_kernel(frequency=1 / nb_pix, n_stds=3, offset=0,
                            theta=0, bandwidth=1, dtype=np.complex64,
                            sigma_x=4, sigma_y=7.5)

  plt.figure()
  plt.subplot(2, 1, 1)
  plt.imshow(gab_kernel.real)
  plt.subplot(2, 1, 2)
  plt.imshow(gab_kernel.imag)
  plt.show()

  nb_ang = 45
  res = np.zeros(shape=(*img.shape, nb_ang), dtype='float64')

  if parallel:
    print()
    pool_iterables = zip(repeat(nb_pix),
                         np.linspace(0, np.pi, nb_ang),
                         repeat(img))
    with ProcessPoolExecutor(max_workers=8) as executor:
      for i, gab in tqdm(enumerate(executor.map(convolve_gabor,
                                                pool_iterables)),
                         total=nb_ang,
                         desc='Gabor kernel convolution',
                         file=sys.stdout,
                         colour='green'):
        res[:, :, i] = gab

  else:
    print()
    for i, angle in tqdm(enumerate(np.linspace(0, np.pi, nb_ang)),
                         total=nb_ang,
                         desc='Gabor kernel convolution',
                         file=sys.stdout,
                         colour='green'):
      angle: float
      i: int
      gab_kernel = gabor_kernel(frequency=1 / nb_pix, n_stds=3, offset=0,
                                theta=np.pi / 2 - angle, bandwidth=1,
                                dtype=np.complex64, sigma_x=4, sigma_y=7.5)
      filtered = convolve2d(img, gab_kernel,
                            mode='same', boundary='symm').astype(
        np.complex64)
      gab = np.sqrt(filtered.real ** 2 + filtered.imag ** 2)
      res[:, :, i] = gab

  plt.figure()
  plt.plot(np.linspace(0, 180, nb_ang), res[367, 250])
  plt.show(block=False)

  dirr = np.linspace(0, 180, nb_ang)[np.argmax(res, axis=2)]
  intensity = np.max(res, axis=2) / img

  section_y = 350
  plt.figure()
  plt.subplot(2, 2, 1)
  plt.hlines(section_y, 0, img.shape[1], colors='k')
  plt.imshow(img)
  plt.subplot(2, 2, 3)
  plt.hlines(section_y, 0, dirr.shape[1], colors='k')
  plt.imshow(dirr, cmap='twilight')
  ax = plt.subplot(2, 2, 2, projection='3d')
  x = np.linspace(0, 180, nb_ang)
  y = np.arange(res.shape[1])
  x, y = np.meshgrid(x, y)
  z = res[section_y]
  ax.plot_surface(x, y, z, cmap='plasma')
  plt.subplot(2, 2, 4)
  plt.hlines(section_y, 0, intensity.shape[1], colors='k')
  plt.imshow(intensity, cmap='plasma',
             clim=(np.percentile(intensity, 3),
                   np.percentile(intensity, 97)))
  plt.show()

  intensity = np.nan_to_num(intensity,
                            nan=np.nanmean(intensity),
                            posinf=np.nanmean(intensity),
                            neginf=np.nanmean(intensity))

  intensity[intensity < np.percentile(intensity, 5)] = np.percentile(
    intensity, 5)
  intensity[intensity > np.percentile(intensity, 95)] = np.percentile(
    intensity, 95)

  intensity = ((intensity - intensity.min()) /
               (intensity.max() - intensity.min())).astype('float64')

  nb_pix = 15
  nb_ang = 45
  res = np.zeros(shape=(*intensity.shape, nb_ang), dtype='float64')

  if parallel:
    print()
    pool_iterables = zip(repeat(nb_pix),
                         np.linspace(0, np.pi, nb_ang),
                         repeat(intensity))
    with ProcessPoolExecutor(max_workers=8) as executor:
      for i, gab in tqdm(enumerate(executor.map(convolve_gabor,
                                                pool_iterables)),
                         total=nb_ang,
                         desc='Gabor kernel convolution',
                         file=sys.stdout,
                         colour='green'):
        res[:, :, i] = gab

  else:
    print()
    for i, angle in tqdm(enumerate(np.linspace(0, np.pi, nb_ang)),
                         total=nb_ang,
                         desc='Gabor kernel convolution',
                         file=sys.stdout,
                         colour='green'):
      gab_kernel = gabor_kernel(frequency=1 / nb_pix, n_stds=3, offset=0,
                                theta=np.pi / 2 - angle, bandwidth=1,
                                dtype=np.complex64, sigma_x=4, sigma_y=7.5)
      filtered = convolve2d(intensity, gab_kernel,
                            mode='same', boundary='symm').astype(
        np.complex64)
      gab = np.sqrt(filtered.real ** 2 + filtered.imag ** 2)
      res[:, :, i] = gab

  np.save('./result.npy', res)

  section_y = 350
  ax = plt.figure().add_subplot(1, 1, 1, projection='3d')
  x = np.linspace(0, 180, nb_ang)
  y = np.arange(0, res.shape[1], 20)
  polys = [[(np.min(x), np.min(res[section_y])),
            *zip(x, res[section_y, loc]),
            (np.max(x), np.min(res[section_y]))]
           for loc in y]
  colors = plt.colormaps['gist_rainbow'](np.linspace(0, 1, len(polys)))
  poly = PolyCollection(polys, edgecolor='black', facecolors=colors, alpha=.4)
  ax.add_collection3d(poly, zs=y, zdir='y')
  ax.set_xlim([np.min(x), np.max(x)])
  ax.set_ylim([np.min(y), np.max(y)])
  ax.set_zlim([np.min(res[section_y]), np.max(res[section_y])])
  plt.show(block=False)

  dirr = np.linspace(0, 180, nb_ang)[np.argmax(res, axis=2)]
  intensity = np.max(res, axis=2) / (intensity + 1)

  section_y = 350
  plt.figure()
  plt.subplot(2, 2, 1)
  plt.hlines(section_y, 0, intensity.shape[1], colors='k')
  plt.imshow(intensity)
  plt.subplot(2, 2, 3)
  plt.hlines(section_y, 0, dirr.shape[1], colors='k')
  plt.imshow(dirr, cmap='twilight')
  ax = plt.subplot(2, 2, 2, projection='3d')
  x = np.linspace(0, 180, nb_ang)
  y = np.arange(0, res.shape[1], 20)
  polys = [[(np.min(x), np.min(res[section_y])),
            *zip(x, res[section_y, loc]),
            (np.max(x), np.min(res[section_y]))]
           for loc in y]
  colors = plt.colormaps['gist_rainbow'](np.linspace(0, 1, len(polys)))
  poly = PolyCollection(polys, edgecolor='black', facecolors=colors, alpha=.4)
  ax.add_collection3d(poly, zs=y, zdir='y')
  ax.set_xlim([np.min(x), np.max(x)])
  ax.set_ylim([np.min(y), np.max(y)])
  ax.set_zlim([np.min(res[section_y]), np.max(res[section_y])])
  plt.subplot(2, 2, 4)
  plt.hlines(section_y, 0, intensity.shape[1], colors='k')
  plt.imshow(intensity, cmap='plasma',
             clim=(np.percentile(intensity, 3),
                   np.percentile(intensity, 97)))
  plt.show()
