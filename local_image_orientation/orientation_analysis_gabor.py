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

from itertools import product
from skimage.feature import structure_tensor
from scipy.ndimage import sobel
from pathlib import Path


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

  img = cv2.imread('/home/weis/Downloads/Transmission/2/capture_5-'
                   'capture_320-Weights= Triangular - Response curve= '
                   'Linear - Model= Debevec.exr',
                   cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  img = img[750: 1600, 1750: 2750, 1]

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

  for i, angle in enumerate(np.linspace(0, np.pi, nb_ang)):
    gab_kernel = gabor_kernel(frequency=1 / nb_pix, n_stds=3, offset=0,
                              theta=angle, bandwidth=1,
                              dtype=np.complex64, sigma_x=4, sigma_y=7.5)
    filtered = convolve2d(img, gab_kernel,
                          mode='same', boundary='symm').astype(
      np.complex64)
    gab = np.sqrt(filtered.real ** 2 + filtered.imag ** 2)
    res[:, :, i] = gab
    print(f"{i + 1} / {nb_ang}")

  if False:
    for i, _ in enumerate(np.linspace(0, np.pi, nb_ang)):
      res[:, :, i] = convolve2d(res[:, :, i], np.ones((5, 5)) / 25,
                                mode='same', boundary='symm')

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

  for i, angle in enumerate(np.linspace(0, np.pi, nb_ang)):
    gab_kernel = gabor_kernel(frequency=1 / nb_pix, n_stds=3, offset=0,
                              theta=angle, bandwidth=1,
                              dtype=np.complex64, sigma_x=4, sigma_y=7.5)
    filtered = convolve2d(intensity, gab_kernel,
                          mode='same', boundary='symm').astype(
      np.complex64)
    gab = np.sqrt(filtered.real ** 2 + filtered.imag ** 2)
    res[:, :, i] = gab
    print(f"{i + 1} / {nb_ang}")

  plt.figure()
  plt.plot(np.linspace(0, 180, nb_ang), res[367, 250])
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
