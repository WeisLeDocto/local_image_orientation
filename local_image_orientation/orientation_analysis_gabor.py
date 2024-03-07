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

  if False:

    sigma = 10

    gradient = np.swapaxes(np.swapaxes(np.array([
      sobel(img, 0, mode='nearest'), sobel(img, 1, mode='nearest')]),
      1, 0), 2, 1)

    struct_tens = np.swapaxes(np.swapaxes(np.array(
      [[gaussian_filter(gradient[:, :, 0] ** 2, sigma=sigma, order=0,
                        mode='nearest', truncate=4.0, axes=None),
        gaussian_filter(gradient[:, :, 0] * gradient[:, :, 1], sigma=sigma,
                        order=0, mode='nearest', truncate=4.0, axes=None)],
       [gaussian_filter(gradient[:, :, 0] * gradient[:, :, 1], sigma=sigma,
                        order=0, mode='nearest', truncate=4.0, axes=None),
        gaussian_filter(gradient[:, :, 1] ** 2, sigma=sigma, order=0,
                        mode='nearest', truncate=4.0, axes=None)]]),
      2, 0), 3, 1)

    # Trace of the structure tensor
    trs2 = np.trace(np.matmul(struct_tens, struct_tens), axis1=2, axis2=3)
    tr2s = np.trace(struct_tens, axis1=2, axis2=3) ** 2
    trs2[trs2 <= 0] = np.percentile(trs2, 0.1)

    # Fractional anisotropy
    ampl = np.sqrt(0.5 * (3 - tr2s / trs2))
    # ampl = (ampl - ampl.min()) / (ampl.max() - ampl.min())
    flat_ampl = ampl.flatten()

    plt.figure()
    plt.imshow(ampl)
    plt.show()

    # Eigenvalues calculation
    lambda_1 = (struct_tens[:, :, 0, 0] + struct_tens[:, :, 1, 1] +
                np.sqrt(
                  4 * struct_tens[:, :, 0, 1] ** 2 + (
                      struct_tens[:, :, 0, 0] -
                      struct_tens[:, :, 1, 1]) ** 2)) / 2
    lambda_2 = (struct_tens[:, :, 0, 0] + struct_tens[:, :, 1, 1] -
                np.sqrt(
                  4 * struct_tens[:, :, 0, 1] ** 2 + (
                      struct_tens[:, :, 0, 0] -
                      struct_tens[:, :, 1, 1]) ** 2)) / 2

    # Sorting the eigenvalues
    lambda_max = np.maximum(lambda_1, lambda_2)

    # Eigenvectors calculation
    dir_max_x = struct_tens[:, :, 0, 1]
    dir_max_y = lambda_max - struct_tens[:, :, 0, 0]
    dir_max_x, dir_max_y = (
      dir_max_x / np.sqrt(dir_max_x ** 2 + dir_max_y ** 2),
      dir_max_y / np.sqrt(dir_max_x ** 2 + dir_max_y ** 2))

    # Angles of the eigenvectors
    theta_max = np.angle(dir_max_x + dir_max_y * 1.0j, deg=True)
    theta_max = np.mod(theta_max, 180)
    flat_theta_max = theta_max.flatten()

    plt.figure()
    plt.subplot(211)
    plt.imshow(theta_max, cmap='hsv')
    plt.subplot(212)
    plt.imshow(img)
    plt.show()

    # Generating the histograms
    hist_max, _ = np.histogram(flat_theta_max, 180,
                               range=(0, 180),
                               weights=flat_ampl)
    hist_max = hist_max / hist_max.max()

    h, w = img.shape

    div = 5
    X = np.arange(0, w, div)
    Y = np.arange(0, h, div)
    X, Y = np.meshgrid(X, Y)
    U = np.zeros((h // div, w // div))
    V = np.zeros((h // div, w // div))
    x_weighted = dir_max_x * ampl
    y_weighted = dir_max_y * ampl

    for x, y in product(range(h // div), range(w // div)):
      U[x, y] = np.average(x_weighted[div * x: div * (x + 1) - 1,
                           div * y: div * (y + 1) - 1])
      V[x, y] = np.average(y_weighted[div * x: div * (x + 1) - 1,
                           div * y: div * (y + 1) - 1])

    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    ax1.imshow(img)
    ax1.quiver(X, Y, U, V, scale=0.8, scale_units='dots', pivot='middle')
    ax2.imshow(x_weighted, cmap='hsv')
    ax3.imshow(y_weighted, cmap='hsv')
    ax4.imshow(theta_max, cmap='hsv')
    ax5.imshow(U, cmap='hsv')
    ax6.imshow(V, cmap='hsv')
    plt.show()

    _, ax = plt.subplots()
    ax.imshow(img)
    ax.quiver(X, Y, U, V, scale=0.06, scale_units='dots', pivot='middle',
              headwidth=3, headlength=3, headaxislength=3)
    plt.show()

    plot_histogram(hist_max)

if True:
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
