# coding: utf-8

from __future__ import annotations
import os
import cv2
import matplotlib
from matplotlib import pyplot as plt
from skimage.feature import structure_tensor
from scipy.ndimage import gaussian_filter, median_filter, sobel
import numpy as np
from itertools import product
from pathlib import Path
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
matplotlib.use("TkAgg")

gauss = True


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

  img = ((img - img.min()) / (img.max() - img.min())).astype('float64')

  if gauss:

    struct = structure_tensor(img, sigma=20, mode='nearest', order='rc')
    struct_tens = np.stack((np.stack((struct[0], struct[1]), axis=2),
                            np.stack((struct[1], struct[2]), axis=2)),
                           axis=3)
  else:
    size = 10

    img = gaussian_filter(img, sigma=1, order=0, mode='nearest', truncate=4.0,
                          axes=None)

    gradient = np.swapaxes(np.swapaxes(np.array([
      sobel(img, 0, mode='nearest'), sobel(img, 1, mode='nearest')]),
      1, 0), 2, 1)

    struct_tens = np.swapaxes(np.swapaxes(np.array(
      [[median_filter(gradient[:, :, 0] ** 2, size=(size, size),
                      mode='nearest', axes=None),
        median_filter(gradient[:, :, 0] * gradient[:, :, 1], size=(size, size),
                      mode='nearest', axes=None)],
       [median_filter(gradient[:, :, 0] * gradient[:, :, 1], size=(size, size),
                      mode='nearest', axes=None),
        median_filter(gradient[:, :, 1] ** 2, size=(size, size),
                      mode='nearest', axes=None)]]),
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
