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
from matplotlib.collections import PolyCollection
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
matplotlib.use("TkAgg")


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


def process_gabor(image, n_ang, n_pix):

  res = np.zeros(shape=(*img.shape, n_ang), dtype='float64')

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


def plot_summary(section_y, image, normalized, directions, n_ang, dir_dist):

  plt.figure()
  plt.subplot(2, 2, 1)
  plt.hlines(section_y, 0, image.shape[1], colors='k')
  plt.imshow(image,
             cmap='plasma',
             clim=(np.percentile(image, 3),
                   np.percentile(image, 97)))
  plt.subplot(2, 2, 3)
  plt.hlines(section_y, 0, directions.shape[1], colors='k')
  plt.imshow(directions, cmap='twilight')
  ax = plt.subplot(2, 2, 2, projection='3d')
  x = np.linspace(0, 180, n_ang)
  y = np.arange(0, dir_dist.shape[1], 20)
  polys = [[(np.min(x), np.min(dir_dist[section_y])),
            *zip(x, dir_dist[section_y, loc]),
            (np.max(x), np.min(dir_dist[section_y]))]
           for loc in y]
  colors = plt.colormaps['gist_rainbow'](np.linspace(0, 1, len(polys)))
  poly = PolyCollection(polys, edgecolor='black', facecolors=colors, alpha=.4)
  ax.add_collection3d(poly, zs=y, zdir='y')
  ax.set_xlim([np.min(x), np.max(x)])
  ax.set_ylim([np.min(y), np.max(y)])
  ax.set_zlim([np.min(dir_dist[section_y]), np.max(dir_dist[section_y])])
  plt.subplot(2, 2, 4)
  plt.hlines(section_y, 0, normalized.shape[1], colors='k')
  plt.imshow(normalized,
             cmap='plasma',
             clim=(np.percentile(normalized, 3),
                   np.percentile(normalized, 97)))
  plt.show()


nb_pix = 15
nb_ang = 45

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

  gab_kernel = gabor_kernel(frequency=1 / nb_pix, n_stds=3, offset=0,
                            theta=0, bandwidth=1, dtype=np.complex64,
                            sigma_x=4, sigma_y=7.5)

  plt.figure()
  plt.subplot(2, 1, 1)
  plt.imshow(gab_kernel.real)
  plt.subplot(2, 1, 2)
  plt.imshow(gab_kernel.imag)
  plt.show()

  res = process_gabor(img, nb_ang, nb_pix)
  dir_ = np.linspace(0, 180, nb_ang)[np.argmax(res, axis=2)]
  intensity = np.max(res, axis=2) / img

  plot_summary(352, img, intensity, dir_, nb_ang, res)

  intensity[intensity < np.percentile(intensity, 5)] = np.percentile(
    intensity, 5)
  intensity[intensity > np.percentile(intensity, 95)] = np.percentile(
    intensity, 95)

  intensity = ((intensity - intensity.min()) /
               (intensity.max() - intensity.min())).astype('float64')

  res = process_gabor(intensity, nb_ang, nb_pix)
  norm = np.max(res, axis=2) / (intensity + 1)

  np.save('./result.npy', res)

  plot_summary(352, intensity, norm, dir_, nb_ang, res)
