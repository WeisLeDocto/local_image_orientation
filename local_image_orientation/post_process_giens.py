# coding: utf-8

from pathlib import Path
from itertools import batched
import cv2
import numpy as np
from re import fullmatch
from tqdm import tqdm
import sys
from typing import Tuple
import matplotlib
from matplotlib import animation as anim, pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cucim.skimage.filters as gpu_filters
import cupy as cp
from time import time

from workflow_gabor import process_gabor_gpu, search_maxima_wrapper, fit_curve

matplotlib.use('TkAgg')

base_path = Path('/home/weis/Desktop/HDR/Fascia_Ina_1')
n_images = 8
roi_x = slice(712, 1187, 1)
roi_y = slice(626, 1326, 1)
nb_ang = 45
nb_pix = 15

if __name__ == '__main__':

  ang = np.linspace(0, 180, nb_ang)

  images_path = base_path / 'images'
  hdr_path = base_path / 'hdr'
  gabor_path = base_path / 'gabor'
  peak_path = base_path / 'peaks'
  fit_path = base_path / 'fit'
  anim_path = base_path / 'anim'

  if True:
    hdr_path.mkdir(parents=False, exist_ok=True)

    images = tuple(batched(sorted(images_path.glob('*.npy')), n_images))
    for i, step in tqdm(enumerate(images),
                        total=len(images),
                        desc='Converting to HDR',
                        file=sys.stdout,
                        colour='green',
                        mininterval=0.01,
                        maxinterval=0.1):
      i: int
      step: Tuple[Path, ...]

      hdr = cv2.createMergeMertens(
        contrast_weight=0,
        saturation_weight=1,
        exposure_weight=1).process(tuple(np.load(img)[roi_x, roi_y]
                                         for img in step),
                                   tuple(int(fullmatch(r'\d+_(\d+)\.npy',
                                                       img.name).groups()[0])
                                         for img in step))

      name = fullmatch(r'(\d+)_\d+\.npy', step[0].name).groups()[0]
      np.save(hdr_path / f'{i}_{name}.npy', hdr)

  if True:
    gabor_path.mkdir(parents=False, exist_ok=True)

    print('\nSetting convolution kernels')
    t0 = time()
    kernels = {i: gpu_filters.gabor_kernel(frequency=1 / nb_pix,
                                           theta=np.pi / 2 - ang,
                                           n_stds=3,
                                           offset=0,
                                           bandwidth=1,
                                           dtype=cp.complex64,
                                           sigma_x=4,
                                           sigma_y=7.5)
               for i, ang in enumerate(np.linspace(0, np.pi, nb_ang))}
    print(f'Set convolution kernels in {time() - t0:.2f}s\n')

    images = tuple(sorted(hdr_path.glob('*.npy')))
    for img_path in tqdm(images,
                         total=len(images),
                         desc='Applying Gabor filter',
                         file=sys.stdout,
                         colour='green',
                         mininterval=0.01,
                         maxinterval=0.1):
      img_path: Path

      img = np.load(img_path)

      res = process_gabor_gpu(img, kernels, nb_ang)
      intensity = np.max(res, axis=2) / np.min(img[img > 0])

      intensity[intensity < np.percentile(intensity, 2)] = np.percentile(
        intensity, 2)
      intensity[intensity > np.percentile(intensity, 98)] = np.percentile(
        intensity, 98)

      intensity = ((intensity - intensity.min()) /
                   (intensity.max() - intensity.min())).astype('float64')

      res = process_gabor_gpu(intensity, kernels, nb_ang)

      np.save(gabor_path / img_path.name, res)

    del kernels

  if True:
    peak_path.mkdir(parents=False, exist_ok=True)

    images = tuple(sorted(gabor_path.glob('*.npy')))

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
      for img_path, peaks, amp, sigma, offset in tqdm(
          executor.map(search_maxima_wrapper, zip(images, repeat(ang))),
          total=len(images),
          desc='Detecting peaks',
          file=sys.stdout,
          colour='green'):

        np.savez(peak_path / f'{img_path.stem}.npz', peaks, amp, sigma, offset)

  if True:
    anim_path.mkdir(parents=False, exist_ok=True)

    def sort_images(path: Path):
      """"""

      sec, = fullmatch(r'(\d+)_\d+\.npy', path.name).groups()
      return int(sec)

    images = tuple(sorted(gabor_path.glob('*.npy'), key=sort_images))
    hdrs = tuple(sorted(hdr_path.glob('*.npy'), key=sort_images))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.axis('off')
    ax2.axis('off')
    img1 = ax1.imshow(ang[np.argmax(np.load(images[0]), axis=2)],
                      cmap='twilight', clim=(0, 180))
    divider_1 = make_axes_locatable(ax1)
    cax1 = divider_1.append_axes('bottom', size='5%', pad=0.05)
    fig.colorbar(img1, cax=cax1, orientation='horizontal')
    img2 = ax2.imshow(np.load(hdrs[0]), cmap='grey', clim=(0, 1))
    divider_2 = make_axes_locatable(ax2)
    cax2 = divider_2.append_axes('bottom', size='5%', pad=0.05)
    fig.colorbar(img2, cax=cax2, orientation='horizontal')


    def update(frame):
      """"""

      img1.set_array(ang[np.argmax(np.load(images[frame + 1]), axis=2)])
      img2.set_array(np.load(hdrs[frame + 1]))
      return img1, img2


    ani = anim.FuncAnimation(fig=fig, func=update, frames=len(images) - 1,
                             interval=500, repeat=True, repeat_delay=2000)
    ani.save(anim_path / "orientation_2.mkv", writer='ffmpeg', fps=2)
    plt.show()

    fig, ax = plt.subplots()
    img = plt.imshow(np.average(np.load(images[0]), axis=2),
                     cmap='plasma', clim=(0, 0.25))
    bar = plt.colorbar()


    def update(frame):
      """"""

      img.set_array(np.average(np.load(images[frame + 1]), axis=2))
      img.set_clim(0, 0.25)
      return img


    ani = anim.FuncAnimation(fig=fig, func=update, frames=len(images) - 1,
                             interval=500, repeat=False)
    ani.save(anim_path / "intensity.gif", writer='imagemagick', fps=2)
    plt.show()

    images = tuple(sorted(hdr_path.glob('*.npy'), key=sort_images))

    fig, ax = plt.subplots()
    img = plt.imshow(np.load(images[0]), cmap='plasma')
    bar = plt.colorbar()


    def update(frame):
      """"""

      img.set_array(np.load(images[frame + 1]))
      return img


    ani = anim.FuncAnimation(fig=fig, func=update, frames=len(images) - 1,
                             interval=500, repeat=True)
    ani.save(anim_path / "raw_hdr.gif", writer='imagemagick', fps=2)
    plt.show()

  if True:
    fit_path.mkdir(parents=False, exist_ok=True)
    images_names = tuple(path.stem for path in peak_path.glob('*.npz'))

    for img_name in tqdm(images_names,
                         total=len(images_names),
                         desc='Fitting Gaussian curves',
                         file=sys.stdout,
                         colour='green'):
      data = np.load(peak_path / f'{img_name}.npz')
      peaks, amp, sigma, offset = (data['arr_0'], data['arr_1'],
                                   data['arr_2'], data['arr_3'])
      res = np.load(gabor_path / f'{img_name}.npy')

      fit, residual = fit_curve(peaks * np.pi / 180,
                                res,
                                ang * np.pi / 180,
                                amp,
                                sigma,
                                offset)

      np.savez(fit_path / f'{img_name}.npz', fit, residual)
