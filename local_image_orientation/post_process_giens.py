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
from matplotlib import pyplot as plt
from time import sleep


from workflow_gabor import process_gabor_gpu

matplotlib.use('TkAgg')

base_path = Path('/home/weis/Desktop/HDR/Fascia_Ina_1')
n_images = 8
roi_x = slice(712, 1187, 1)
roi_y = slice(626, 1326, 1)
nb_ang = 45
nb_pix = 15

if __name__ == '__main__':

  images_path = base_path / 'images'
  hdr_path = base_path / 'hdr'
  gabor_path = base_path / 'gabor'
  peak_path = base_path / 'peaks'
  fit_path = base_path / 'fit'

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
      hdr = ((hdr - hdr.min()) / (hdr.max() - hdr.min())).astype('float64')

      name = fullmatch(r'(\d+)_\d+\.npy', step[0].name).groups()[0]
      np.save(hdr_path / f'{i}_{name}.npy', hdr)

  if True:
    gabor_path.mkdir(parents=False, exist_ok=True)

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

      res = process_gabor_gpu(img, nb_ang, nb_pix)
      intensity = np.max(res, axis=2) / np.min(img[img > 0])

      intensity[intensity < np.percentile(intensity, 2)] = np.percentile(
        intensity, 2)
      intensity[intensity > np.percentile(intensity, 98)] = np.percentile(
        intensity, 98)

      intensity = ((intensity - intensity.min()) /
                   (intensity.max() - intensity.min())).astype('float64')

      res = process_gabor_gpu(intensity, nb_ang, nb_pix)

      np.save(gabor_path / img_path.name, res)
