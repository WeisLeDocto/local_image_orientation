# coding: utf-8

import cv2
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from re import search
from itertools import product
from typing import Sequence

roi_x = slice(750, 1600, 1)
roi_y = slice(1750, 2750, 1)
color = 1


def show_image(img: np.ndarray, percentile: float) -> None:
  """"""

  plt.figure()
  plt.imshow(img, clim=(np.percentile(img, percentile),
                        np.percentile(img, 100 - percentile)))
  plt.show()


def process_mertens(img: Sequence[np.ndarray],
                    exp: Sequence[float],
                    contrast: float = 1.0,
                    saturation: float = 1.0,
                    exposure: float = 1.0) -> np.ndarray:
  """"""

  # Performing the Mertens merge
  merge = cv2.createMergeMertens(
    contrast_weight=contrast,
    saturation_weight=saturation,
    exposure_weight=exposure).process(img, exp)

  # Cropping, isolating the green channel and rescaling
  merge = merge[roi_x, roi_y, color]
  merge = (merge - merge.min()) / (merge.max() - merge.min())

  # Calculating the energy of the image
  return np.sqrt(np.gradient(merge, axis=0) ** 2 +
                 np.gradient(merge, axis=1) ** 2)


if __name__ == '__main__':

  matplotlib.use('TkAgg')

  # Loading the images and the shutter speed
  images_names = tuple(Path('./raw_images').glob('*.tiff'))
  images = tuple(cv2.imread(str(image)) for image in images_names)
  shutter_speed = tuple(int(search(r'\d+', image.name)[0])
                        for image in images_names)
  exposure_time = tuple(1 / shut for shut in shutter_speed)

  res = process_mertens(images, exposure_time, 0, 1, 1)
  show_image(res, 10)
  np.save('mertens.npy', res)

  if False:

    test_values = [0, 1, 2]
    res = np.empty((len(test_values),
                    len(test_values),
                    roi_x.stop - roi_x.start,
                    roi_y.stop - roi_y.start))

    for (i, saturation_weight), (j, exposure_weight) in (
        product(enumerate(test_values), repeat=2)):
      print(f"Processing image {i}, {j}")
      res[i, j] = process_mertens(images, exposure_time,
                                  1, saturation_weight, exposure_weight)

    plt.figure()
    plt.tight_layout()
    for (i, _), (j, _) in product(enumerate(test_values), repeat=2):
      plt.subplot(len(test_values), len(test_values),
                  i * len(test_values) + j + 1)
      plt.title(f'saturation {test_values[i]}, exposure {test_values[j]}')
      plt.imshow(res[i, j], clim=(np.percentile(res[i, j], 10),
                                  np.percentile(res[i, j], 90)))
    plt.show()
