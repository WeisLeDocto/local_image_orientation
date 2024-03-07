# coding: utf-8
# https://docs.opencv.org/3.4/d2/df0/tutorial_py_hdr.html

from pathlib import Path
import matplotlib
from re import search
import cv2
from matplotlib import pyplot as plt
import numpy as np

roi_x = slice(750, 1600, 1)
roi_y = slice(1750, 2750, 1)
color = 1
calibrate = True


def show_image(img: np.ndarray, percentile: float) -> None:
  """"""

  plt.figure()
  plt.imshow(img, clim=(np.percentile(img, percentile),
                        np.percentile(img, 100 - percentile)))
  plt.show()


if __name__ == '__main__':

  matplotlib.use('TkAgg')

  # Loading the images and the shutter speed
  images_names = tuple(Path('./raw_images').glob('*.tiff'))
  images = tuple(cv2.imread(str(image)) for image in images_names)
  shutter_speed = tuple(int(search(r'\d+', image.name)[0])
                        for image in images_names)
  exposure_time = tuple(1 / shut for shut in shutter_speed)

  if calibrate:
    cal = cv2.createCalibrateRobertson().process(
      src=images, times=np.array(exposure_time, dtype=np.float32))
    merge = cv2.createMergeRobertson().process(
      src=images, times=np.array(exposure_time, dtype=np.float32),
      response=cal)
  else:
    merge = cv2.createMergeRobertson().process(
      src=images, times=np.array(exposure_time, dtype=np.float32))

  # Cropping, isolating the green channel and rescaling
  merge = merge[roi_x, roi_y, color]
  merge = (merge - merge.min()) / (merge.max() - merge.min())

  merge = np.sqrt(np.gradient(merge, axis=0) ** 2 +
                  np.gradient(merge, axis=1) ** 2)

  show_image(merge, 10)
  np.save('robertson_cal.npy' if calibrate else 'robertson.npy', merge)
