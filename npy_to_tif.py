# coding: utf-8

"""Converts .npy files to .tif

Takes all the .npy files in the specified folder, loads them using numpy,
rotates them and then saves them to another folder in the .tif format using
SimpleITK.
"""

from numpy import load, swapaxes, float64
from SimpleITK import GetImageFromArray, WriteImage, Cast, sitkUInt8
from pathlib import Path

if __name__ == "__main__":

  folder_tif = 'tif'
  folder_npy = 'npy'
  parent_path = Path(__file__).parent
  list_ = sorted((parent_path / folder_npy).glob('*.npy'))

  for img_path in list_:
    target_path = parent_path / folder_tif / \
                  img_path.name.replace('.npy', '.tif')

    if not target_path.parent.is_dir():
      target_path.parent.mkdir()

    if not target_path.is_file():
      print('Processing {}'.format(str(img_path)))
      img = swapaxes(swapaxes(load(str(img_path)), 0, 1), 2, 0).astype(float64)
      WriteImage(Cast(GetImageFromArray(img), sitkUInt8), str(target_path),
                 imageIO="TIFFImageIO")

    else:
      print('The file {} already exists'.format(str(target_path)))
