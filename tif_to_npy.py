# coding: utf-8

"""Converts .tif files to .npy

Takes all the .tif files in the specified folder, loads them using SimpleITK,
rotates them and then saves them to another folder in the .tif format using
numpy.
"""

from numpy import save, swapaxes, float64
from SimpleITK import ReadImage, GetArrayFromImage
from pathlib import Path

if __name__ == "__main__":

  folder_tif = 'tif'
  folder_npy = 'npy'
  parent_path = Path(__file__).parent
  list_ = sorted((parent_path / folder_tif).glob('*.tif'))

  for img_path in list_:
    target_path = parent_path / folder_npy / \
                  img_path.name.replace('.tif', '.npy')

    if not target_path.parent.is_dir():
      target_path.parent.mkdir()

    if not target_path.is_file():
      print('Processing {}'.format(str(img_path)))
      img = GetArrayFromImage(ReadImage(str(img_path)))
      save(str(target_path),
           swapaxes(swapaxes(img, 0, 2), 1, 0).astype(float64))

    else:
      print('{} already exists'.format(str(target_path)))
