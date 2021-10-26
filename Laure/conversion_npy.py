# coding: utf-8

from numpy import zeros, save
from SimpleITK import ReadImage, GetArrayFromImage
# from os import system
from pathlib import Path

if __name__ == "__main__":

  folder_tif = 'tif'
  folder_npy = 'npy'
  parent_path = Path(__file__).parent.parent
  list_ = sorted((parent_path / folder_tif).glob('*.tif'))
  print(len(list_))

  for img_path in list_:
    target_path = parent_path / folder_npy / \
                  img_path.name.replace('.tif', '.npy')

    if not target_path.is_file():
      image = ReadImage(str(img_path))
      fib_c = GetArrayFromImage(image)
      fib_col = zeros((fib_c.shape[1], fib_c.shape[2], fib_c.shape[0]))
      for n in range(fib_c.shape[0]):
        print(n)
        fib_col[:, :, n] = fib_c[n, :, :]
      save(str(target_path), fib_col)
      # command = 'rm ' + img
      # system(command)
      print('next')

    else:
      print('here')
