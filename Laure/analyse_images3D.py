# coding: utf-8

from functions_IM import side_effect, distribution_2d, structure_tensor_f

from numpy import load
from matplotlib.pyplot import ion, close, savefig
from matplotlib import rc
from pathlib import Path

if __name__ == "__main__":

  rc('font', **{'family': 'serif', 'size': 16})

  ion()
  # sigma = [0.1, 0.5, 1.0, 1.05, 1.5, 3, 5, 10]
  sigma = [3]

  parent_path = Path(__file__).parent.parent
  list_ = sorted((parent_path / 'npy').glob('*.npy'))
  print(len(list_))

  # img = 'valentin_image.npy'

  for img_path in list_:
    for s in sigma:
      fib = load(str(img_path))
      print("Loaded")
      theta, phi, ampl = side_effect(structure_tensor_f, fib, s)
      print("Side effect")
      distrib_aniso = distribution_2d(ampl, theta, phi)
      print("Distrib aniso")
      target_path = img_path.parent / img_path.name.replace('.npy', '')
      savefig(str(target_path) + '_G' + str(s) + '.svg', format='svg')
      print("Saved")
  close()
