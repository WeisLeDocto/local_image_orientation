# coding: utf-8

from __future__ import annotations
import dask.array as da
from dask.distributed import Client, LocalCluster, progress
from numpy import load, pi, ndarray
from numpy.linalg import eigh
from matplotlib import pyplot as plt
from scipy.ndimage.filters import sobel, median_filter, gaussian_filter
from pathlib import Path
from time import sleep
from ctypes import CDLL
from multiprocessing.connection import Connection
from multiprocessing import Pipe
from threading import Thread

methods = ['exact ST', 'fast ST', 'gradient']


def trim_memory() -> int:
  """Deletes the useless cached memory during dask's execution."""

  libc = CDLL("libc.so.6")
  return libc.malloc_trim(0)


def run(pipe: Connection) -> None:
  """Regularly calls trim_memory until told to stop."""

  while not pipe.poll():
    trim_memory()
    sleep(0.1)


def plot_histogram(histogram: ndarray,
                   title: str = '',
                   save_path: Path = None) -> None:
  """Plots the 2D histogram of the anisotropy distribution and optionally saves
  it.

  Args:
    histogram: The histogram to plot, as an ndarray.
    title: The title of the histogram.
    save_path: If not None, saves the histogram to this file.
  """

  plt.figure()
  plt.imshow(histogram, cmap='hot', interpolation='nearest',
             extent=[-90, 90, 90, -90])
  plt.xticks([-90, -45, 0, 45, 90])
  plt.yticks([-90, -45, 0, 45, 90])
  plt.colorbar()
  plt.xlabel('Phi')
  plt.ylabel('Theta')
  plt.title(title)
  if save_path is not None:
    plt.savefig(str(save_path), format='svg')


def orientation(img_path: str | Path | list,
                mask_path: str | Path | list = None,
                method: str = 'gradient',
                memory_limit: str = '12GB',
                save_folder: str | Path | list = None) -> None:
  """Calculates the anisotropy distribution of a 3D image and plots it in a 2D
  histogram.

  Args:
    img_path: The path(s) of the image(s) to compute, as a str, pathlib Path, or
      a list of str and/or Path.
    mask_path: The path(s) to the mask(s) to apply to the image(s). One mask
      must be provided for each image. If left to None, no mask is applied.
    method: Three methods are currently implemented : the 'gradient' one that
      calculates only the main orientation from the gradient, the 'fast ST' that
      calculates the three eigen orientations from the structure tensor using an
      explicit method, and 'exact ST' that calculates the same three
      orientations from the structure tensor but using the recursive method of
      numpy.linalg.eigh.
    memory_limit: The memory limit dask will try not to exceed.
    save_folder: The path(s) the the folder(s) where the histogram(s) should be
      saved. One folder must be provided for each image. If left to None, the
      histograms are not saved.
  """

  if isinstance(img_path, str) or isinstance(img_path, Path):
    img_path = [img_path]

  if mask_path is not None and isinstance(mask_path, str) or \
          isinstance(mask_path, Path):
    mask_path = [mask_path]

  if save_folder is not None and isinstance(save_folder, str) or \
          isinstance(save_folder, Path):
    save_folder = [save_folder]

  # The img and mask lists must have the same length
  if mask_path is not None and len(mask_path) != len(img_path):
    raise ValueError("A mask should be given for each image if any is given !")

  # The img and save folder lists must have the same length
  if save_folder is not None and len(save_folder) != len(img_path):
    raise ValueError("A save folder should be given for each image if any "
                     "is given !")

  if method not in methods:
    raise ValueError("Wrong method chosen ! Available methods are "
                     "{}".format(methods))

  # Checking that the given paths are valid
  paths = [Path(path) if isinstance(path, str) else path for path in img_path]
  for path in paths:
    if not path.is_file():
      raise IOError("Could not find the image on path : {}".format(str(path)))

  # Checking that the given masks are valid
  # If no mask is given, building a list of None for consistency
  masks = [Path(path) if isinstance(path, str)
           else path for path in mask_path] if mask_path is not None \
      else [None] * len(img_path)
  if mask_path is not None:
    for path in masks:
      if not path.is_file():
        raise IOError("Could not find the mask on path : {}".format(str(path)))

  # Creating the folders that do not exist yet
  # If no save folder is given, building a list of None for consistency
  folders = [Path(path) if isinstance(path, str)
             else path for path in save_folder] if save_folder is not None \
      else [None] * len(img_path)
  if save_folder is not None:
    for path in folders:
      if not path.is_dir():
        print("Folder {} not found, creating it.".format(str(path)))
        path.mkdir()

  # Starting the local client
  cluster = LocalCluster(processes=False,
                         memory_limit=memory_limit,
                         silence_logs=40)
  Client(cluster)

  for path, mask, folder in zip(paths, masks, folders):

    print("Processing image {}".format(str(path)))

    # Loading image
    img = da.from_array(load(str(path)), chunks='auto')

    # Loading mask if any
    if mask is not None:
      mask = da.from_array(load(str(mask)), chunks='auto')

    # Mirror padding
    img = da.pad(img, 10, 'symmetric')

    if method != 'exact ST':

      # Spatial derivatives
      d_dx = img.map_overlap(sobel, axis=0, mode='nearest', depth=10)
      d_dy = img.map_overlap(sobel, axis=1, mode='nearest', depth=10)
      d_dz = img.map_overlap(sobel, axis=2, mode='nearest', depth=10)

      # Structure tensor components
      median_xz = da.map_overlap(gaussian_filter,
                                 d_dx * d_dz,
                                 sigma=0.2, depth=10, mode='nearest')
      median_xy = da.map_overlap(gaussian_filter,
                                 d_dx * d_dy,
                                 sigma=0.2, depth=10, mode='nearest')
      median_yz = da.map_overlap(gaussian_filter,
                                 d_dy * d_dz,
                                 sigma=0.2, depth=10, mode='nearest')
      median_xx = da.map_overlap(gaussian_filter, d_dx ** 2, sigma=0.2,
                                 depth=10, mode='nearest')
      median_yy = da.map_overlap(gaussian_filter, d_dy ** 2, sigma=0.2,
                                 depth=10, mode='nearest')
      median_zz = da.map_overlap(gaussian_filter, d_dz ** 2, sigma=0.2,
                                 depth=10, mode='nearest')

      # Trace of the structure tensor
      trs2 = median_xx ** 2 + median_yy ** 2 + median_zz ** 2 + 2 * \
          (median_yz ** 2 + median_xz ** 2 + median_xy ** 2)
      tr2s = median_xx ** 2 + median_yy ** 2 + median_zz ** 2 + 2 * \
          (median_xx * median_yy + median_xx * median_zz +
           median_yy * median_zz)
      trs2[trs2 == 0] = 1e-17

      # Fractional anisotropy
      ampl = da.sqrt(0.5 * (3 - tr2s / trs2))
      ampl = ampl[10:-10, 10:-10, 10:-10]
      flat_ampl = ampl[~mask] if mask is not None else ampl.flatten()

      if method == 'fast ST':

        # Intermediate parameters for the eigenvalues
        c0 = median_xx * median_yz ** 2 + median_yy * median_xz ** 2 + \
            median_zz * median_xy ** 2 - median_xx * median_yy * median_zz - \
            2 * median_xy * median_xz * median_yz
        c1 = median_xx * median_yy + median_xx * median_zz + \
            median_yy * median_zz - median_xy ** 2 - median_xz ** 2 - \
            median_yz ** 2
        c2 = - median_xx - median_yy - median_zz
        p = c2 ** 2 - 3 * c1
        p[p < 0] = 0
        q = -13.5 * c0 - c2 ** 3 + 4.5 * c2 * c1
        r = 27 * (0.25 * c1 ** 2 * (p - c1) + c0 * (q + 6.75 * c0))
        r[r < 0] = 0
        mat_phi = da.arctan2(da.sqrt(r), q) / 3

        # Eigenvalues calculation
        lambda_1 = da.sqrt(p) / 3 * 2 * da.cos(mat_phi) - c2 / 3
        lambda_2 = da.sqrt(p) / 3 * 2 * da.cos(mat_phi + 2 * pi / 3) - c2 / 3
        lambda_3 = da.sqrt(p) / 3 * 2 * da.cos(mat_phi - 2 * pi / 3) - c2 / 3

        # Sorting the eigenvalues
        lambda_max = da.maximum(lambda_1, da.maximum(lambda_2, lambda_3))
        lambda_min = da.minimum(lambda_1, da.minimum(lambda_2, lambda_3))
        lambda_int = lambda_1 + lambda_2 + lambda_3 - lambda_min - lambda_max

        # Fractional anisotropy in the transverse direction
        tr2 = (lambda_int ** 2 + lambda_min ** 2)
        tr2[tr2 <= 0] = 1e-17
        ampl2 = da.sqrt((lambda_int - lambda_min) ** 2 / tr2)
        ampl2 = ampl2[10:-10, 10:-10, 10:-10]
        flat_ampl2 = ampl2[~mask] if mask is not None else ampl2.flatten()

        # Eigenvectors calculation
        dir_max_x = ((median_xx - lambda_int) * (median_xx - lambda_min) +
                     median_xy ** 2 + median_xz ** 2)
        dir_max_y = (median_xy * (median_xx - lambda_min) +
                     (median_yy - lambda_int) * median_xy + median_yz *
                     median_xz)
        dir_max_z = (median_xz * (median_xx - lambda_min) + median_yz *
                     median_xy + (median_zz - lambda_int) *
                     median_xz)

        dir_int_x = ((median_xx - lambda_max) * (median_xx - lambda_min) +
                     median_xy ** 2 + median_xz ** 2)
        dir_int_y = (median_xy * (median_xx - lambda_min) +
                     (median_yy - lambda_max) * median_xy + median_yz *
                     median_xz)
        dir_int_z = (median_xz * (median_xx - lambda_min) + median_yz *
                     median_xy + (median_zz - lambda_max) *
                     median_xz)

        dir_min_x = ((median_xx - lambda_max) * (median_xx - lambda_int) +
                     median_xy ** 2 + median_xz ** 2)
        dir_min_y = (median_xy * (median_xx - lambda_int) +
                     (median_yy - lambda_max) * median_xy + median_yz *
                     median_xz)
        dir_min_z = (median_xz * (median_xx - lambda_int) + median_yz *
                     median_xy + (median_zz - lambda_max) *
                     median_xz)

        # Angles of the eigenvectors
        theta_max = ((da.angle(dir_max_x +
                               dir_max_y * 1.0j,
                               deg=True) - 90) % 180) - 90
        theta_max = theta_max[10:-10, 10:-10, 10:-10]
        flat_theta_max = theta_max[~mask] if mask is not None \
            else theta_max.flatten()
        phi_max = da.angle(da.sqrt(dir_max_x ** 2 +
                                   dir_max_y ** 2) +
                           dir_max_z * 1.0j, deg=True)
        phi_max = phi_max[10:-10, 10:-10, 10:-10]
        flat_phi_max = phi_max[~mask] if mask is not None else phi_max.flatten()
        theta_int = ((da.angle(dir_int_x +
                               dir_int_y * 1.0j,
                               deg=True) - 90) % 180) - 90
        theta_int = theta_int[10:-10, 10:-10, 10:-10]
        flat_theta_int = theta_int[~mask] if mask is not None \
            else theta_int.flatten()
        phi_int = da.angle(da.sqrt(dir_int_x ** 2 +
                                   dir_int_y ** 2) +
                           dir_int_z * 1.0j, deg=True)
        phi_int = phi_int[10:-10, 10:-10, 10:-10]
        flat_phi_int = phi_int[~mask] if mask is not None else phi_int.flatten()
        theta_min = ((da.angle(dir_min_x +
                               dir_min_y * 1.0j,
                               deg=True) - 90) % 180) - 90
        theta_min = theta_min[10:-10, 10:-10, 10:-10]
        flat_theta_min = theta_min[~mask] if mask is not None \
            else theta_min.flatten()
        phi_min = da.angle(da.sqrt(dir_min_x ** 2 +
                                   dir_min_y ** 2) +
                           dir_min_z * 1.0j, deg=True)
        phi_min = phi_min[10:-10, 10:-10, 10:-10]
        flat_phi_min = phi_min[~mask] if mask is not None else phi_min.flatten()

        # Generating the histograms
        hist_max, _, _ = da.histogram2d(flat_theta_max, flat_phi_max, 180,
                                        range=[[-90, 90], [-90, 90]],
                                        weights=flat_ampl)
        hist_max = hist_max / hist_max.max()
        hist_int, _, _ = da.histogram2d(flat_theta_int, flat_phi_int, 180,
                                        range=[[-90, 90], [-90, 90]],
                                        weights=flat_ampl2)
        hist_int = hist_int / hist_int.max()
        hist_min, _, _ = da.histogram2d(flat_theta_min, flat_phi_min, 180,
                                        range=[[-90, 90], [-90, 90]],
                                        weights=1 - flat_ampl2)
        hist_min = hist_min / hist_min.max()

        # Computing histograms
        hist_max = hist_max.persist()
        hist_int = hist_int.persist()
        hist_min = hist_min.persist()
        pipe_in, pipe_out = Pipe()
        thread = Thread(target=run, args=(pipe_out,))
        thread.start()
        progress(hist_max, hist_int, hist_min)

        h_max, h_int, h_min = da.compute(hist_max, hist_int, hist_min)
        pipe_in.send('stop')
        thread.join()

        plot_histogram(h_max, 'Main orientation, fast ST method',
                       folder / 'Main_fastST.svg' if folder is not None
                       else None)
        plot_histogram(h_int, 'Second orientation, fast ST method',
                       folder / 'Second_fastST.svg' if folder is not None
                       else None)
        plot_histogram(h_min, 'Third orientation, fast ST method',
                       folder / 'Third_fastST.svg' if folder is not None
                       else None)

      elif method == 'gradient':

        # Median on the spatial derivatives
        med_d_dx = da.map_overlap(median_filter, d_dx, size=3,
                                  depth=10, mode='nearest')
        med_d_dx[med_d_dx == 0] = 1e-17
        med_d_dy = da.map_overlap(median_filter, d_dy, size=3,
                                  depth=10, mode='nearest')
        med_d_dy[med_d_dy == 0] = 1e-17
        med_d_dz = da.map_overlap(median_filter, d_dz, size=3,
                                  depth=10, mode='nearest')

        # Angles of the main orientation
        theta = ((da.angle(med_d_dx + med_d_dy * 1.0j,
                           deg=True) - 90) % 180) - 90
        theta = theta[10:-10, 10:-10, 10:-10]
        flat_theta = theta[~mask] if mask is not None else theta.flatten()

        phi = da.angle(da.sqrt(med_d_dx ** 2 + med_d_dy ** 2) + med_d_dz * 1.0j,
                       deg=True)
        phi = phi[10:-10, 10:-10, 10:-10]
        flat_phi = phi[~mask] if mask is not None else phi.flatten()

        # Generating the histogram
        hist, _, _ = da.histogram2d(flat_theta, flat_phi, 180,
                                    range=[[-90, 90], [-90, 90]],
                                    weights=flat_ampl)
        hist = hist / hist.max()

        # Computing histogram
        hist = hist.persist()
        pipe_in, pipe_out = Pipe()
        thread = Thread(target=run, args=(pipe_out,))
        thread.start()
        progress(hist)

        h = hist.compute()
        pipe_in.send('stop')
        thread.join()

        plot_histogram(h, 'Main orientation, gradient method',
                       folder / 'Main_grad.svg' if folder is not None else None)

    else:
      # Spatial derivatives
      gradient = da.swapaxes(da.swapaxes(da.swapaxes(da.array([
        img.map_overlap(sobel, axis=0, mode='nearest', depth=10),
        img.map_overlap(sobel, axis=1, mode='nearest', depth=10),
        img.map_overlap(sobel, axis=2, mode='nearest', depth=10)]), 1, 0), 2,
        1), 3, 2)

      # Structure tensor
      struct_tens = da.swapaxes(da.swapaxes(da.swapaxes(da.swapaxes(da.array(
        [[da.map_overlap(gaussian_filter, gradient[:, :, :, 0] ** 2, sigma=0.2,
                         depth=10, mode='nearest'),
          da.map_overlap(gaussian_filter,
                         gradient[:, :, :, 0] * gradient[:, :, :, 1],
                         sigma=0.2, depth=10, mode='nearest'),
          da.map_overlap(gaussian_filter,
                         gradient[:, :, :, 0] * gradient[:, :, :, 2],
                         sigma=0.2, depth=10, mode='nearest')],
         [da.map_overlap(gaussian_filter,
                         gradient[:, :, :, 0] * gradient[:, :, :, 1],
                         sigma=0.2, depth=10, mode='nearest'),
          da.map_overlap(gaussian_filter, gradient[:, :, :, 1] ** 2, sigma=0.2,
                         depth=10, mode='nearest'),
          da.map_overlap(gaussian_filter,
                         gradient[:, :, :, 1] * gradient[:, :, :, 2],
                         sigma=0.2, depth=10, mode='nearest')],
         [da.map_overlap(gaussian_filter,
                         gradient[:, :, :, 0] * gradient[:, :, :, 2],
                         sigma=0.2, depth=10, mode='nearest'),
          da.map_overlap(gaussian_filter,
                         gradient[:, :, :, 1] * gradient[:, :, :, 2],
                         sigma=0.2, depth=10, mode='nearest'),
          da.map_overlap(gaussian_filter, gradient[:, :, :, 2] ** 2, sigma=0.2,
                         depth=10, mode='nearest')]]), 2, 0), 3, 1), 4, 3), 3,
        2)

      # Trace of the structure tensor
      trs2 = da.trace(da.matmul(struct_tens, struct_tens), axis1=3, axis2=4)
      tr2s = da.trace(struct_tens, axis1=3, axis2=4) ** 2
      trs2[trs2 == 0] = 1e-17

      # Fractional anisotropy
      ampl = da.sqrt(0.5 * (3 - tr2s / trs2))
      ampl = ampl[10:-10, 10:-10, 10:-10]
      flat_ampl = ampl[~mask] if mask is not None else ampl.flatten()

      # Eigen vectors calculation
      values, vectors = da.apply_gufunc(eigh, '(i,j)->(i),(i,j)', struct_tens)

      # Fractional anisotropy in the transverse direction
      tr2 = values[:, :, :, 0] ** 2 + values[:, :, :, 1] ** 2
      tr2[tr2 <= 0] = 1e-17
      ampl2 = da.sqrt((values[:, :, :, 1] - values[:, :, :, 0]) ** 2 / tr2)
      ampl2 = ampl2[10:-10, 10:-10, 10:-10]
      flat_ampl2 = ampl2[~mask] if mask is not None else ampl2.flatten()

      # Angles of the eigenvectors
      theta_max = ((da.angle(vectors[:, :, :, 0, 2] +
                             vectors[:, :, :, 1, 2] * 1.0j,
                             deg=True) - 90) % 180) - 90
      theta_max = theta_max[10:-10, 10:-10, 10:-10]
      flat_theta_max = theta_max[~mask] if mask is not None \
          else theta_max.flatten()
      phi_max = da.angle(da.sqrt(vectors[:, :, :, 0, 2] ** 2 +
                                 vectors[:, :, :, 1, 2] ** 2) +
                         vectors[:, :, :, 2, 2] * 1.0j, deg=True)
      phi_max = phi_max[10:-10, 10:-10, 10:-10]
      flat_phi_max = phi_max[~mask] if mask is not None else phi_max.flatten()
      theta_int = ((da.angle(vectors[:, :, :, 0, 1] +
                             vectors[:, :, :, 1, 1] * 1.0j,
                             deg=True) - 90) % 180) - 90
      theta_int = theta_int[10:-10, 10:-10, 10:-10]
      flat_theta_int = theta_int[~mask] if mask is not None \
          else theta_int.flatten()
      phi_int = da.angle(da.sqrt(vectors[:, :, :, 0, 1] ** 2 +
                                 vectors[:, :, :, 1, 1] ** 2) +
                         vectors[:, :, :, 2, 1] * 1.0j, deg=True)
      phi_int = phi_int[10:-10, 10:-10, 10:-10]
      flat_phi_int = phi_int[~mask] if mask is not None else phi_int.flatten()
      theta_min = ((da.angle(vectors[:, :, :, 0, 0] +
                             vectors[:, :, :, 1, 0] * 1.0j,
                             deg=True) - 90) % 180) - 90
      theta_min = theta_min[10:-10, 10:-10, 10:-10]
      flat_theta_min = theta_min[~mask] if mask is not None \
          else theta_min.flatten()
      phi_min = da.angle(da.sqrt(vectors[:, :, :, 0, 0] ** 2 +
                                 vectors[:, :, :, 1, 0] ** 2) +
                         vectors[:, :, :, 2, 0] * 1.0j, deg=True)
      phi_min = phi_min[10:-10, 10:-10, 10:-10]
      flat_phi_min = phi_min[~mask] if mask is not None else phi_min.flatten()

      # Generating the histograms
      flat_ampl = da.rechunk(flat_ampl, flat_theta_max.chunks)
      flat_ampl2 = da.rechunk(flat_ampl2, flat_theta_max.chunks)

      hist_max, _, _ = da.histogram2d(flat_theta_max, flat_phi_max, 180,
                                      range=[[-90, 90], [-90, 90]],
                                      weights=flat_ampl)
      hist_max = hist_max / hist_max.max()
      hist_int, _, _ = da.histogram2d(flat_theta_int, flat_phi_int, 180,
                                      range=[[-90, 90], [-90, 90]],
                                      weights=flat_ampl2)
      hist_int = hist_int / hist_int.max()
      hist_min, _, _ = da.histogram2d(flat_theta_min, flat_phi_min, 180,
                                      range=[[-90, 90], [-90, 90]],
                                      weights=1 - flat_ampl2)
      hist_min = hist_min / hist_min.max()

      # Computing histograms
      hist_max = hist_max.persist()
      hist_int = hist_int.persist()
      hist_min = hist_min.persist()
      pipe_in, pipe_out = Pipe()
      thread = Thread(target=run, args=(pipe_out,))
      thread.start()
      progress(hist_max, hist_int, hist_min)

      h_max, h_int, h_min = da.compute(hist_max, hist_int, hist_min)
      pipe_in.send('stop')
      thread.join()

      plot_histogram(h_max, 'Main orientation, exact ST method',
                     folder / 'Main_exactST.svg' if folder is not None
                     else None)
      plot_histogram(h_int, 'Second orientation, exact ST method',
                     folder / 'Second_exactST.svg' if folder is not None
                     else None)
      plot_histogram(h_min, 'Third orientation, exact ST method',
                     folder / 'Third_exactST.svg' if folder is not None
                     else None)

    plt.show()
