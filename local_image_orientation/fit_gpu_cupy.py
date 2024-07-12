# coding: utf-8

import cupy as cp
import numpy as np
from matplotlib import pyplot as plt


def periodic_gaussian_1(x: cp.ndarray, sigma_1: cp.ndarray, a_1: cp.ndarray,
                        b: cp.ndarray, mu_1: cp.ndarray) -> cp.ndarray:
  """"""

  return b + a_1 * cp.exp(
    - (((x + cp.pi / 2 - mu_1) % cp.pi - cp.pi / 2) / sigma_1) ** 2)


def periodic_gaussian_2(x: cp.ndarray, sigma_1: cp.ndarray, a_1: cp.ndarray,
                        sigma_2: cp.ndarray, a_2: cp.ndarray, b: cp.ndarray,
                        mu_1: cp.ndarray, mu_2: cp.ndarray) -> cp.ndarray:
  """"""

  return (periodic_gaussian_1(x, sigma_1, a_1, cp.zeros_like(a_1), mu_1) +
          periodic_gaussian_1(x, sigma_2, a_2, b, mu_2))


def periodic_gaussian_3(x: cp.ndarray, sigma_1: cp.ndarray, a_1: cp.ndarray,
                        sigma_2: cp.ndarray, a_2: cp.ndarray,
                        sigma_3: cp.ndarray, a_3: cp.ndarray, b: cp.ndarray,
                        mu_1: cp.ndarray, mu_2: cp.ndarray,
                        mu_3: cp.ndarray) -> cp.ndarray:
  """"""

  return (periodic_gaussian_1(x, sigma_1, a_1, cp.zeros_like(a_1), mu_1) +
          periodic_gaussian_1(x, sigma_2, a_2, cp.zeros_like(a_2), mu_2) +
          periodic_gaussian_1(x, sigma_3, a_3, b, mu_3))


def periodic_gaussian_derivative_1(x: cp.ndarray, y: cp.ndarray,
                                   sigma_1: cp.ndarray, a_1: cp.ndarray,
                                   b: cp.ndarray,
                                   mu_1: cp.ndarray) -> cp.ndarray:
  """"""

  diff = y - periodic_gaussian_1(x, sigma_1, a_1, b, mu_1)
  exp_1 = cp.exp(-((x - mu_1) / sigma_1) ** 2)
  zeros = cp.zeros_like(a_1)

  return cp.concatenate((
    -4 * a_1 * cp.sum(diff * exp_1 * (x - mu_1) ** 2 /
                      (sigma_1 ** 3), axis=1)[:, cp.newaxis],
    -2 * cp.sum(diff * exp_1, axis=1)[:, cp.newaxis],
    zeros, zeros, zeros, zeros,
    -2 * cp.sum(diff, axis=1)[:, cp.newaxis]), axis=1)


def periodic_gaussian_derivative_2(x: cp.ndarray, y: cp.ndarray,
                                   sigma_1: cp.ndarray, a_1: cp.ndarray,
                                   sigma_2: cp.ndarray, a_2: cp.ndarray,
                                   b: cp.ndarray, mu_1: cp.ndarray,
                                   mu_2: cp.ndarray) -> cp.ndarray:
  """"""

  diff = y - periodic_gaussian_2(x, sigma_1, a_1, sigma_2, a_2, b, mu_1, mu_2)

  exp_1 = cp.exp(-((x - mu_1) / sigma_1) ** 2)
  exp_2 = cp.exp(-((x - mu_2) / sigma_2) ** 2)
  zeros = cp.zeros_like(a_1)

  return cp.concatenate((
    -4 * a_1 * cp.sum(diff * exp_1 * (x - mu_1) ** 2 /
                      (sigma_1 ** 3), axis=1)[:, cp.newaxis],
    -2 * cp.sum(diff * exp_1, axis=1)[:, cp.newaxis],
    -4 * a_2 * cp.sum(diff * exp_2 * (x - mu_2) ** 2 /
                      (sigma_2 ** 3), axis=1)[:, cp.newaxis],
    -2 * cp.sum(diff * exp_2, axis=1)[:, cp.newaxis],
    zeros, zeros,
    -2 * cp.sum(diff, axis=1)[:, cp.newaxis]), axis=1)


def periodic_gaussian_derivative_3(x: cp.ndarray, y: cp.ndarray,
                                   sigma_1: cp.ndarray, a_1: cp.ndarray,
                                   sigma_2: cp.ndarray, a_2: cp.ndarray,
                                   sigma_3: cp.ndarray, a_3: cp.ndarray,
                                   b: cp.ndarray, mu_1: cp.ndarray,
                                   mu_2: cp.ndarray,
                                   mu_3: cp.ndarray) -> cp.ndarray:
  """"""

  diff = y - periodic_gaussian_3(x, sigma_1, a_1, sigma_2, a_2, sigma_3,
                                 a_3, b, mu_1, mu_2, mu_3)

  exp_1 = cp.exp(-((x - mu_1) / sigma_1) ** 2)
  exp_2 = cp.exp(-((x - mu_2) / sigma_2) ** 2)
  exp_3 = cp.exp(-((x - mu_3) / sigma_3) ** 2)

  return cp.concatenate((
    -4 * a_1 * cp.sum(diff * exp_1 * (x - mu_1) ** 2 /
                      (sigma_1 ** 3), axis=1)[:, cp.newaxis],
    -2 * cp.sum(diff * exp_1, axis=1)[:, cp.newaxis],
    -4 * a_2 * cp.sum(diff * exp_2 * (x - mu_2) ** 2 /
                      (sigma_2 ** 3), axis=1)[:, cp.newaxis],
    -2 * cp.sum(diff * exp_2, axis=1)[:, cp.newaxis],
    -4 * a_3 * cp.sum(diff * exp_3 * (x - mu_3) ** 2 /
                      (sigma_3 ** 3), axis=1)[:, cp.newaxis],
    -2 * cp.sum(diff * exp_3, axis=1)[:, cp.newaxis],
    -2 * cp.sum(diff, axis=1)[:, cp.newaxis]), axis=1)


def gradient_descent_1(x: cp.ndarray, y: cp.ndarray, params: cp.ndarray,
                       mu: cp.ndarray, n_iter: int) -> cp.ndarray:
  """"""

  mu_1 = mu[:, 0:1]

  weight = 0.001
  residuals = cp.sum((y - periodic_gaussian_1(
    x, params[:, 0:1], params[:, 1:2], params[:, 6:7], mu_1)) ** 2)
  params -= weight * periodic_gaussian_derivative_1(
    x, y, params[:, 0:1], params[:, 1:2], params[:, 6:7], mu_1)
  new_residuals = cp.sum((y - periodic_gaussian_1(
    x, params[:, 0:1], params[:, 1:2], params[:, 6:7], mu_1)) ** 2)

  for _ in range(n_iter):
    weight = 1.2 * weight if new_residuals < residuals else 0.5 * weight
    residuals = new_residuals

    params -= weight * periodic_gaussian_derivative_1(
      x, y, params[:, 0:1], params[:, 1:2], params[:, 6:7], mu_1)
    new_residuals = cp.sum((y - periodic_gaussian_1(
      x, params[:, 0:1], params[:, 1:2], params[:, 6:7], mu_1)) ** 2)

  return params


def gradient_descent_2(x: cp.ndarray, y: cp.ndarray, params: cp.ndarray,
                       mu: cp.ndarray, n_iter: int) -> cp.ndarray:
  """"""

  mu_1 = mu[:, 0:1]
  mu_2 = mu[:, 1:2]

  weight = 0.001
  residuals = cp.sum((y - periodic_gaussian_2(
    x, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
    params[:, 6:7], mu_1, mu_2)) ** 2)
  params -= weight * periodic_gaussian_derivative_2(
    x, y, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
    params[:, 6:7], mu_1, mu_2)
  new_residuals = cp.sum((y - periodic_gaussian_2(
    x, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
    params[:, 6:7], mu_1, mu_2)) ** 2)

  for _ in range(n_iter):
    weight = 1.2 * weight if new_residuals < residuals else 0.5 * weight
    residuals = new_residuals

    params -= weight * periodic_gaussian_derivative_2(
      x, y, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
      params[:, 6:7], mu_1, mu_2)
    new_residuals = cp.sum((y - periodic_gaussian_2(
      x, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
      params[:, 6:7], mu_1, mu_2)) ** 2)

  return params


def gradient_descent_3(x: cp.ndarray, y: cp.ndarray, params: cp.ndarray,
                       mu: cp.ndarray, n_iter: int) -> cp.ndarray:
  """"""

  mu_1 = mu[:, 0:1]
  mu_2 = mu[:, 1:2]
  mu_3 = mu[:, 2:3]

  weight = 0.001
  residuals = cp.sum((y - periodic_gaussian_3(
    x, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
    params[:, 4:5], params[:, 5:6], params[:, 6:7], mu_1, mu_2, mu_3)) ** 2)
  params -= weight * periodic_gaussian_derivative_3(
    x, y, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
    params[:, 4:5], params[:, 5:6], params[:, 6:7], mu_1, mu_2, mu_3)
  new_residuals = cp.sum((y - periodic_gaussian_3(
    x, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
    params[:, 4:5], params[:, 5:6], params[:, 6:7], mu_1, mu_2, mu_3)) ** 2)

  for _ in range(n_iter):

    weight = 1.2 * weight if new_residuals < residuals else 0.5 * weight
    residuals = new_residuals

    params -= weight * periodic_gaussian_derivative_3(
      x, y, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
      params[:, 4:5], params[:, 5:6], params[:, 6:7], mu_1, mu_2, mu_3)
    new_residuals = cp.sum((y - periodic_gaussian_3(
      x, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4],
      params[:, 4:5], params[:, 5:6], params[:, 6:7], mu_1, mu_2, mu_3)) ** 2)

  return params


def fit_gpu(n_peaks_cpu: np.ndarray, x_data_cpu: np.ndarray,
            y_data_cpu: np.ndarray, params_cpu: np.ndarray, mu_cpu: np.ndarray,
            n_iter: int):
  """"""

  w, h, *_ = y_data_cpu.shape
  x_data_cpu = x_data_cpu[np.newaxis, np.newaxis, :]
  x_data_cpu = x_data_cpu.repeat(h, axis=1).repeat(w, axis=0)
  x_data_cpu = x_data_cpu.reshape(-1, x_data_cpu.shape[-1])
  x_data = cp.asarray(x_data_cpu, dtype=cp.float32)
  del x_data_cpu

  n_peaks_cpu = n_peaks_cpu.flatten()
  one_peak_index = cp.asarray(np.where(n_peaks_cpu == 1)[0], dtype=cp.int32)
  two_peak_index = cp.asarray(np.where(n_peaks_cpu == 2)[0], dtype=cp.int32)
  three_peak_index = cp.asarray(np.where(n_peaks_cpu == 3)[0], dtype=cp.int32)
  del n_peaks_cpu

  y_data_cpu = y_data_cpu.reshape(-1, y_data_cpu.shape[-1])
  y_data = cp.asarray(y_data_cpu, dtype=cp.float32)
  del y_data_cpu

  mu_cpu = mu_cpu.reshape(-1, mu_cpu.shape[-1])
  mu = cp.asarray(mu_cpu, dtype=cp.float32)
  del mu_cpu

  params = cp.asarray(params_cpu.reshape(-1, params_cpu.shape[-1]),
                      dtype=cp.float32)

  if one_peak_index.size:
    params_one = gradient_descent_1(
      x_data[one_peak_index], y_data[one_peak_index], params[one_peak_index],
      mu[one_peak_index], n_iter)
    params[one_peak_index] = params_one
    del params_one

  if two_peak_index.size:
    params_two = gradient_descent_2(
      x_data[two_peak_index], y_data[two_peak_index], params[two_peak_index],
      mu[two_peak_index], n_iter)
    params[two_peak_index] = params_two
    del params_two

  if three_peak_index.size:
    params_three = gradient_descent_3(
      x_data[three_peak_index], y_data[three_peak_index],
      params[three_peak_index], mu[three_peak_index], n_iter)
    params[three_peak_index] = params_three
    del params_three

  return params.get().reshape(params_cpu.shape)


if __name__ == '__main__':
  from workflow_gabor import periodic_gaussian_3 as pg3

  size = 2

  x = np.linspace(0, 180, 45, dtype=np.float32)
  y = np.zeros((size, size, *x.shape), dtype=np.float32)
  y[:, :] = 3 * np.exp(-np.power((((np.radians(x) + np.pi / 2 - 0.3)
                                   % np.pi) - np.pi / 2) / 0.6, 2))
  y[:, :] += 2 * np.exp(-np.power((((np.radians(x) + np.pi / 2 - 1.5)
                                    % np.pi) - np.pi / 2) / 0.4, 2))
  y[:, :] += 1 * np.exp(-np.power((((np.radians(x) + np.pi / 2 - 2.2)
                                    % np.pi) - np.pi / 2) / 0.2, 2))
  y += 0.1

  n = np.full((size, size), 3, dtype=np.int32)
  p = np.zeros((size, size, 7), dtype=np.float32)
  p[:, :] = np.array((0.7, 2.5, 0.3, 1.5, 0.3, 1.5, 0.05), dtype=np.float32)
  m = np.zeros((size, size, 3), dtype=np.float32)
  m[:, :] = np.array((0.3, 1.5, 2.2), dtype=np.float32)

  p_2 = p.copy()
  m_2 = m.copy()

  p = fit_gpu(n, np.radians(x), y, p, m, 300)  # 50000)

  plt.figure()
  plt.plot(x, y[0, 0])
  plt.plot(x, pg3(np.radians(x), *p[0, 0], *m[0, 0]))
  plt.show()
