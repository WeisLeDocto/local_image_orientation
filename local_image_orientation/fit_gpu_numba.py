# coding: utf-8

from numba import cuda, types
import math
import numpy as np
import matplotlib.pyplot as plt

NB_ANGLES = 45


@cuda.jit(types.float32[:](types.float32[:], types.float32[:]), device=True)
def gpu_exp(array_in, array_out):
  """"""

  for i in range(array_in.shape[0]):
    array_out[i] = math.exp(array_in[i])
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.int32, types.float32[:]),
          device=True)
def gpu_pow(array_in, power, array_out):
  """"""

  for i in range(array_in.shape[0]):
    array_out[i] = math.pow(array_in[i], power)
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32[:]), device=True)
def gpu_minus(array_in, array_out):
  """"""

  for i in range(array_in.shape[0]):
    array_out[i] = -array_in[i]
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32, types.float32[:]),
          device=True)
def gpu_add(array_in, val, array_out):
  """"""

  for i in range(array_in.shape[0]):
    array_out[i] = array_in[i] + val
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32, types.float32[:]),
          device=True)
def gpu_sub(array_in, val, array_out):
  """"""

  for i in range(array_in.shape[0]):
    array_out[i] = array_in[i] - val
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32, types.float32[:]),
          device=True)
def gpu_mul(array_in, val, array_out):
  """"""

  for i in range(array_in.shape[0]):
    array_out[i] = array_in[i] * val
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32, types.float32[:]),
          device=True)
def gpu_div(array_in, val, array_out):
  """"""

  for i in range(array_in.shape[0]):
    array_out[i] = array_in[i] / val
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32[:],
                           types.float32[:]), device=True)
def gpu_array_add(array_in_1, array_in_2, array_out):
  """"""

  for i in range(array_in_1.shape[0]):
    array_out[i] = array_in_1[i] + array_in_2[i]
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32[:],
                           types.float32[:]), device=True)
def gpu_array_sub(array_in_1, array_in_2, array_out):
  """"""

  for i in range(array_in_1.shape[0]):
    array_out[i] = array_in_1[i] - array_in_2[i]
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32[:],
                           types.float32[:]), device=True)
def gpu_array_mul(array_in_1, array_in_2, array_out):
  """"""

  for i in range(array_in_1.shape[0]):
    array_out[i] = array_in_1[i] * array_in_2[i]
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32[:],
                           types.float32[:]), device=True)
def gpu_array_div(array_in_1, array_in_2, array_out):
  """"""

  for i in range(array_in_1.shape[0]):
    array_out[i] = array_in_1[i] / array_in_2[i]
  return array_out


@cuda.jit(types.float32[:](types.float32[:], types.float32[:]), device=True)
def gpu_array_copy(array_in_1, array_in_2):
  """"""

  for i in range(array_in_1.shape[0]):
    array_in_2[i] = array_in_1[i]
  return array_in_2


@cuda.jit(types.float32(types.float32[:]), device=True)
def gpu_sum(array_in):
  """"""

  ret = 0.
  for i in range(array_in.shape[0]):
    ret += array_in[i]
  return ret


@cuda.jit(
  types.float32[:](types.float32[:], types.float32, types.float32,
                   types.float32, types.float32, types.float32, types.float32,
                   types.float32, types.float32, types.float32, types.float32,
                   types.int32, types.float32[:]), device=True)
def periodic_gaussian_gpu(x, sigma_1, a_1, sigma_2, a_2, sigma_3, a_3,
                          b, mu_1, mu_2, mu_3, n, array_out):
  """"""

  if n == 1:
    for i in range(x.shape[0]):
      array_out[i] = b + a_1 * math.exp(-math.pow((math.fmod(
        x[i] + math.pi / 2 - mu_1, math.pi) - math.pi / 2) / sigma_1, 2))
  elif n == 2:
    for i in range(x.shape[0]):
      array_out[i] = b + a_1 * math.exp(-math.pow(
        (math.fmod(x[i] + math.pi / 2 - mu_1, math.pi) - math.pi / 2)
        / sigma_1, 2)) + a_2 * math.exp(-math.pow(
          (math.fmod(x[i] + math.pi / 2 - mu_2, math.pi) - math.pi / 2)
          / sigma_2, 2))
  elif n == 3:
    for i in range(x.shape[0]):
      array_out[i] = b + a_1 * math.exp(-math.pow(
        (math.fmod(x[i] + math.pi / 2 - mu_1, math.pi) - math.pi / 2)
        / sigma_1, 2)) + a_2 * math.exp(-math.pow(
          (math.fmod(x[i] + math.pi / 2 - mu_2, math.pi) - math.pi / 2)
          / sigma_2, 2)) + a_3 * math.exp(-math.pow(
            (math.fmod(x[i] + math.pi / 2 - mu_3, math.pi) - math.pi / 2)
            / sigma_3, 2))
  return array_out


@cuda.jit(
  types.float32[:](types.float32[:], types.float32[:], types.float32,
                   types.float32, types.float32, types.float32, types.float32,
                   types.float32, types.float32, types.float32, types.float32,
                   types.float32, types.int32, types.float32[:]), device=True)
def periodic_gaussian_derivative(x, y, sigma_1, a_1, sigma_2, a_2, sigma_3,
                                 a_3, b, mu_1, mu_2, mu_3, n, array_out):
  """"""

  buf = cuda.local.array((NB_ANGLES,), dtype=types.float32)
  diff = cuda.local.array((NB_ANGLES,), dtype=types.float32)
  exp_1 = cuda.local.array((NB_ANGLES,), dtype=types.float32)
  exp_2 = cuda.local.array((NB_ANGLES,), dtype=types.float32)
  exp_3 = cuda.local.array((NB_ANGLES,), dtype=types.float32)

  gpu_array_sub(y, periodic_gaussian_gpu(x, sigma_1, a_1, sigma_2, a_2,
                                         sigma_3, a_3, b, mu_1, mu_2,
                                         mu_3, n, buf), diff)

  # exp_1 = np.exp(-((x - mu_1) / sigma_1) ** 2)
  gpu_exp(gpu_minus(gpu_pow(gpu_div(gpu_sub(x, mu_1, buf),
                                    sigma_1, buf), 2, buf), buf), exp_1)

  if n == 1:

    # -4 * a_1 * np.sum(diff * exp_1 * (x - mu_1) ** 2 / (sigma_1 ** 3))
    array_out[0] = -4 * a_1 * gpu_sum(gpu_div(gpu_array_mul(
      diff, gpu_array_mul(exp_1, gpu_pow(gpu_sub(x, mu_1, buf), 2, buf), buf),
      buf), sigma_1 ** 3, buf))
    # -2 * np.sum(diff * exp_1)
    array_out[1] = -2 * gpu_sum(gpu_array_mul(diff, exp_1, buf))
    # -2 * np.sum(diff)
    array_out[2] = 0.
    array_out[3] = 0.
    array_out[4] = 0.
    array_out[5] = 0.

    array_out[6] = -2 * gpu_sum(diff)

  elif n == 2:

    gpu_exp(gpu_minus(gpu_pow(gpu_div(
      gpu_sub(x, mu_2, buf), sigma_2, buf), 2, buf), buf), exp_2)

    array_out[0] = -4 * a_1 * gpu_sum(gpu_div(gpu_array_mul(
      diff, gpu_array_mul(exp_1, gpu_pow(gpu_sub(x, mu_1, buf), 2, buf), buf),
      buf), sigma_1 ** 3, buf))
    array_out[1] = -2 * gpu_sum(gpu_array_mul(diff, exp_1, buf))
    array_out[2] = -4 * a_2 * gpu_sum(gpu_div(gpu_array_mul(
      diff, gpu_array_mul(exp_2, gpu_pow(gpu_sub(x, mu_2, buf), 2, buf), buf),
      buf), sigma_2 ** 3, buf))
    array_out[3] = -2 * gpu_sum(gpu_array_mul(diff, exp_2, buf))
    array_out[6] = -2 * gpu_sum(diff)

  elif n == 3:

    gpu_exp(gpu_minus(gpu_pow(gpu_div(
      gpu_sub(x, mu_2, buf), sigma_2, buf), 2, buf), buf), exp_2)
    gpu_exp(gpu_minus(gpu_pow(gpu_div(
      gpu_sub(x, mu_3, buf), sigma_3, buf), 2, buf), buf), exp_3)

    array_out[0] = -4 * a_1 * gpu_sum(gpu_div(gpu_array_mul(
      diff, gpu_array_mul(exp_1, gpu_pow(gpu_sub(x, mu_1, buf), 2, buf), buf),
      buf), sigma_1 ** 3, buf))
    array_out[1] = -2 * gpu_sum(gpu_array_mul(diff, exp_1, buf))
    array_out[2] = -4 * a_2 * gpu_sum(gpu_div(gpu_array_mul(
      diff, gpu_array_mul(exp_2, gpu_pow(gpu_sub(x, mu_2, buf), 2, buf), buf),
      buf), sigma_2 ** 3, buf))
    array_out[3] = -2 * gpu_sum(gpu_array_mul(diff, exp_2, buf))
    array_out[4] = -4 * a_3 * gpu_sum(gpu_div(gpu_array_mul(
      diff, gpu_array_mul(exp_3, gpu_pow(gpu_sub(x, mu_3, buf), 2, buf), buf),
      buf), sigma_3 ** 3, buf))
    array_out[5] = -2 * gpu_sum(gpu_array_mul(diff, exp_3, buf))
    array_out[6] = -2 * gpu_sum(diff)

  return array_out


@cuda.jit(types.float32[:](types.int32, types.float32[:], types.float32[:],
                           types.float32[:], types.float32[:], types.float32,
                           types.int32), device=True)
def gradient_descent(n_peak, x_data, y_data, params, mu, thresh, max_iter):
  """"""

  buf = cuda.local.array((NB_ANGLES,), dtype=types.float32)
  buf_7 = cuda.local.array((7,), dtype=types.float32)

  weight = 0.001
  residuals = gpu_sum(gpu_pow(gpu_array_sub(y_data, periodic_gaussian_gpu(
    x_data, params[0], params[1], params[2], params[3], params[4], params[5],
    params[6], mu[0], mu[1], mu[2], n_peak, buf), buf), 2, buf))
  gpu_array_sub(params, gpu_mul(periodic_gaussian_derivative(
    x_data, y_data, params[0], params[1], params[2], params[3], params[4],
    params[5], params[6], mu[0], mu[1], mu[2], n_peak, buf_7), weight,
    buf_7), params)
  new_residuals = gpu_sum(gpu_pow(gpu_array_sub(y_data, periodic_gaussian_gpu(
    x_data, params[0], params[1], params[2], params[3], params[4], params[5],
    params[6], mu[0], mu[1], mu[2], n_peak, buf), buf), 2, buf))

  n = 0
  while True:
    if ((new_residuals < residuals and residuals - new_residuals < thresh)
        or n > max_iter):
      break
    n += 1

    weight = 1.2 * weight if new_residuals < residuals else 0.5 * weight

    gpu_array_sub(params, gpu_mul(periodic_gaussian_derivative(
      x_data, y_data, params[0], params[1], params[2], params[3], params[4],
      params[5], params[6], mu[0], mu[1], mu[2], n_peak, buf_7), weight,
      buf_7), params)
    residuals = new_residuals
    new_residuals = gpu_sum(gpu_pow(gpu_array_sub(
      y_data, periodic_gaussian_gpu(x_data, params[0], params[1], params[2],
                                    params[3], params[4], params[5], params[6],
                                    mu[0], mu[1], mu[2], n_peak, buf),
      buf), 2, buf))

  return params


@cuda.jit(types.void(types.int32[:, :], types.float32[:],
                     types.float32[:, :, :], types.float32[:, :, :],
                     types.float32[:, :, :], types.float32, types.int32))
def fit_gpu(n_peak, x_data, y_data, params, mu, thresh, max_iter):
  """"""

  x, y = cuda.grid(2)
  # if x == 1 and y == 3:
  #   from pdb import set_trace; set_trace()
  if x < params.shape[0] and y < params.shape[1]:
    buf_7 = cuda.local.array((7,), dtype=types.float32)
    gpu_array_copy(params[x, y], buf_7)
    gradient_descent(n_peak[x, y], x_data, y_data[x, y],
                     buf_7, mu[x, y], thresh, max_iter)
    gpu_array_copy(buf_7, params[x, y])


if __name__ == '__main__':

  size = 1

  x = np.linspace(0, 180, NB_ANGLES, dtype=np.float32)
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

  tpb = (16, 16)
  bpg = (int(math.ceil(p.shape[0] / tpb[0])),
         int(math.ceil(p.shape[1] / tpb[1])))
  n_gpu = cuda.to_device(n)
  x_gpu = cuda.to_device(np.radians(x))
  y_gpu = cuda.to_device(y)
  p_gpu = cuda.to_device(p)
  m_gpu = cuda.to_device(m)
  fit_gpu[bpg, tpb](n_gpu, x_gpu, y_gpu, p_gpu, m_gpu, 1e-8, 300)  # 50000)

  p = p_gpu.copy_to_host()

  from workflow_gabor import periodic_gaussian_3, fit_curve_newton
  p_2, _ = fit_curve_newton(m_2,
                            y.astype(np.float64),
                            np.radians(x).astype(np.float64),
                            p_2[:, :, 1:6:2],
                            p_2[:, :, 0:6:2],
                            p_2[:, :, -1],
                            1e-8,
                            300)

  plt.figure()
  plt.plot(x, y[0, 0])
  plt.plot(x, periodic_gaussian_3(np.radians(x), *p[0, 0], *m[0, 0]))
  plt.plot(x, periodic_gaussian_3(np.radians(x), *p_2[0, 0], *m_2[0, 0]))
  plt.show()
