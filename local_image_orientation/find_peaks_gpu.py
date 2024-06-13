# coding: utf-8

from numba import cuda
from numba import types
import cupy as cp
import math


def find_peaks(gabor_data_cpu, ang_cpu):
  """"""

  gabor_data = cp.asarray(gabor_data_cpu, dtype=cp.float32)
  ang = cp.asarray(ang_cpu, dtype=cp.float32)
  del gabor_data_cpu

  tpb = (8, 8, 8)
  bpg = (int(math.ceil(gabor_data.shape[0] / tpb[0])),
         int(math.ceil(gabor_data.shape[1] / tpb[1])),
         int(math.ceil(gabor_data.shape[2] / tpb[2])))

  min_idx = cp.argmin(gabor_data, axis=2, dtype=cp.int32)
  to_search = cp.empty_like(gabor_data, dtype=cp.float32)
  rearrange(gabor_data, min_idx, to_search)

  peaks = cp.empty_like(gabor_data, dtype=cp.int32)
  local_maxima_gpu[bpg, tpb](to_search, peaks)

  peak_mask = cp.empty_like(gabor_data, dtype=cp.int32)
  make_peak_mask(peaks, peak_mask)
  del peaks

  prominences = cp.empty_like(gabor_data, dtype=cp.float32)
  left_bases = cp.empty_like(gabor_data, dtype=cp.int32)
  right_bases = cp.empty_like(gabor_data, dtype=cp.int32)

  peak_prominences_gpu[bpg, tpb](to_search, peak_mask, prominences, left_bases,
                                 right_bases)

  min_val = cp.min(gabor_data, axis=2)

  min_prominence = 0.05 * (cp.max(gabor_data, axis=2) - min_val)
  prominence_mask = prominences < min_prominence
  peak_mask[prominence_mask] = -1
  prominences[prominence_mask] = -1
  del min_prominence, prominence_mask

  widths = cp.empty_like(gabor_data, dtype=cp.float32)
  width_heights = cp.empty_like(gabor_data, dtype=cp.float32)

  peak_width_gpu(to_search, peak_mask, prominences, left_bases, right_bases,
                 widths, width_heights)

  del left_bases, right_bases

  heights = to_search - min_val
  widths *= cp.radians(ang[1] - ang[0])
  width_heights -= min_val

  sorted_order = cp.argsort(prominences, axis=2)
  del prominences

  heights_final = cp.take_along_axis(heights, sorted_order, axis=2)[:, :, :3]
  heights_final[heights_final < 0] = 1
  del heights
  widths_final = cp.take_along_axis(widths, sorted_order, axis=2)[:, :, :3]
  widths_final[widths_final < 0] = 1
  del widths
  width_heights_final = cp.take_along_axis(width_heights, sorted_order,
                                           axis=2)[:, :, :3]
  width_heights_final[width_heights_final < 0] = 1
  del width_heights

  deviation_final = widths_final / (2 * cp.sqrt(cp.log(heights_final /
                                                       width_heights_final)))
  del width_heights_final, widths_final

  peak_mask_final = cp.take_along_axis(peak_mask, sorted_order,
                                       axis=2)[:, :, :3]
  del peak_mask

  peak_index = sorted_order[:, :, :3]
  peak_index_final = (peak_index + min_idx) % to_search.shape[2]
  del peak_index, min_idx, to_search, sorted_order

  peak_value_final = ang[peak_index_final]

  invalid_peak_mask_final = peak_mask_final < 0
  del peak_mask_final

  peak_index_final[invalid_peak_mask_final] = cp.nan
  deviation_final[invalid_peak_mask_final] = cp.nan
  heights_final[invalid_peak_mask_final] = cp.nan
  del invalid_peak_mask_final

  params = cp.full((*gabor_data.shape[:2], 7), -1, dtype=cp.float32)
  params[:, :, -1] = min_val
  del min_val
  params[:, :, 0:6:2] = deviation_final
  del deviation_final
  params[:, :, 1:6:2] = heights_final
  del heights_final

  return peak_index_final.get(), peak_value_final.get(), params.get()


@cuda.jit(types.void(types.float32[:, :, :], types.int32[:, :],
                     types.float32[:, :, :]))
def rearrange(array_in, min_idx, array_out):
  """"""

  x, y = cuda.grid(2)
  if x < array_in.shape[0] and y < array_in.shape[1]:
    dim = array_in.shape[2]
    idx = min_idx[x, y]
    # todo
    array_out[x, y, :dim - idx] = array_in[x, y, idx:]
    array_out[x, y, dim - idx:] = array_in[x, y, :idx]


@cuda.jit(types.void(types.int32[:, :, :], types.int32[:, :, :]))
def make_peak_mask(peaks, mask):
  """"""

  x, y, z = cuda.grid(3)
  if x < peaks.shape[0] and y < peaks.shape[1] and z < peaks.shape[2]:
    peak_idx = peaks[x, y, z]
    if peak_idx > 0:
      mask[x, y, peak_idx] = 1


@cuda.jit(types.void(types.float32[:, :, :], types.int32[:, :, :]))
def local_maxima_gpu(data, midpoints):
  """"""

  x, y, z = cuda.grid(3)
  if x < data.shape[0] and y < data.shape[1] and z < data.shape[2] - 2:

    z_max = data.shape[2] - 1
    midpoint = -1

    if data[x, y, z] < data[x, y, z + 1]:
      ahead = z + 2

      while (ahead < z_max and
             data[x, y, ahead] == data[x, y, z + 1]):
        ahead += 1

      if data[x, y, ahead] < data[x, y, z + 1]:
        midpoint = (z + ahead) // 2

    midpoints[x, y, z] = midpoint


@cuda.jit(types.void(types.float32[:, :, :], types.int32[:, :, :],
                     types.float32[:, :, :], types.int32[:, :, :],
                     types.int32[:, :, :]))
def peak_prominences_gpu(data, peak_mask, prominences, left_bases,
                         right_bases):
  """"""

  x, y, z = cuda.grid(3)
  if x < data.shape[0] and y < data.shape[1] and z < data.shape[2]:
    if peak_mask[x, y, z] > 0:

      i_min = 0
      i_max = data.shape[2] - 1

      left_bases[x, y, z] = z
      i = z
      left_min = data[x, y, z]

      while i_min <= i and data[x, y, i] <= data[x, y, z]:
        if data[x, y, i] < left_min:
          left_min = data[x, y, i]
          left_bases[x, y, z] = i
        i -= 1

      right_bases[x, y, z] = z
      i = z
      right_min = data[x, y, z]

      while i <= i_max and data[x, y, i] <= data[x, y, z]:
        if data[x, y, i] < right_min:
          right_min = data[x, y, i]
          right_bases[x, y, z] = i
        i += 1

      prominences[x, y, z] = data[x, y, z] - max(left_min, right_min)


@cuda.jit(types.void(types.float32[:, :, :], types.int32[:, :, :],
                     types.float32[:, :, :], types.int32[:, :, :],
                     types.int32[:, :, :], types.float32[:, :, :],
                     types.float32[:, :, :]))
def peak_width_gpu(data, peak_mask, prominences, left_bases, right_bases,
                   widths, width_heights):
  """"""

  x, y, z = cuda.grid(3)
  if x < data.shape[0] and y < data.shape[1] and z < data.shape[2]:
    if peak_mask[x, y, z] > 0:
      i_min = left_bases[x, y, z]
      i_max = right_bases[x, y, z]

      height = data[x, y, z] - prominences[x, y, z] * 0.5
      width_heights[x, y, z] = height

      i = z
      while i_min < i and height < data[x, y, i]:
        i -= 1

      left_ip = i
      if data[x, y, i] < height:
        left_ip += ((height - data[x, y, i]) /
                    (data[x, y, i + 1] - data[x, y, i]))

      i = z
      while i < i_max and height < data[x, y, i]:
        i += 1

      right_ip = i
      if data[x, y, i] < height:
        right_ip -= ((height - data[x, y, i]) /
                     (data[x, y, i - 1] - data[x, y, i]))

      widths[x, y, z] = right_ip - left_ip


if __name__ == '__main__':
  
  ...
