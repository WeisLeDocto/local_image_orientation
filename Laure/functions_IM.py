# coding: utf-8

import numpy as np
from scipy.ndimage.filters import gaussian_filter, sobel, median_filter
from scipy import signal
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from skimage.feature import structure_tensor


def side_effect(f, im, sigma=None):
  p = 10
  img = mirror_padding(im, p)
  print("Mirror padding")
  img1, img2, img3, img4 = image_cutting(img, p)
  del img
  print("Image cutting")
  if f == sobel:
    x1, y1, z1 = f(img1, 0), f(img1, 1), f(img1, 2)
    x2, y2, z2 = f(img2, 0), f(img2, 1), f(img2, 2)
    x3, y3, z3 = f(img3, 0), f(img3, 1), f(img3, 2)
    x4, y4, z4 = f(img4, 0), f(img4, 1), f(img4, 2)
  else:
    if sigma:
      x1, y1, z1 = f(img1, sigma)
      print("NR 1")
      x2, y2, z2 = f(img2, sigma)
      print("NR 2")
      x3, y3, z3 = f(img3, sigma)
      print("NR 3")
      x4, y4, z4 = f(img4, sigma)
      print("NR 4")
    else:
      x1, y1, z1 = f(img1)
      x2, y2, z2 = f(img2)
      x3, y3, z3 = f(img3)
      x4, y4, z4 = f(img4)
  x = image_float_colling(x1, x2, x3, x4, p)
  del x1, x2, x3, x4
  print("float colling x")
  y = image_float_colling(y1, y2, y3, y4, p)
  del y1, y2, y3, y4
  print("float colling y")
  z = image_float_colling(z1, z2, z3, z4, p)
  del z1, z2, z3, z4
  print("float colling z")
  return x, y, z


def structure_tensor_f(im, sigma, courb=False):
  f = median_filter
  imx = sobel(im, 0, mode='nearest')
  print('sobel x')
  imy = sobel(im, 1, mode='nearest')
  print('sobel y')
  imz = sobel(im, 2, mode='nearest')
  print('sobel z')

  if courb:
    imxx = gaussian_filter(sobel(sobel(im, 0, mode='nearest'), 0,
                                 mode='nearest'), sigma, mode='nearest')
    imxy = gaussian_filter(sobel(sobel(im, 0, mode='nearest'), 1,
                                 mode='nearest'), sigma, mode='nearest')
    imxz = gaussian_filter(sobel(sobel(im, 0, mode='nearest'), 2,
                                 mode='nearest'), sigma, mode='nearest')
    imyy = gaussian_filter(sobel(sobel(im, 1, mode='nearest'), 1,
                                 mode='nearest'), sigma, mode='nearest')
    imyz = gaussian_filter(sobel(sobel(im, 1, mode='nearest'), 2,
                                 mode='nearest'), sigma, mode='nearest')
    imzz = gaussian_filter(sobel(sobel(im, 2, mode='nearest'), 2,
                                 mode='nearest'), sigma, mode='nearest')
  else:
    imxz = f(imx * imz, sigma, mode='nearest')
    print("median filter xz")
    imyz = f(imy * imz, sigma, mode='nearest')
    print("median filter yz")
    imxy = f(imx * imy, sigma, mode='nearest')
    print("median filter xy")
    imxx = f(imx * imx, sigma, mode='nearest')
    print("median filter xx")
    imyy = f(imy * imy, sigma, mode='nearest')
    print("median filter xy")
    imzz = f(imz * imz, sigma, mode='nearest')
    print("median filter zz")

  del im

  trs2 = imxx * imxx + imyy * imyy + imzz * imzz + \
      2 * (imxy * imxy + imxz * imxz + imyz * imyz)
  del imxy, imxz, imyz
  print("TRS2")
  tr2s = imxx * imxx + imyy * imyy + imzz * imzz + \
      2 * (imxx * imyy + imxx * imzz + imyy * imzz)
  tr2s[tr2s == 0] = 1e-17
  del imxx, imyy, imzz
  print("TR2S")
  # ampl = np.zeros_like(trs2)
  ampl = np.sqrt(trs2/tr2s)
  del trs2, tr2s
  imx = f(imx, sigma, mode='nearest')
  imx[imx == 0] = 1e-17
  print("IMX")
  imy = f(imy, sigma, mode='nearest')
  imy[imy == 0] = 1e-17
  print("IMY")
  # theta, phi = np.full(imx.shape, 90), np.full(imx.shape, 90)
  theta = np.degrees(np.arctan(imy/imx))
  w = (imx * imx + imy * imy) ** 0.5
  del imx, imy
  print('W')
  imz = f(imz, sigma, mode='nearest')
  phi = np.degrees(np.arctan(imz/w))
  del imz, w
  print("IMZ")
  return theta, phi, ampl


def algo_gaussian(im, sigma):
  imx = gaussian_filter(sobel(im, 0, mode='nearest'), sigma, mode='nearest')
  imy = gaussian_filter(sobel(im, 1, mode='nearest'), sigma, mode='nearest')
  imz = gaussian_filter(sobel(im, 2, mode='nearest'), sigma, mode='nearest')
  return imx, imy, imz


def algo_median(im, sigma):
  imx = median_filter(sobel(im, 0, mode='nearest'), sigma, mode='nearest')
  imy = median_filter(sobel(im, 1, mode='nearest'), sigma, mode='nearest')
  imz = median_filter(sobel(im, 2, mode='nearest'), sigma, mode='nearest')
  return imx, imy, imz


def dog(im, s1, s2):
  im1 = gaussian_filter(im, s1)
  im2 = gaussian_filter(im, s2)
  return im1 - im2


def distribution_1d(ampl, theta, phi):
  ampl_ = ampl.flatten()
  theta_ = np.round((theta+90).flatten()).astype('int')
  phi_ = np.round((phi+90).flatten()).astype('int')
  distrib_theta = np.zeros(181)
  distrib_phi = np.zeros(181)
  tuple_theta = list(zip(theta_, ampl_))
  # tuple_phi = list(zip(theta_, ampl_)) ??
  tuple_phi = list(zip(phi_, ampl_))
  for i in range(len(tuple_theta)):
    distrib_theta[tuple_theta[i][0]] += tuple_theta[i][1]
    tuple_phi[tuple_phi[i][0]] += tuple_phi[i][1]
  distrib_theta = distrib_theta / np.amax(distrib_theta)
  distrib_phi = distrib_phi / np.amax(distrib_phi)
  angle = np.arange(-90, 91, 1)
  plt.figure()
  plt.plot(angle, distrib_theta, label='Theta')
  plt.plot(angle, distrib_phi, label='Phi')
  plt.legend()
  plt.grid()
  plt.xlim(-90, 90)
  plt.xticks([-90, -45, 0, 45, 90])
  plt.xlabel('Orientation')
  plt.ylabel('Amplitude')
  return distrib_theta, distrib_phi


def distribution_2d(ampl: np.ndarray, theta, phi):
  ampl_ = ampl.flatten()
  theta_ = np.round((theta+90).flatten()).astype('int')
  phi_ = np.round((phi+90).flatten()).astype('int')
  aniso = np.zeros((181, 181))
  tuple_ = list(zip(theta_, phi_, ampl_))
  for i in range(len(tuple_)):
    aniso[tuple_[i][0], tuple_[i][1]] += tuple_[i][2]
  aniso = aniso / np.amax(aniso)
  plt.figure()
  plt.imshow(aniso, cmap='hot', interpolation='nearest',
             extent=[-90, 90, 90, -90])
  plt.xticks([-90, -45, 0, 45, 90])
  plt.yticks([-90, -45, 0, 45, 90])
  plt.colorbar()
  plt.xlabel('Phi')
  plt.ylabel('Theta')
  return aniso


def mirror_padding(im, p):
  im112 = np.flip(np.flip(im[:p, -p:, :], 0), 1)
  im212 = np.flip(im[:, -p:, :], 1)
  im312 = np.flip(np.flip(im[-p:, -p:, :], 0), 1)
  im122 = np.flip(im[:p, :, :], 0)
  im222 = im
  im322 = np.flip(im[-p:, :, :], 0)
  im132 = np.flip(np.flip(im[:p, :p, :], 0), 1)
  im232 = np.flip(im[:, :p, :], 1)
  im332 = np.flip(np.flip(im[-p:, :p, :], 0), 1)
  im2 = np.concatenate((np.concatenate((im132, im232, im332), 0),
                        np.concatenate((im122, im222, im322), 0),
                        np.concatenate((im112, im212, im312), 0)), 1)
  del im112, im212, im312, im122, im222, im322, im132, im232, im332
  print('IM2')
  im221 = np.flip(im[:, :, :p], 2)
  im111 = np.flip(np.flip(im221[:p, -p:, :], 0), 1)
  im211 = np.flip(im221[:, -p:, :], 1)
  im311 = np.flip(np.flip(im221[-p:, -p:, :], 0), 1)
  im121 = np.flip(im221[:p, :, :], 0)
  im321 = np.flip(im221[-p:, :, :], 0)
  im131 = np.flip(np.flip(im221[:p, :p, :], 0), 1)
  im231 = np.flip(im221[:, :p, :], 1)
  im331 = np.flip(np.flip(im221[-p:, :p, :], 0), 1)
  im1 = np.concatenate((np.concatenate((im131, im231, im331), 0),
                        np.concatenate((im121, im221, im321), 0),
                        np.concatenate((im111, im211, im311), 0)), 1)
  del im111, im211, im311, im121, im221, im321, im131, im231, im331
  print('IM1')
  im223 = np.flip(im[:, :, -p:], 2)
  im113 = np.flip(np.flip(im223[:p, -p:, :], 0), 1)
  im213 = np.flip(im223[:, -p:, :], 1)
  im313 = np.flip(np.flip(im223[-p:, -p:, :], 0), 1)
  im123 = np.flip(im223[:p, :, :], 0)
  im323 = np.flip(im223[-p:, :, :], 0)
  im133 = np.flip(np.flip(im223[:p, :p, :], 0), 1)
  im233 = np.flip(im223[:, :p, :], 1)
  im333 = np.flip(np.flip(im223[-p:, :p, :], 0), 1)
  im3 = np.concatenate((np.concatenate((im133, im233, im333), 0),
                        np.concatenate((im123, im223, im323), 0),
                        np.concatenate((im113, im213, im313), 0)), 1)
  del im113, im213, im313, im123, im223, im323, im133, im233, im333
  print('IM3')
  im = np.concatenate((im1, im2, im3), 2)
  del im1, im2, im3
  print("IM")
  im = im.astype('float64')
  print("IM ASTYPE")
  return im


def image_cutting(im, p):
  a = np.shape(im)[0]
  b = np.shape(im)[1]
  im1 = im[:int(a/2)+p, :int(b/2)+p, :]
  im2 = im[int(a/2)-p:, :int(b/2)+p, :]
  im3 = im[:int(a/2)+p, int(b/2)-p:, :]
  im4 = im[int(a/2)-p:, int(b/2)-p:, :]
  return im1, im2, im3, im4


def image_float_colling(im1, im2, im3, im4, p):
  a = np.shape(im1)[0]
  b = np.shape(im1)[1]
  aa = (a - 2 * p) * 2
  bb = (b - 2 * p) * 2
  im = np.zeros((aa, bb, np.shape(im1)[2] - 2 * p))
  im1 = im1[p:a-p, p:b-p, p:-p]
  im2 = im2[p:a-p, p:b-p, p:-p]
  im3 = im3[p:a-p, p:b-p, p:-p]
  im4 = im4[p:a-p, p:b-p, p:-p]
  im[0:aa / 2, 0:bb / 2, :] = im1
  im[aa / 2:, 0:bb / 2, :] = im2
  im[0:aa / 2, bb / 2:, :] = im3
  im[aa / 2:, bb / 2:, :] = im4
  return im


def scatter3d(x, y, z, w):
  t = x
  x = z
  z = t
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_aspect('equal')
  scat = ax.scatter(x, y, z, c=w, s=20, cmap='Spectral_r')
  # ax.set_xlabel('X axis')
  # ax.set_ylabel('Y axis')
  # ax.set_zlabel('Z axis')
  ax.set_xlabel('Z axis')
  ax.set_ylabel('Y axis')
  ax.set_zlabel('X axis')
  ax.view_init(elev=20, azim=45.)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])
  # Get rid of the panes
  ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  fig.colorbar(scat, shrink=0.5, aspect=5)


def representation_3d(ampl, theta):
  tol = 0.2
  theta_ = theta[ampl > tol]
  ampl_ = ampl
  ampl_[ampl < tol] = 0
  index = np.argwhere(ampl_)
  taille = 3
  indexf = np.zeros((index.shape[0] / taille, 3))
  thetaf = np.zeros((index.shape[0] / taille))
  for i in range(len(thetaf)):
    indexf[i] = index[i + taille]
    thetaf[i] = theta_[i + taille]
  scatter3d(index[:, 0], index[:, 1], index[:, 2], theta_)


def fonction_skimage_3d(im, sigma):
  imx = sobel(im, 0)
  imy = sobel(im, 1)
  imz = sobel(im, 2)
  imxx = gaussian_filter(imx*imx, sigma).flatten()
  imxy = gaussian_filter(imx*imy, sigma).flatten()
  imxz = gaussian_filter(imx*imz, sigma).flatten()
  del imx
  imyy = gaussian_filter(imy*imy, sigma).flatten()
  imyz = gaussian_filter(imy*imz, sigma).flatten()
  del imy
  imzz = gaussian_filter(imz*imz, sigma).flatten()
  del imz
  x = y = z = np.zeros_like(imxx, float)
  for i in range(len(imxx)):
    m = np.asarray([[imxx[i], imxy[i], imxz[i]],
                    [imxy[i], imyy[i], imyz[i]],
                    [imxz[i], imyz[i], imzz[i]]])
    w, v = eigh(m)
    x[i] = v[0, 2]
    y[i] = v[1, 2]
    z[i] = v[2, 2]
  return x.reshape((im.shape[0], im.shape[1], im.shape[2])), \
      y.reshape((im.shape[0], im.shape[1], im.shape[2])), \
      z.reshape((im.shape[0], im.shape[1], im.shape[2]))


def fonction_skimage_2d(im, sigma):
  axx, axy, ayy = structure_tensor(im[:, :, im.shape[2]/4], sigma)
  axx = axx.flatten()
  axy = axy.flatten()
  ayy = ayy.flatten()
  x = y = np.zeros_like(axx)
  for i in range(len(axx)):
    m = np.asarray([[axx[i], axy[i]], [axy[i], ayy[i]]])
    w, v = eigh(m)
    x[i] = v[1, 1]
    y[i] = v[0, 1]
  return x.reshape(im.shape[0], im.shape[1]), \
      y.reshape(im.shape[0], im.shape[1])


def sobel_maison(im, ordre):
  if ordre == 1:
    v1 = [-1, 0, 1]
    v2 = [1, 2, 1]
    v3 = [1, 2, 1]
  elif ordre == 2:
    v1 = [-2, -1, 0, 1, 2]
    v2 = [5, 8, 10, 8, 5]
    v3 = [5, 8, 10, 8, 5]
  else:
    return None, None, None
  gx = np.tensordot(np.tensordot(v1, v2, 0), v3, 0)
  gy = np.tensordot(np.tensordot(v2, v1, 0), v3, 0)
  gz = np.tensordot(np.tensordot(v2, v3, 0), v1, 0)
  x = signal.convolve(im, gx, mode='same')
  y = signal.convolve(im, gy, mode='same')
  z = signal.convolve(im, gz, mode='same')
  return x, y, z


def distribution_2d_comment(ampl, theta, phi):
  ampl_f = np.round(ampl*100).flatten().astype('int')
  theta_f = np.round(theta).flatten().astype('int')
  phi_f = np.round(phi).flatten().astype('int')
  shape_ = ampl_f.shape[0]
  aniso = np.zeros((181, 181))
  for i in range(0, 10):
    ampl_ = ampl_f[i*shape_/10: (i+1)*shape_/10]
    theta_ = theta_f[i*shape_/10: (i+1)*shape_/10]
    phi_ = phi_f[i*shape_/10: (i+1)*shape_/10]
    tuple_aniso = list(zip(theta_, phi_, ampl_))
    aniso_ = np.zeros((181, 181))
    for j in range(0, len(tuple_aniso)):
      aniso_[tuple_aniso[j][0]+90, tuple_aniso[j][1]+90] += tuple_aniso[j][2]
      aniso += aniso_
  for i in range(0, 10):
    ampl_ = ampl_f[10*shape_/10: shape_ % 10]
    theta_ = theta_f[10*shape_/10: shape_ % 10]
    phi_ = phi_f[10*shape_/10: shape_ % 10]
    tuple_aniso = list(zip(theta_, phi_, ampl_))
    aniso_ = np.zeros((181, 181))
    for j in range(0, len(tuple_aniso)):
      aniso_[tuple_aniso[j][0]+90, tuple_aniso[j][1]+90] += tuple_aniso[j][2]
      aniso += aniso_
  # aniso = aniso / aniso.max()
  plt.figure()
  plt.imshow(aniso, cmap='hot', interpolation='nearest',
             extent=[-90, 90, 90, -90])
  plt.xticks([-90, -45, 0, 45, 90])
  plt.yticks([-90, -45, 0, 45, 90])
  plt.colorbar()
  plt.xlabel('Phi')
  plt.ylabel('Theta')
  # plt.clim(0, 0.5)
  return aniso


def distribution_1d_comment(ampl, theta, phi):
  ampl_f = np.round(ampl*10).flatten().astype('int')
  theta_f = np.round(theta).flatten().astype('int')
  phi_f = np.round(phi).flatten().astype('int')
  distrib_theta = np.zeros(181)
  distrib_phi = np.zeros(181)
  shape_ = ampl_f.shape[0]
  for i in range(10):
    ampl_ = ampl_f[i*shape_/10: (i+1)*shape_/10]
    theta_ = theta_f[i*shape_/10: (i+1)*shape_/10]
    phi_ = phi_f[i*shape_/10: (i+1)*shape_/10]
    tuple_theta = list(zip(theta_, ampl_))
    tuple_phi = list(zip(phi_, ampl_))
    del theta_, phi_, ampl_
    distrib_t = np.zeros(181)
    distrib_p = np.zeros(181)
    for j in range(len(tuple_theta)):
      distrib_t[tuple_theta[j][0]+90] += tuple_theta[j][1]
      distrib_p[tuple_phi[j][0]+90] += tuple_phi[j][1]
      distrib_theta += distrib_t
      distrib_phi += distrib_p
    del distrib_t, distrib_p

  ampl_ = ampl_f[10*shape_/10: shape_ % 10]
  theta_ = theta_f[10*shape_/10: shape_ % 10]
  phi_ = phi_f[10*shape_/10: shape_ % 10]
  tuple_theta = list(zip(theta_, ampl_))
  tuple_phi = list(zip(phi_, ampl_))
  del theta_, ampl_, phi_
  distrib_t = np.zeros(181)
  distrib_p = np.zeros(181)
  for j in range(len(tuple_theta)):
    distrib_t[tuple_theta[j][0]+90] += tuple_theta[j][1]
    distrib_p[tuple_phi[j][0]+90] += tuple_phi[j][1]
    distrib_theta += distrib_t
    distrib_phi += distrib_p
  del distrib_t, distrib_p
  distrib_theta = distrib_theta / np.amax(distrib_theta)
  distrib_phi = distrib_phi / np.amax(distrib_phi)
  angle = np.arange(-90, 91, 1)
  plt.figure()
  plt.plot(angle, distrib_theta, label='Theta')
  plt.plot(angle, distrib_phi, label='Phi')
  plt.legend()
  plt.grid()
  plt.xlim(-90, 90)
  plt.xticks([-90, -45, 0, 45, 90])
  plt.xlabel('Orientation')
  plt.ylabel('Amplitude')
  return distrib_theta, distrib_phi
