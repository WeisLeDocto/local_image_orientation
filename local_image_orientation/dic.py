# coding: utf-8

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
from re import fullmatch

matplotlib.use('TkAgg')
plt.ioff()


def cast(img, pers):

  up = np.percentile(img, 100 - pers)
  down = np.percentile(img, pers)
  img[img > up] = up
  img[img < down] = down

  return (255 * (img - img.min()) / (img.max() - img.min())).astype('uint8')


def correct_image(image, flow):
  U = flow[::, ::, 0]
  V = flow[::, ::, 1]
  Y, X = np.meshgrid(np.arange(0, U.shape[1]), np.arange(0, U.shape[0]))
  X = X.astype(np.float32)
  Y = Y.astype(np.float32)
  return cv2.remap(image, Y + U, X + V, cv2.INTER_LINEAR)


def sort_images(path: Path):
  """"""

  sec, = fullmatch(r'(\d+)_\d+\.npy', path.name).groups()
  return int(sec)


if __name__ == '__main__':

  base_path = Path('/home/antoine/Documents/HDR/Fascia_Laure_1/hdr')
  correl_path = base_path.parent / 'correl'

  correl_path.mkdir(exist_ok=True, parents=False)
  iter_path = tuple(sorted(base_path.glob('*.npy'), key=sort_images))

  dis = cv2.DISOpticalFlow().create()
  dis.setPatchSize(16)
  dis.setFinestScale(0)
  dis.setPatchStride(4)
  dis.setVariationalRefinementAlpha(4)
  dis.setVariationalRefinementDelta(1)
  dis.setVariationalRefinementEpsilon(2e-1)
  dis.setVariationalRefinementGamma(0)
  dis.setVariationalRefinementIterations(30)

  img0 = cast(np.load(iter_path[0]), 0)
  flow_init = np.zeros((*img0.shape, 2))

  cv2.imwrite(str(correl_path / f'{iter_path[0].stem}.tiff'), img0)

  for img_path in tqdm(iter_path[1:],
                       total=len(iter_path[1:]),
                       desc='DIS',
                       file=sys.stdout,
                       colour='green'):

    img = cast(np.load(img_path), 0)
    flow = dis.calc(img0, img, flow_init)
    flow_init = flow.copy()

    img_remap = correct_image(img, flow)

    cv2.imwrite(str(correl_path / f'{img_path.stem}.tiff'), img)
    cv2.imwrite(str(correl_path / f'{img_path.stem}_remap.tiff'), img_remap)
    np.save(correl_path / f'{img_path.stem}_flow.npy', flow)

  if False:

    img0 = np.load('/home/antoine/Documents/HDR/Fascia_Ina_1/hdr/0_2377.npy')
    img1 = np.load('/home/antoine/Documents/HDR/Fascia_Ina_1/hdr/1_2421.npy')

    img0 = cast(img0, 0)
    img1 = cast(img1, 0)

    dis = cv2.DISOpticalFlow().create()
    dis.setPatchSize(16)
    dis.setFinestScale(0)
    dis.setPatchStride(4)
    dis.setVariationalRefinementAlpha(4)
    dis.setVariationalRefinementDelta(1)
    dis.setVariationalRefinementEpsilon(2e-1)
    dis.setVariationalRefinementGamma(0)
    dis.setVariationalRefinementIterations(30)

    flow = dis.calc(img0, img1, None)

    e_xy, e_xx = np.gradient(flow[:, :, 0])
    e_yy, e_yx = np.gradient(flow[:, :, 1])

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(e_yx + e_xy, clim=(-0.3, 0.3))
    plt.subplot(2, 2, 2)
    plt.imshow(e_yx - e_xy, clim=(-0.3, 0.3))
    plt.subplot(2, 2, 3)
    plt.imshow(e_xx, clim=(-0.3, 0.3))
    plt.subplot(2, 2, 4)
    plt.imshow(e_yy, clim=(-0.3, 0.3))
    plt.show()

    plt.figure()
    plt.imshow(correct_image(img1, flow).astype(np.float32) -
               img0.astype(np.float32), clim=(-20, 20), cmap='grey')
    plt.show()

    plt.figure()
    plt.imshow(flow[:, :, 1])
    plt.show()
