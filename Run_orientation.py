# coding: utf-8

from local_image_orientation import orientation
from pathlib import Path

if __name__ == "__main__":
  orientation(Path(__file__).parent / 'npy' / 'demo.npy',
              save_folder=Path(__file__).parent / 'svg')
