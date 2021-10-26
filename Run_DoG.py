# coding: utf-8

from sys import argv, exit
from PyQt5.QtWidgets import QApplication
from difference_of_gaussians import DoG_interface

if __name__ == "__main__":
  app = QApplication(argv)
  DoG_interface()
  exit(app.exec_())
