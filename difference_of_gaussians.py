# coding: utf-8

from functools import partial
from pathlib import Path
from numpy import load, uint8, save, flip, concatenate, eye, array, newaxis
from scipy.ndimage.filters import gaussian_filter
from dask.array import from_array, subtract, map_overlap
from dask.distributed import Client, LocalCluster

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QStyle
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QSize, QObject, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread


class DoG(QObject):
  """Computes a difference of gaussians in a separate thread."""

  finished = pyqtSignal()

  def __init__(self,
               interface: QMainWindow) -> None:
    """Initializes the class.

    Args:
      interface: The parent QMainWindow object.
    """

    super().__init__()
    self.interface = interface

  def run(self,
          s1: float,
          s2: float) -> None:
    """Computes the difference of gaussians and updates the display.

    Args:
      s1: Sigma 1
      s2: Sigma 2
    """

    self.interface.dog(s1, s2)
    self.interface.update_image()
    self.finished.emit()


class DoG_interface_3D(QMainWindow):
  """An interface for easily visualizing the difference of gaussians on a 3D
  image with various sigma values."""

  def __init__(self,
               max_sigma: float = 5) -> None:
    """Loads the image, sets the interface and computes a first DoG.

    Args:
      max_sigma: The maximum settable value for the standard deviation.
    """

    super().__init__()
    self._max_sigma = max_sigma

    # Choosing the file to open
    path = Path(__file__).parent / 'npy'
    file = QFileDialog.getOpenFileName(caption="Select the 3D image to open",
                                       directory=str(path),
                                       options=QFileDialog.DontUseNativeDialog,
                                       filter='*.npy',
                                       initialFilter='*.npy')[0]

    # If no file was selected, displaying a black image and disabling commands
    if file == '':
      self._img = concatenate(array([(eye(500) + flip(eye(500), axis=0))
                                     [:, :, newaxis]] * 15), axis=2)
      QMessageBox(QMessageBox.Warning, 'No file selected',
                  'No file selected !\nAll commands are disabled.').exec()
      self._enable = False
    else:
      self._file = file
      self._img = load(file)
      self._enable = True

    # Starting the dask client
    cluster = LocalCluster(processes=False,
                           memory_limit='10GB',
                           silence_logs=40)
    Client(cluster)

    # Applying the first filter on the image
    self._img_filtered = self._img
    self.dog(33 * self._max_sigma / 100, 66 * self._max_sigma / 100)

    # Building the graphical interface
    self._set_layout()
    self._set_connections()
    self.move(int((QDesktopWidget().availableGeometry().width() -
                   self.frameGeometry().width()) / 2),
              self.frameGeometry().top())
    self.show()

  def _set_layout(self) -> None:
    """Sets the graphical interface layout."""

    # General layout
    self.setWindowTitle('Difference of Gaussians Interface')
    self._general_layout = QVBoxLayout()
    self._central_widget = QWidget(self)
    self.setCentralWidget(self._central_widget)
    self._central_widget.setLayout(self._general_layout)

    self._setting_layout = QHBoxLayout()
    self._slider_sigma_layout = QVBoxLayout()
    self._slider_1_layout = QHBoxLayout()
    self._slider_2_layout = QHBoxLayout()

    # Widget containing the image to display
    self._label = QLabel()
    height, width, depth = self._img_filtered.shape
    q_img = QImage(self._img_filtered[:, :, int(depth / 2)].astype(uint8),
                   width, height, width, QImage.Format_Grayscale8)
    self._image = QPixmap(q_img)
    self._image = self._image.scaledToHeight(int(
        QDesktopWidget().availableGeometry().height() - 250))
    self._label.setPixmap(self._image)
    self._label.setStyleSheet("border: 1px solid black;")
    self._general_layout.addWidget(self._label)

    # Slider for the slice selection
    self._general_layout.addWidget(QLabel("Slice selection :"))
    self._slider_layout = QHBoxLayout()
    self._slider_layout.addWidget(QLabel("0"))
    self._slider = QSlider(Qt.Horizontal)
    self._slider.setMinimum(0)
    self._slider.setMaximum(depth - 1)
    self._slider.setValue(int(depth / 2))
    self._slider_layout.addWidget(self._slider)
    self._slider_layout.addWidget(QLabel(str(self._img.shape[2])))
    self._general_layout.addLayout(self._slider_layout)

    # Slider for choosing sigma 1
    self._slider_1_layout.addWidget(QLabel("Sigma 1 :"))
    self._slider_1_layout.addWidget(QLabel(" "))
    self._slider_1_label = QLabel("{:.2f}".format(33 * self._max_sigma / 100))
    self._slider_1_layout.addWidget(self._slider_1_label)
    self._slider_1_layout.addWidget(QLabel(" "))
    self._slider_1_layout.addWidget(QLabel("0.00"))
    self._slider_1 = QSlider(Qt.Horizontal)
    self._slider_1.setMinimum(0)
    self._slider_1.setMaximum(100)
    self._slider_1.setValue(33)
    self._slider_1_layout.addWidget(self._slider_1)
    self._slider_1_layout.addWidget(QLabel("{:.2f}".format(self._max_sigma)))
    self._slider_1_layout.addWidget(QLabel(" "))

    # Slider for choosing sigma 1
    self._slider_2_layout.addWidget(QLabel("Sigma 2 :"))
    self._slider_2_layout.addWidget(QLabel(" "))
    self._slider_2_label = QLabel("{:.2f}".format(66 * self._max_sigma / 100))
    self._slider_2_layout.addWidget(self._slider_2_label)
    self._slider_2_layout.addWidget(QLabel(" "))
    self._slider_2_layout.addWidget(QLabel("0.00"))
    self._slider_2 = QSlider(Qt.Horizontal)
    self._slider_2.setMinimum(0)
    self._slider_2.setMaximum(100)
    self._slider_2.setValue(66)
    self._slider_2_layout.addWidget(self._slider_2)
    self._slider_2_layout.addWidget(QLabel("{:.2f}".format(self._max_sigma)))
    self._slider_2_layout.addWidget(QLabel(" "))

    self._slider_sigma_layout.addLayout(self._slider_1_layout)
    self._slider_sigma_layout.addLayout(self._slider_2_layout)
    self._setting_layout.addLayout(self._slider_sigma_layout)

    # Button for saving the image
    self._save_button = QPushButton("Save image")
    self._setting_layout.addWidget(self._save_button)
    self._save_button.setIcon(self.style().standardIcon(
      QStyle.SP_DialogSaveButton))
    self._save_button.setIconSize(QSize(12, 12))

    self._general_layout.addLayout(self._setting_layout)

    # Disabling the sliders and buttons if no image was selected
    self._slider_1.setEnabled(self._enable)
    self._slider_2.setEnabled(self._enable)
    self._slider.setEnabled(self._enable)
    self._save_button.setEnabled(self._enable)

  def _set_connections(self) -> None:
    """Sets the actions triggered by the sliders and the button."""

    # The window doesn't show when no connection uses partial
    self._save_button.clicked.connect(partial(self._pass, None))
    # Saves the image
    self._save_button.clicked.connect(self._save_image)

    # this keeps sigma 1 > sigma 2
    self._slider_1.valueChanged.connect(self._slider_1_management)
    self._slider_2.valueChanged.connect(self._slider_2_management)

    # Triggers DoG calculation when releasing a slider
    self._slider_1.sliderReleased.connect(self._handle_dog)
    self._slider_2.sliderReleased.connect(self._handle_dog)

    # Updates the display with the current slide
    self._slider.valueChanged.connect(self.update_image)

  def _slider_1_management(self) -> None:
    """Keeps sigma 1 < sigma 2 and updates the displayed value."""

    if self._slider_1.value() > self._slider_2.value():
      self._slider_1.setValue(self._slider_2.value())
    self._slider_1_label.setText("{:.2f}".format(self._slider_1.value() / 100 *
                                                 self._max_sigma))

  def _slider_2_management(self) -> None:
    """Keeps sigma 2 > sigma 1 and updates the displayed value."""

    if self._slider_1.value() > self._slider_2.value():
      self._slider_2.setValue(self._slider_1.value())
    self._slider_2_label.setText("{:.2f}".format(self._slider_2.value() / 100 *
                                                 self._max_sigma))

  def update_image(self) -> None:
    """Updates the image after computing the DoG or changing the current
    slice."""

    # Generating a QImage object containing the image
    height, width, depth = self._img_filtered.shape
    q_img = QImage(self._img_filtered[:, :, self._slider.value()].astype(uint8),
                   width, height, width, QImage.Format_Grayscale8)
    self._image = QPixmap(q_img)
    # Rescaling the image to keep a constant window size
    self._image = self._image.scaledToHeight(int(
        QDesktopWidget().availableGeometry().height() - 250))
    self._label.setPixmap(self._image)

  def _handle_dog(self) -> None:
    """runs the DoG in a separate thread and disables the commands while the
    calculation is running."""

    # Disabling the sliders and button
    self._slider_1.setEnabled(False)
    self._slider_2.setEnabled(False)
    self._slider.setEnabled(False)
    self._save_button.setEnabled(False)

    # Creating a _thread for running the DoG calculation in parallel
    self._thread = QThread()
    self._worker = DoG(self)
    self._worker.moveToThread(self._thread)
    self._thread.started.connect(partial(self._worker.run,
                                         self._slider_1.value() / 100 *
                                         self._max_sigma,
                                         self._slider_2.value() / 100 *
                                         self._max_sigma))
    self._worker.finished.connect(self._thread.quit)
    self._worker.finished.connect(self._worker.deleteLater)
    self._thread.finished.connect(self._thread.deleteLater)

    self._thread.start()

    # The sliders and buttons are only re-enabled once the _thread finishes
    self._thread.finished.connect(lambda: self._slider_1.setEnabled(True))
    self._thread.finished.connect(lambda: self._slider_2.setEnabled(True))
    self._thread.finished.connect(lambda: self._slider.setEnabled(True))
    self._thread.finished.connect(lambda: self._save_button.setEnabled(True))

  def dog(self,
          sigma_1: float,
          sigma_2: float) -> None:
    """Computes the DoG using dask."""

    img = from_array(self._img)
    img_filtered = subtract(map_overlap(gaussian_filter, img,
                                        sigma=sigma_1, depth=15),
                            map_overlap(gaussian_filter, img,
                                        sigma=sigma_2, depth=15))
    img_filtered = 255 * (img_filtered - img_filtered.min()) / \
        (img_filtered.max() - img_filtered.min())
    self._img_filtered = img_filtered.compute()

  def _save_image(self) -> None:
    """Saves the image to the desired location and confirms it to the user."""

    # Generating a default name for the file
    path = Path(self._file).parent / (Path(self._file).name.replace('.npy', '')
                                      + '_' + "{:.2f}".format(
        self._slider_1.value() / 100 * self._max_sigma) + '_' + "{:.2f}".format(
        self._slider_2.value() / 100 * self._max_sigma) + '.npy')
    # Choosing the path to write the image to
    file = QFileDialog.getSaveFileName(caption="Create a file to save the "
                                               "image to",
                                       directory=str(path),
                                       options=QFileDialog.DontUseNativeDialog,
                                       filter='*.npy',
                                       initialFilter='*.npy')[0]
    # If no file selected, aborting
    if file == '':
      return
    # Adding the extension if not already there
    if not file.endswith('.npy'):
      file += '.npy'

    save(file, self._img_filtered)

    # Checking whether the file was successfully written, and displaying the
    # corresponding message
    if Path(file).is_file():
      QMessageBox(QMessageBox.Information, 'File saved',
                  'The file {} has successfully '
                  'been saved !'.format(Path(file).name)).exec()
    else:
      QMessageBox(QMessageBox.Warning, 'File not saved',
                  'Something went wrong !\nThe file {} could not be saved '
                  'to the desired location.'.format(Path(file).name)).exec()

  def _pass(self, *_, **__) -> None:
    """Does nothing, but used with partial otherwise the window doesn't show."""

    pass
