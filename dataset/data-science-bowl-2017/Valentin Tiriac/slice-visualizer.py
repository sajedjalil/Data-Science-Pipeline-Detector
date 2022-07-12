## Settings:
# Path to data
DATA = 'd:/data/dsb2017/sample'
# Size to view images as (smaller is faster)
VIEWER_SIZE = 256

## Requirements:
# pip install pyqt5 numpy scipy pydicom

## Instructions:
# - use left side to search for a person; click to load (takes a few seconds)
# - click/drag each image to slice through the _other_ two images

## Example:
# When opened with the `sample` data, it loads person `00cba0` by default.
# Click and drag on top-left image until Y = 77%.
# Notice the nodule on the top-right image (left side); click on it.
# The nodule should now be visible in all three views.

import os
import sys
from functools import total_ordering, partial
import numpy as np
import dicom
from scipy.misc import imresize

from PyQt5.QtWidgets import (
	QApplication, QLabel, QGridLayout, QMainWindow, QWidget, QPushButton,
	QListWidget, QVBoxLayout, QLineEdit, QHBoxLayout,
)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QSize

SHAPE = (VIEWER_SIZE, VIEWER_SIZE)

def main():
	app = App(sys.argv)
	return app.exec_()

class App(QApplication):
	def __init__(self, argv):
		super().__init__(argv)
		
		self._setup_ui()
		
		self._persons = []
		self._person = None
		self._person_index = 0
		self._position = [VIEWER_SIZE // 2, VIEWER_SIZE // 2, VIEWER_SIZE // 2]
		
		self._load_initial()
	
	def _setup_ui(self):
		window = QMainWindow()
		window.setWindowTitle("xyz-util")
		window.resize(150 + VIEWER_SIZE * 2, VIEWER_SIZE * 2)
		# Keep or it gets GC'd
		self._q_window = window
		
		centralWidget = QWidget(window)
		layout = QGridLayout()
		centralWidget.setLayout(layout)
		window.setCentralWidget(centralWidget)
		
		self._setup_left(layout)
		self._setup_viewer(layout)
		
		window.show()
	
	def _setup_left(self, layout):
		vbl = QVBoxLayout()
		
		hbl = QHBoxLayout()
		
		btn1 = QPushButton("Prev Person")
		btn1.clicked.connect(self._on_prev)
		hbl.addWidget(btn1)
		
		btn2 = QPushButton("Next Person")
		btn2.clicked.connect(self._on_next)
		hbl.addWidget(btn2)
		
		wid = QWidget()
		wid.setLayout(hbl)
		vbl.addWidget(wid)
		
		q_info = QLabel()
		vbl.addWidget(q_info)
		
		q_search = QLineEdit()
		q_search.textChanged.connect(self._on_search_changed)
		vbl.addWidget(q_search)
		
		q_results = QListWidget()
		q_results.itemClicked.connect(self._on_result_clicked)
		q_results.setMinimumSize(QSize(100, 400))
		vbl.addWidget(q_results)
		
		wid = QWidget()
		wid.setLayout(vbl)
		layout.addWidget(wid, 0, 0, alignment = Qt.AlignTop)
		
		self._q_info = q_info
		self._q_results = q_results
	
	def _setup_viewer(self, layout):
		q_viewers = [
			ImageViewer(self, axis) for _, axis in AXES
		]
		
		grid = QGridLayout()
		grid.addWidget(q_viewers[0], 0, 0)
		grid.addWidget(q_viewers[1], 0, 1)
		grid.addWidget(q_viewers[2], 1, 0)
		wid = QWidget()
		wid.setLayout(grid)
		layout.addWidget(wid, 0, 1)
		self._q_viewers = q_viewers
	
	def _load_initial(self):
		for uuid in os.listdir(DATA):
			self._persons.append(Person(uuid))
		self._on_search_changed('')
		self._move_to_person(0)
	
	def _on_search_changed(self, text):
		self._q_results.clear()
		for p in self._persons:
			if p.uuid.startswith(text):
				self._q_results.addItem(p.uuid[:10])
	
	def _on_result_clicked(self, item):
		uuid = item.text()
		for i, p in enumerate(self._persons):
			if p.uuid.startswith(uuid):
				self._person_index = i
				self._person = p
				self._redraw()
				break
	
	def _on_viewer_click(self, axis, pos):
		y = int(np.clip(pos.y(), 0, VIEWER_SIZE - 1))
		self._position[axis[1]] = y
		
		x = int(np.clip(pos.x(), 0, VIEWER_SIZE - 1))
		self._position[axis[2]] = x
		
		self._redraw()
	
	def _on_next(self):
		self._move_to_person(+1)
	
	def _on_prev(self):
		self._move_to_person(-1)
	
	def _move_to_person(self, dir):
		self._person_index = (self._person_index + dir) % len(self._persons)
		self._person = self._persons[self._person_index]
		self._redraw()
	
	def _redraw(self):
		person = self._person
		
		if person is None:
			return
		
		for q_viewer in self._q_viewers:
			axis = q_viewer.axis
			pos = self._position[axis[0]]
			slice = person.cube.transpose(axis)[pos]
			q_viewer.setImage(slice)
		
		info_text = "Pers: {}\nX: {:04.0%}   Y: {:04.0%}   Z: {:04.0%}".format(
			person.uuid[:6],
			self._position[2] / VIEWER_SIZE,
			self._position[1] / VIEWER_SIZE,
			self._position[0] / VIEWER_SIZE,
		)
		self._q_info.setText(info_text)

AXES = [
	("Left-Right (X)", (2, 1, 0)),
	("Top-Bottom (Y)", (1, 0, 2)),
	("Front-Back (Z)", (0, 1, 2)),
]

class Person:
	__slots__ = ('uuid', '_data')
	
	def __init__(self, uuid):
		self.uuid = uuid
		self._data = None
	
	@property
	def slices(self):
		self._load()
		return self._data['slices']
	
	@property
	def cube(self):
		self._load()
		return self._data['cube']
	
	def _load(self):
		if self._data is not None:
			return
		slices = sorted(
			Slice(self.uuid, file)
			for file in os.listdir(DATA + '/' + self.uuid)
		)
		
		stacked = _stack_and_clean_up(slices)
		cube = _resize(stacked)
		
		self._data = {
			'slices': slices,
			'cube': cube,
		}

def _stack_and_clean_up(slices):
	stacked = np.array([s.dcm.pixel_array for s in slices], dtype = np.float16)
	stacked[stacked <= -2000] = 0
	return stacked

def _resize(stacked):
	tmp = stacked
	
	# Hack to make the resizing faster
	r0 = int(tmp.shape[1] / SHAPE[0])
	r1 = int(tmp.shape[2] / SHAPE[1])
	if r0 > 1 or r1 > 1:
		tmp = tmp[:,::r0,::r1]
	
	tmp = np.array([imresize(tmp[i,:], SHAPE) for i in range(tmp.shape[0])], dtype = np.float16)
	tmp = np.array([imresize(tmp[:,i], SHAPE) for i in range(tmp.shape[1])], dtype = np.float16)
	tmp -= np.min(tmp)
	tmp *= 255 / np.max(tmp)
	return tmp.astype(np.uint8)

@total_ordering
class Slice:
	def __init__(self, person_uuid, name):
		self.person_uuid = person_uuid
		self.name = name
		self.filepath = '{}/{}/{}'.format(DATA, person_uuid, name)
		
		dcm = dicom.read_file(self.filepath)
		self.inum = int(dcm.InstanceNumber)
		self.dcm = dcm
		self.size = dcm.pixel_array.shape
	
	def __lt__(self, other):
		assert isinstance(other, Slice)
		return self.inum < other.inum
	
	def __eq__(self, other):
		assert isinstance(other, Slice)
		return self.inum == other.inum

class ImageViewer(QLabel):
	def __init__(self, app, axis):
		super().__init__()
		
		self.axis = axis
		
		size = QSize(*SHAPE)
		
		self._app = app
		self._pixmap = QPixmap(size)
		self._buffer = np.zeros(np.prod(SHAPE), dtype = np.uint8)
		self._image = QImage(self._buffer, *SHAPE, QImage.Format_Grayscale8)
		
		self.setMinimumSize(size)
		self.setMaximumSize(size)
		self._draw()
	
	def mousePressEvent(self, ev):
		self._app._on_viewer_click(self.axis, ev.localPos())
	
	def mouseMoveEvent(self, ev):
		self._app._on_viewer_click(self.axis, ev.localPos())
	
	def setImage(self, data):
		self._buffer[:] = data.flatten()
		self._draw()
	
	def _draw(self):
		self._pixmap.convertFromImage(self._image)
		self.setPixmap(self._pixmap)

if __name__ == '__main__':
	sys.exit(main())
