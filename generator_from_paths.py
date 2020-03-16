import cv2
import math
import numpy as np
import random
from tensorflow import keras as tfk


class GeneratorFromPaths(tfk.utils.Sequence):
	"""
	Image generator for Keras.
	Only image paths are kept in memory. Images are loaded one at a time in __getitem__.
	Labels are provided in one-hot encoding.
	Only 1- and 3-channel images are supported.
	"""

	def __init__(self, label_path_list, batch_size, image_shape, class_count, shuffle=True):
		self.label_path_list = label_path_list
		self.batch_size = batch_size
		self.image_shape = image_shape
		self.class_count = class_count
		self.shuffle = shuffle
		self.channels = self.image_shape[2]
		assert (self.channels == 1 or self.channels == 3)
		self.sample_count = len(self.label_path_list)
		self.length = int(math.ceil(self.sample_count / float(self.batch_size)))
		if self.shuffle:
			random.shuffle(self.label_path_list)

	@property
	def classes(self):
		return np.array([el[0] for el in self.label_path_list])

	@property
	def paths(self):
		return [el[1] for el in self.label_path_list]

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		images = []
		labels = []
		i = idx * self.batch_size
		for _ in range(self.batch_size):
			label, path = self.label_path_list[i]
			i += 1
			raw_image = cv2.imread(path) if self.channels == 3 else cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(raw_image, self.image_shape[0:2])
			image = image.astype(float) / 255.
			images.append(image)

			one_hot_label = self.to_one_hot(label, self.class_count)
			labels.append(one_hot_label)
			if i == self.sample_count:
				if self.shuffle:
					random.shuffle(self.label_path_list)
				break
		return np.array(images), np.array(labels)

	@staticmethod
	def to_one_hot(label, class_count):
		assert (class_count > label)
		one_hot = np.zeros(class_count)
		one_hot[label] = 1
		return one_hot
