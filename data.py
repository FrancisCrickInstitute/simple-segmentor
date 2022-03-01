import random
import numpy as np

from skimage.io import imread

class Sampler:
	def __init__(self, image_path, label_path, patch_shape, batch_size, steps_per_epoch):
		self.image_path = image_path
		self.label_path = label_path
		self.patch_shape = patch_shape
		self.batch_size = batch_size
		self.steps_per_epoch = steps_per_epoch

		self.image_stack = imread(image_path)
		self.label_stack = imread(label_path)

	def random_patches(self):
		rng = np.random.default_rng()
		image_patches = np.zeros((1, *self.patch_shape))
		label_patches = np.zeros((1, *self.patch_shape))

		max_coords_for_patch_start = (
			self.image_stack.shape[0] - self.patch_shape[0],
			self.image_stack.shape[1] - self.patch_shape[1],
			self.image_stack.shape[2] - self.patch_shape[2],
		)
		start_coords = rng.integers((0, 0, 0), max_coords_for_patch_start, endpoint=True)
		image_patches[0, :, :, :] = self.image_stack[...,
									start_coords[0]:start_coords[0] + self.patch_shape[0],
                                    start_coords[1]:start_coords[1] + self.patch_shape[1],
                                    start_coords[2]:start_coords[2] + self.patch_shape[2]
		]

		label_patches[0, :, :, :] = self.label_stack[...,
									start_coords[0]:start_coords[0] + self.patch_shape[0],
                                    start_coords[1]:start_coords[1] + self.patch_shape[1],
                                    start_coords[2]:start_coords[2] + self.patch_shape[2]
		]
		return image_patches, label_patches

	def sample(self):
		image_patches = np.zeros((self.batch_size, *self.patch_shape))
		label_patches = np.zeros((self.batch_size, *self.patch_shape))

		for i in range(self.batch_size):
			rand_image, rand_label = self.random_patches()
			image_patches[i, ...] = rand_image[0, ...]
			label_patches[i, ...] = rand_label[0, ...]

		return image_patches, label_patches