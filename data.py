import random
import numpy as np

from torch.utils import data

from skimage.io import imread


def get_dataloader(image_path, label_path, patch_shape, batch_size, shuffle, num_samples=None):
	dataset = SequentialDataset(image_path, label_path, patch_shape)
	if num_samples is None:
		return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
	else:
		sampler = data.RandomSampler(dataset, replacement=True, num_samples=num_samples)
		return data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle, num_workers=4)


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


class SequentialDataset(data.Dataset):
	def __init__(self, image_path, label_path, patch_shape):
		self.image_path = image_path
		self.label_path = label_path
		self.patch_shape = patch_shape

		self.image_stack = imread(image_path)
		self.label_stack = imread(label_path)

		self.z_patches = self.image_stack.shape[0] // self.patch_shape[0]
		self.x_patches = self.image_stack.shape[1] // self.patch_shape[1]
		self.y_patches = self.image_stack.shape[2] // self.patch_shape[2]

	def __len__(self):
		return self.z_patches * self.x_patches * self.y_patches

	def __getitem__(self, idx):
		y_patch = idx % self.y_patches
		x_patch = (idx // self.y_patches) % self.x_patches
		z_patch = (idx // (self.y_patches * self.x_patches)) % self.z_patches

		img_patch = self.image_stack[
					z_patch * self.patch_shape[0]: (z_patch + 1) * self.patch_shape[0],
					x_patch * self.patch_shape[1]: (x_patch + 1) * self.patch_shape[1],
					y_patch * self.patch_shape[2]: (y_patch + 1) * self.patch_shape[2]
		]
		label_patch = self.label_stack[
					z_patch * self.patch_shape[0]: (z_patch + 1) * self.patch_shape[0],
					x_patch * self.patch_shape[1]: (x_patch + 1) * self.patch_shape[1],
					y_patch * self.patch_shape[2]: (y_patch + 1) * self.patch_shape[2]
		]

		return img_patch, label_patch
