import torch
import numpy as np
import torch.utils.data as data
import os

from skimage.io import imread


def get_dataloader(img_dir, label_dir, batch_size, shuffle):
	dataset = ImageFolder(img_dir, label_dir)
	return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


class ImageFolder(data.Dataset):
	def __init__(self, img_dir, label_dir):
		self.patch_shape = (12, 256, 256)

		self.img_dir = img_dir
		self.image_stacks = []
		for file in os.listdir(self.img_dir):
			if os.path.isfile(file):
				filename, ext = os.path.splitext(file)
				if ext == 'tiff' or ext == 'tif':
					self.image_stacks.append(imread(os.path.join(self.img_dir, file)))

		self.label_dir = label_dir
		self.label_stacks = []
		for file in os.listdir(self.label_dir):
			if os.path.isfile(file):
				filename, ext = os.path.splitext(file)
				if ext == 'tiff' or ext == 'tif':
					self.label_stacks.append(imread(os.path.join(self.label_dir, file)))

	def __getitem__(self, idx):
		img_patch = np.zeros(*self.patch_shape)
		label_patch = np.zeros(*self.patch_shape)

		image_stack = self.image_stacks[idx]
		label_stack = self.label_stack[idx]

		image_patch, label_patch = random_patches_from_stacks(image_stack,
		                                                    label_stack,
		                                                    self.patch_shape)
		return image_patch, label_patch

	def __len__(self):
		return len(self.image_stacks)


def random_patches_from_stacks(image_stack, label_stack, patch_shape):
	rng = np.random.default_rng()

	image_patches = np.zeros(*patch_shape)

	max_coords_for_patch_start = (
		image_stack.shape[0] - patch_shape[0],
		image_stack.shape[1] - patch_shape[1],
		image_stack.shape[2] - patch_shape[2]
	)
	start_coords = rng.integers((0, 0, 0), max_coords_for_patch_start, endpoint=True)
	image_patch = image_stack[...,
					start_coords[0]:start_coords[0] + patch_shape[0],
	                start_coords[1]:start_coords[1] + patch_shape[1],
	                start_coords[2]: start_coords[2] + patch_shape[2]]
	label_patch = label_stack[...,
	              start_coords[0]:start_coords[0] + patch_shape[0],
	              start_coords[1]:start_coords[1] + patch_shape[1],
	              start_coords[2]: start_coords[2] + patch_shape[2]]

	return image_patch, label_patch