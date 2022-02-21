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
		for file in os.listdir(self.img_dir):
			if os.path.isfile(file):
				filename, ext = os.path.splitext(file)
				if ext == 'tiff' or ext == 'tif':
					self.image_stack = imread(os.path.join(self.img_dir, file))
					break

		self.label_dir = label_dir
		for file in os.listdir(self.label_dir):
			if os.path.isfile(file):
				filename, ext = os.path.splitext(file)
				if ext == 'tiff' or ext == 'tif':
					self.label_stack = imread(os.path.join(self.label_dir, file))
					break

	def __getitem__(self, idx):
		rng = np.random.default_rng()

		max_coords_for_patch_start = (
			self.image_stack.shape[1] - self.patch_shape[1],
			self.image_stack.shape[2] - self.patch_shape[2]
		)
		start_coords = rng.integers((0, 0), max_coords_for_patch_start, endpoint=True)
		image_patch = self.image_stack[...,
					  idx * self.patch_shape[0] : idx * self.patch_shape[0] + self.patch_shaep[0],
					  start_coords[1] : start_coords[1] + self.patch_shape[1],
					  start_coords[2]: start_coords[2] + self.patch_shape[2],
		]

		label_patch = self.label_stack[...,
		              idx * self.patch_shape[0]: idx * self.patch_shape[0] + self.patch_shaep[0],
		              start_coords[1]: start_coords[1] + self.patch_shape[1],
		              start_coords[2]: start_coords[2] + self.patch_shape[2],
		              ]

		return image_patch, label_patch

	def __len__(self):
		return self.image_stack.shape[0] // self.patch_shape[0]