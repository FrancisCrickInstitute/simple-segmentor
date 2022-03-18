import os

from torch.utils import data

from skimage.io import imread


def get_dataloader(image_path, label_path, patch_shape, batch_size, shuffle):
    dataset = SequentialDataset(image_path, label_path, patch_shape)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


class SequentialDataset(data.Dataset):
    def __init__(self, image_path, label_path, patch_shape):
        self.image_path = image_path
        self.label_path = label_path
        self.patch_shape = patch_shape

        self.image_name = os.path.basename(image_path)
        self.label_name = os.path.basename(label_path)

        self.image_stack = imread(image_path)
        self.label_stack = imread(label_path)

        # note: for the purposes of this demo code, image shapes which aren't evenly divided by the
        #       patch shape discard the ends
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
