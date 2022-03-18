import os
import time
from datetime import datetime
import numpy as np

import torch
from torch.nn import functional as F

from skimage.io import imsave

# Add validation at the end of each epoch


def recall(pred, y):
    tp = np.sum(np.logical_and(pred, y))
    fn = np.sum(np.logical_and(pred == 0, y == 1))
    return float(tp/(tp + fn))


def precision(pred, y):
    tp = np.sum(np.logical_and(pred, y))
    fp = np.sum(np.logical_and(pred == 1, y == 0))
    return float(tp/(tp + fp))


def iou(pred, y):
    intersection = np.sum(pred * y)
    union = np.sum(np.logical_or(pred, y))
    return intersection / union


def np_dice(pred, y, smooth=0):
    intersection = np.sum(pred * y)
    return (2 * intersection + smooth) / (np.sum(pred) + np.sum(y) + smooth)


def dice(pred, y, smooth=0):
    intersection = torch.sum(pred * y)
    return (2 * intersection + smooth) / (torch.sum(pred) + torch.sum(y) + smooth)


def dice_loss(pred, y, smooth=1):
    return 1 - dice(pred, y, smooth=smooth)


class Trainer:
    def __init__(self, name, model, device, optimizer, train_dataloader,
                 val_dataloader, n_epochs, working_folder):
        self.bce = torch.nn.BCELoss()

        self.epoch = 0
        self.total_cpu_time = 0
        self.total_gpu_time = 0
        self.collected_loss = []
        self.best_epoch = {'epoch': 0, 'train_loss': float('inf'), 'val_loss': float('inf')}

        self.name = name
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.n_epochs = n_epochs
        self.working_folder = working_folder

        self.log_file = os.path.join(self.working_folder, self.name + '.log')
        with open(self.log_file, 'w+') as f:
            f.write('epoch,training loss,validation loss,train cpu time,train gpu time,val time,timestamp\n')

    def normalize_func2d(self, x, y):
        return ((x - 127)/128), (y > 0.5)

    def save_model_checkpoint(self, train_loss, val_loss):
        model_save_folder = os.path.join(self.working_folder, 'model')

        if val_loss < self.best_epoch['val_loss']:
            self.best_epoch = {'epoch': self.epoch, 'train_loss': train_loss, 'val_loss': val_loss}
            files = os.listdir(model_save_folder)
            for file in files:
                filename, ext = os.path.splitext(file)
                if filename.startswith(f'{self.name}.best.') and ext == 'pt':
                    os.remove(os.path.join(model_save_folder, file))

            best_model_fname = os.path.join(model_save_folder, f'{self.name}.best.{self.epoch}.pt')
            torch.save(self.model.state_dict(), best_model_fname)

        model_checkpoint_filename = os.path.join(model_save_folder, f'{self.name}.{self.epoch}.pt')
        torch.save(self.model.state_dict(), model_checkpoint_filename)

    def append_to_log(self, train_loss, val_loss, cpu_time, gpu_time, val_time):
        with open(self.log_file, 'a') as f:
            f.write(f'{self.epoch},{train_loss:.4f},{val_loss:.4f},{cpu_time:.2f}s,{gpu_time:.2f}s,{val_time:.2f}s'
                    f',{datetime.now()}\n')

    def segment_stack(self, image_stack, patch_shape, total_cpu_time):
        self.model.eval()
        cpu_start_time = time.time()

        padded_volume = np.pad(
            image_stack,
            (
                (patch_shape[0], patch_shape[0]),
                (patch_shape[1], patch_shape[1]),
                (patch_shape[2], patch_shape[2]),
            ),
            'symmetric'
        )

        grid_coordinates = [
            (z, y, x)
            for z in range(0, padded_volume.shape[0] - patch_shape[0], patch_shape[0])
            for y in range(0, padded_volume.shape[1] - patch_shape[1], patch_shape[1])
            for x in range(0, padded_volume.shape[2] - patch_shape[2], patch_shape[2])
        ]
        result_volume = np.zeros_like(padded_volume, dtype=np.float32)

        batch_size = 12
        total_cpu_time += time.time() - cpu_start_time
        total_gpu_time = 0
        for batch_start in range(0, len(grid_coordinates), batch_size):
            cpu_start_time = time.time()
            batch_corner_coords = grid_coordinates[batch_start:batch_start + batch_size]
            true_batch_size = len(batch_corner_coords)
            image_patches = np.zeros((true_batch_size, *patch_shape))
            for i, corner_coord in enumerate(batch_corner_coords):
                image_patch = padded_volume[
                    corner_coord[0]:corner_coord[0] + patch_shape[0],
                    corner_coord[1]:corner_coord[1] + patch_shape[1],
                    corner_coord[2]:corner_coord[2] + patch_shape[2],
                ]
                image_patches[i, :, :, :] = image_patch

            image_patches = torch.from_numpy(image_patches.astype(np.float32)).to(self.device)
            gpu_start_time = time.time()
            with torch.no_grad():
                predictions = self.model(image_patches).cpu().detach().numpy()
            gpu_time = time.time() - gpu_start_time
            total_gpu_time += gpu_time

            for i, corner_coord in enumerate(batch_corner_coords):
                result_volume[
                    corner_coord[0]:corner_coord[0] + patch_shape[0],
                    corner_coord[1]:corner_coord[1] + patch_shape[1],
                    corner_coord[2]:corner_coord[2] + patch_shape[2],
                ] = predictions[i]
            total_cpu_time += time.time() - cpu_start_time - gpu_time

        result_volume = result_volume[
            patch_shape[0]:-patch_shape[0],
            patch_shape[1]:-patch_shape[1],
            patch_shape[2]:-patch_shape[2],
        ]
        return result_volume, total_cpu_time, total_gpu_time

    def evaluate(self, image_stack, label_stack, patch_shape, results_path):
        cpu_start_time = time.time()
	
        # To combat underflow
        image_stack = image_stack.astype(np.float16)
        label_stack = label_stack.astype(np.float16)

        image_stack, label_stack = self.normalize_func2d(image_stack, label_stack)
        total_cpu_time = time.time() - cpu_start_time
        segmented_stack, total_cpu_time, total_gpu_time = self.segment_stack(image_stack, patch_shape, total_cpu_time)

        cpu_start_time = time.time()
        prec = precision(segmented_stack, label_stack)
        rec = recall(segmented_stack, label_stack)
        _dice = np_dice(segmented_stack, label_stack)
        _iou = iou(segmented_stack, label_stack)

        imsave(results_path, segmented_stack)

        total_cpu_time += time.time() - cpu_start_time

        return {
            "DICE": _dice,
            "IoU": _iou,
            "Precision": prec,
            "Recall": rec,
            "GPU time": total_gpu_time,
            "CPU time": total_cpu_time
        }

    def save_best_epoch(self):
        model_save_folder = os.path.join(self.working_folder, 'model')
        files = os.listdir(model_save_folder)
        best_epoch = 0
        for file in files:
            filename, ext = os.path.splitext(file)
            if filename.startswith(f'{self.name}.best.') and ext == 'pt':
                self.model.load_state_dict(torch.load(os.path.join(model_save_folder, file)))
                best_epoch = int(filename.split(".")[-1])
                break

        if not os.path.isdir(os.path.join(self.working_folder, "results")):
            os.mkdir(os.path.join(self.working_folder, "results"))

        train_metrics = self.evaluate(self.train_dataloader.dataset.image_stack,
                                      self.train_dataloader.dataset.label_stack,
                                      self.train_dataloader.dataset.patch_shape,
                                      os.path.join(self.working_folder, "results", self.train_dataloader.dataset.image_name))
        val_metrics = self.evaluate(self.val_dataloader.dataset.image_stack,
                                    self.val_dataloader.dataset.label_stack,
                                    self.val_dataloader.dataset.patch_shape,
                                    os.path.join(self.working_folder, "results", self.val_dataloader.dataset.image_name))

        results_file = os.path.join(self.working_folder, "results", self.name + '.results')
        with open(results_file, "w+") as f:
            f.write(f"Total training cpu time: {self.total_cpu_time}\n")
            f.write(f"Total training gpu time: {self.total_gpu_time}\n")
            f.write(f"Best epoch: {best_epoch + 1}\n\n")

            f.write("Best epoch training set results:\n")
            f.write(f" - DICE score: {train_metrics['DICE']}\n")
            f.write(f" - IoU score: {train_metrics['IoU']}\n")
            f.write(f" - Precision: {train_metrics['Precision']}\n")
            f.write(f" - Recall: {train_metrics['Recall']}\n")
            f.write(f" - GPU time: {train_metrics['GPU time']}\n")
            f.write(f" - CPU time: {train_metrics['CPU time']}\n\n")

            f.write("Best epoch validation set results:\n")
            f.write(f" - DICE score: {val_metrics['DICE']}\n")
            f.write(f" - IoU score: {val_metrics['IoU']}\n")
            f.write(f" - Precision: {val_metrics['Precision']}\n")
            f.write(f" - Recall: {val_metrics['Recall']}\n")
            f.write(f" - GPU time: {val_metrics['GPU time']}\n")
            f.write(f" - CPU time: {val_metrics['CPU time']}\n")

    def train(self):
        print(f"Training dataset:\t{len(self.train_dataloader.dataset)} patches")
        print(f"Validation dataset:\t{len(self.val_dataloader.dataset)} patches")
        print(f"=> Training...")
        for self.epoch in range(1, self.n_epochs + 1):
            train_running_loss = 0
            cpu_time = 0
            gpu_time = 0

            print(f'Epoch {self.epoch}/{self.n_epochs}')
            for i, (x_batch, y_batch) in enumerate(self.train_dataloader):
                loop_iter_start_time = time.time()

                x_batch = x_batch.type(torch.int)
                y_batch = y_batch.type(torch.int)

                x_batch, y_batch = self.normalize_func2d(x_batch, y_batch)

                cpu_cycle_done_time = time.time()

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = dice_loss(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_running_loss += loss.item()

                if self.device.type == 'cuda':
                    cpu_time += cpu_cycle_done_time - loop_iter_start_time
                    gpu_time += time.time() - cpu_cycle_done_time
                else:
                    cpu_time += time.time() - loop_iter_start_time

            self.total_cpu_time += cpu_time
            self.total_gpu_time += gpu_time

            val_running_loss = 0
            val_start_time = time.time()
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(self.val_dataloader):
                    x_batch = x_batch.type(torch.int)
                    y_batch = y_batch.type(torch.int)

                    x_batch, y_batch = self.normalize_func2d(x_batch, y_batch)

                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    y_pred = self.model(x_batch)
                    loss = dice_loss(y_pred, y_batch)
                    val_running_loss += loss.item()

            val_time = time.time() - val_start_time

            train_avg_loss = train_running_loss / len(self.train_dataloader)
            val_avg_loss = val_running_loss / len(self.val_dataloader)
            self.save_model_checkpoint(train_avg_loss, val_avg_loss)
            self.append_to_log(train_avg_loss, val_avg_loss, cpu_time, gpu_time, val_time)

        self.save_best_epoch()
