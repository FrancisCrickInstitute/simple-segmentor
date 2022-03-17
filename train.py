import os
import time
from datetime import datetime

import torch
from tqdm import tqdm

# Add validation at the end of each epoch


def recall(pred, y):
    tp = torch.sum(torch.logical_and(pred, y))
    fn = torch.sum(torch.logical_and(pred == 0, y == 1))
    return float(tp/(tp + fn))


def precision(pred, y):
    tp = torch.sum(torch.logical_and(pred, y))
    fp = torch.sum(torch.logical_and(pred == 1, y == 0))
    return float(tp/(tp + fp))


def iou(pred, y):
    intersection = torch.sum(pred * y)
    union = torch.sum(torch.logical_or(pred, y))
    return intersection / union


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

    def evaluate(self, dataloader):
        with torch.no_grad():
            total_dice = 0
            total_iou = 0
            total_precision = 0
            total_recall = 0
            total_cpu_time = 0
            total_gpu_time = 0
            for x_batch, y_batch in dataloader:
                cpu_start_time = time.time()

                x_batch, y_batch = x_batch.type(torch.int), y_batch.type(torch.int)
                x_batch, y_batch = self.normalize_func2d(x_batch, y_batch)
                cpu_done_time = time.time()

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(x_batch)
                total_dice += dice(y_pred, y_batch)

                y_pred = y_pred > 0.5
                y_batch = y_batch > 0.5

                total_iou += iou(y_pred, y_batch)
                total_precision += precision(y_pred, y_batch)
                total_recall += recall(y_pred, y_batch)

                total_cpu_time += cpu_done_time - cpu_start_time
                total_gpu_time += time.time() - cpu_done_time

            avg_dice = total_dice / len(dataloader)
            avg_iou = total_iou / len(dataloader)
            avg_precision = total_precision / len(dataloader)
            avg_recall = total_recall / len(dataloader)

            return {
                "DICE": avg_dice,
                "IoU": avg_iou,
                "Precision": avg_precision,
                "Recall": avg_recall,
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

        train_metrics = self.evaluate(self.train_dataloader)
        val_metrics = self.evaluate(self.val_dataloader)

        results_file = os.path.join(self.working_folder, self.name + '.results')
        with open(results_file, "w+") as f:
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
