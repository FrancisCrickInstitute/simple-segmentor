import torch
import os
import numpy as np

from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm


# Add validation at the end of each epoch


class Trainer:
	def __init__(self, name, model, device, optimizer, dataloader, n_epochs, working_folder):
		self.epoch = 0
		self.total_cpu_time = 0
		self.total_gpu_time = 0
		self.collected_loss = []
		self.best_epoch = {'epoch': 0, 'loss': float('inf')}

		self.name = name
		self.model = model
		self.device = device
		self.optimizer = optimizer
		self.dataloader = dataloader
		self.n_epochs = n_epochs
		self.working_folder = working_folder

		self.steps_per_epoch = len(self.dataloader.dataset) // self.dataloader.batch_size

	def _dice(self, pred, y, smooth=1):
		intersection = torch.sum(pred * y)
		return (2*intersection + smooth) / (torch.sum(pred) + torch.sum(y) + smooth)

	def loss_func(self, pred, y, smooth=1):
		return 1 - self._dice(pred, y, smooth=smooth)

	def normalize_func2d(self, x, y):
		return (x - 127)/128, y > 0.5

	def save_model_checkpoint(self, loss):
		model_save_folder = os.path.join(self.working_folder, 'model')

		if loss < self.best_epoch['loss']:
			self.best_epoch = {'epoch': self.epoch, 'loss': loss}
			files = os.listdir(model_save_folder)
			for file in files:
				filename, ext = os.path.splitext(file)
				if filename.startswith(f'{self.name}.best.') and ext == 'pt':
					os.remove(os.path.join(model_save_folder, file))

			best_model_fname = os.path.join(model_save_folder, f'{self.name}.best.{self.epoch}.pt')
			torch.save(self.model.state_dict(), best_model_fname)

		model_checkpoint_filename = os.path.join(model_save_folder, f'{self.name}.{self.epoch}.pt')
		torch.save(self.model.state_dict(), model_checkpoint_filename)

	def train(self):
		print(f"=> Training...")
		for self.epoch in range(self.epoch + 1, self.epoch + self.n_epochs + 1):
			running_loss = 0
			# _train_one_epoch, get loss, cpu_time, gpu_time
			for x_batch, y_batch in tqdm(self.dataloader, desc=f"Epoch {self.epoch + 1}", total=self.steps_per_epoch):
				# Add timing stuff
				x_batch, y_batch = self.normalize_func2d(x_batch, y_batch)

				x_batch = x_batch.to(self.device)
				y_batch = y_batch.to(self.device)

				self.optimizer.zero_grad()
				y_pred = self.model(x_batch)
				loss = self.loss_func(y_pred, y_batch)
				loss.backward()
				self.optimizer.step()
				running_loss += loss.item()

			avg_loss = running_loss / self.steps_per_epoch
			self.save_model_checkpoint(avg_loss)
		print(f"Epoch {self.epoch}, loss {running_loss}")
