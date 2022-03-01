import numpy as np
import torch
import os

# Add validation at the end of each epoch

def dice(pred, y, smooth=0):
	intersection = torch.sum(pred * y)
	return (2 * intersection + smooth) / (torch.sum(pred) + torch.sum(y) + smooth)

def dice_loss(pred, y, smooth=1):
	return 1 - dice(pred, y, smooth=smooth)


class Trainer:
	def __init__(self, name, model, device, optimizer, train_sampler, val_sampler, n_epochs, working_folder):
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
		self.train_sampler = train_sampler
		self.val_sampler = val_sampler
		self.n_epochs = n_epochs
		self.working_folder = working_folder

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

	def train(self):
		print(f"=> Training...")
		for self.epoch in range(self.epoch + 1, self.epoch + self.n_epochs + 1):
			train_running_loss = 0
			# _train_one_epoch, get loss, cpu_time, gpu_time
			for i in range(self.train_sampler.steps_per_epoch):
				#Add timing stuff
				x_batch, y_batch = self.train_sampler.sample()
				
				x_batch = (x_batch - 127) / 128
				y_batch = (y_batch > 0.5)

				x_batch = torch.from_numpy(x_batch.astype(np.float32)).to(self.device)
				y_batch = torch.from_numpy(y_batch.astype(np.float32)).to(self.device)

				self.optimizer.zero_grad()
				y_pred = self.model(x_batch)
				loss = dice_loss(y_pred, y_batch)
				loss.backward()
				self.optimizer.step()
				train_running_loss += loss.item()

			val_running_loss = 0
			with torch.no_grad():
				for i in range(self.val_sampler.steps_per_epoch):
					x_batch, y_batch = self.val_sampler.sample()
					x_batch = (x_batch - 127) / 128
					y_batch = (y_batch > 0.5)

					x_batch = torch.from_numpy(x_batch.astype(np.float32)).to(self.device)
					y_batch = torch.from_numpy(y_batch.astype(np.float32)).to(self.device)

					y_pred = self.model(x_batch)
					loss = dice_loss(y_pred, y_batch)
					val_running_loss += loss.item()

			train_avg_loss = train_running_loss / self.train_sampler.steps_per_epoch
			val_avg_loss = val_running_loss / self.val_sampler.steps_per_epoch
			self.save_model_checkpoint(train_avg_loss, val_avg_loss)
			print(f"Epoch {self.epoch},\taverage train loss {train_avg_loss},\taverage val loss {val_avg_loss}")
