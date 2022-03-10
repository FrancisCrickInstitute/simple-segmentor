import os
import yaml
import torch
import argparse
import shutil

from train import Trainer
from models import UNet, UNetInception
from data import get_dataloader

"""
Config:
	- num_epochs
	- patch shape (tuple)
	- working_folder
	- patch_shape (tuple)
	- model
		- type (UNet / UNetInception)
		- start_iblock_channels
		- num_down_blocks
		- layers_per_block
	- optimizer
		- lr
	- data
		- train_image_path
		- train_label_path
		- val_image_path
		- val_label_path
		- batch_size
		- steps_per_train_epoch
		- steps_per_val_epoch
"""


def run_experiment(config_filepath):
	if not os.path.exists(config_filepath):
		print("Config filepath does not exist")
		return
	with open(config_filepath) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	experiment_name, _ = os.path.splitext(os.path.basename(config_filepath))
	working_folder = os.path.join("experiments", experiment_name)

	if not os.path.isdir(working_folder):
		os.mkdir(working_folder)
		os.mkdir(os.path.join(working_folder, 'model'))
	elif not os.path.isdir(os.path.join(working_folder, 'model')):
		os.mkdkir(os.path.join(working_folder, 'model'))

	shutil.copy(config_filepath, os.path.join(working_folder, config_filepath))

	patch_shape = config["patch_shape"]
	patch_shape = (int(patch_shape[0]), int(patch_shape[1]), int(patch_shape[2]))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if config["model"]["type"] == "UNet":
		model = UNet(patch_shape[0], patch_shape[0],
		                      start_iblock_channels=config["model"]["start_iblock_channels"],
		                      num_down_blocks=config["model"]["num_down_blocks"],
		                      layers_per_block=config["model"]["layers_per_block"]).to(device)
	else:
		model = UNetInception(patch_shape[0], patch_shape[0],
		                      start_iblock_channels=config["model"]["start_iblock_channels"],
		                      num_down_blocks=config["model"]["num_down_blocks"],
		                      layers_per_block=config["model"]["layers_per_block"]).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=float(config["optimizer"]["lr"]))

	if "steps_per_train_epoch" in config["data"]:
		train_dataloader = get_dataloader(config["data"]["train_image_path"],
		                                  config["data"]["train_label_path"],
		                                  config["patch_shape"],
		                                  config["data"]["batch_size"],
		                                  True,
		                                  num_samples=config["data"]["steps_per_train_epoch"])
	else:
		train_dataloader = get_dataloader(config["data"]["train_image_path"],
		                                  config["data"]["train_label_path"],
		                                  config["patch_shape"],
		                                  config["data"]["batch_size"],
		                                  True)

	if "steps_per_val_epoch" in config["data"]:
		val_dataloader = get_dataloader(config["data"]["val_image_path"],
		                                config["data"]["val_label_path"],
		                                config["patch_shape"],
		                                config["data"]["batch_size"],
		                                num_samples=int(config["data"]["steps_per_val_epoch"]))
	else:
		val_dataloader = get_dataloader(config["data"]["val_image_path"],
		                                config["data"]["val_label_path"],
		                                config["patch_shape"],
		                                config["data"]["batch_size"])

	trainer = Trainer(experiment_name, model, device,
	                  optimizer, train_dataloader, val_dataloader,
	                  int(config["num_epochs"]),
	                  working_folder)

	trainer.train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, help='config filepath')
	args = parser.parse_args()
	run_experiment(args.config)
