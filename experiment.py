import os
import yaml
import torch

from train import Trainer
from models import UNet, UNetInception
from data import get_dataloader

"""
Config:
	- num_epochs
	- working_folder
	- patch shape (tuple)
	- model
		- type (UNet / UNetInception)
		- start_iblock_channels
		- num_down_blocks
		- layers_per_block
	- optimizer
		- lr
	- data
		- img_dir
		- label_dir
		- batch_size
"""


def run_experiment(config_filepath):
	with open(config_filepath) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	experiment_name, _ = os.path.splitext(os.path.basename(config_filepath))

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

	dataloader = get_dataloader(config["data"]["img_dir"],
	                            config["data"]["label_dir"],
	                            int(config["data"]["batch_size"]),
	                            config["patch_shape"],
	                            shuffle=True)

	trainer = Trainer(experiment_name, model, device,
	                  optimizer, dataloader, int(config["num_epochs"]),
	                  config["working_folder"])

	trainer.train()


if __name__ == '__main__':
	run_experiment("initial.yml")