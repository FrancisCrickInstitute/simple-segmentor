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
	- model
		- type (UNet / UNetInception)
		- in_channels
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

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if config["model"]["type"] == "UNet":
		model = UNet(int(config["model"]["in_channels"]), 1,
		                      start_iblock_channels=config["model"]["start_iblock_channels"],
		                      num_down_blocks=config["model"]["num_down_blocks"],
		                      layers_per_block=config["model"]["layers_per_block"]).to(device)
	else:
		model = UNetInception(int(config["model"]["in_channels"]), 1,
		                      start_iblock_channels=config["model"]["start_iblock_channels"],
		                      num_down_blocks=config["model"]["num_down_blocks"],
		                      layers_per_block=config["model"]["layers_per_block"]).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=float(config["optimizer"]["lr"]))

	dataloader = get_dataloader(config["data"]["img_dir"],
	                            config["data"]["label_dir"],
	                            int(config["data"]["batch_size"]),
	                            shuffle=True)

	trainer = Trainer(experiment_name, model, device,
	                  optimizer, dataloader, int(config["num_epochs"]),
	                  config["working_folder"])

	trainer.train()


if __name__ == '__main__':
	run_experiment("initial.yml")