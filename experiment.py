import os
import yaml
import torch

from train import Trainer
from models import UNet, UNetInception
from data import get_dataloader

"""
Config:
	- model (UNet / UNetInception)
	- num_epochs
	- working_folder
	- model
		- in_channels
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

	if config["model"] == "UNet":
		model = UNet(config["model"]["in_channels"], 1)
	else:
		model = UNetInception(config["model"]["in_channels"], 1)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])

	dataloader = get_dataloader(config["data"]["img_dir"],
	                            config["data"]["label_dir"],
	                            config["data"]["batch_size"],
	                            shuffle=True)

	trainer = Trainer(experiment_name, model, device,
	                  optimizer, dataloader, config["num_epochs"],
	                  config["working_folder"])

	trainer.train()
