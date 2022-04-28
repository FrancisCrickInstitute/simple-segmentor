import os
import yaml
import torch
import argparse

from train import Trainer
from models import UNet, UNetInception
from data import get_dataloader

"""
Note: this file assumes the experiment directory exists
"""
def evaluate(experiment_directory):
    experiment_name = os.path.basename(experiment_directory)

    config_path = os.path.join(experiment_directory, experiment_name + ".yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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

    train_dataloader = get_dataloader(config["data"]["train_image_path"],
                                      config["data"]["train_label_path"],
                                      config["patch_shape"],
                                      config["data"]["batch_size"],
                                      shuffle=True)

    val_dataloader = get_dataloader(config["data"]["val_image_path"],
                                    config["data"]["val_label_path"],
                                    config["patch_shape"],
                                    config["data"]["batch_size"],
                                    shuffle=False)
    trainer = Trainer(experiment_name, model, device,
                      None, train_dataloader, val_dataloader,
                      int(config["num_epochs"]),
                      experiment_directory)
    trainer.save_best_epoch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, help='experiment directory')
    args = parser.parse_args()
    evaluate(args.experiment)
