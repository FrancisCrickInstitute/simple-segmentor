import torch

from torch import nn


class InceptionBlock(nn.Module):
	def __init__(self, in_channels, iblock_channels):
		super(InceptionBlock, self).__init__()

		self.p1 = nn.Sequential(
			nn.Conv2d(in_channels, iblock_channels, (1,1)),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(iblock_channels, iblock_channels, (3,3), padding=1),
			nn.LeakyReLU(0.1, True),
		)

		self.p2 = nn.Sequential(
			nn.Conv2d(in_channels, iblock_channels, (1, 1)),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(iblock_channels, iblock_channels, (5, 5), padding=2),
			nn.LeakyReLU(0.1, True),
		)

		self.p3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(in_channels, iblock_channels, (1, 1)),
			nn.LeakyReLU(0.1, True),
		)

	def forward(self, x):
		x1 = self.p1(x)
		x2 = self.p2(x)
		x3 = self.p3(x)
		return torch.cat((x1, x2, x3), dim=1)


class UNetInceptionBlock(nn.Module):
	def __init__(self, in_channels, iblock_channels, layers=1, submodule=None, bottom=False, top=False):
		super(UNetInceptionBlock, self).__init__()
		self.bottom = bottom

		down_layers = [InceptionBlock(in_channels, iblock_channels)]
		for layer in range(layers - 1):
			down_layers += [InceptionBlock(iblock_channels * 3, iblock_channels)]
		self.down_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
		self.down_model = nn.Sequential(*down_layers)

		if top:
			self.submodule = submodule
			up_layers = [nn.Conv2d(in_channels * 6, in_channels, kernel_size=(3, 3), padding=1)]
		elif not bottom:
			self.submodule = submodule
			up_layers = [nn.ConvTranspose2d(in_channels * 6, in_channels, kernel_size=(3, 3), stride=2, padding=1,
			                                output_padding=1)]
		else:
			up_layers = [nn.ConvTranspose2d(iblock_channels * 3, in_channels, kernel_size=(3, 3), stride=2, padding=1,
			                                 output_padding=1)]

		for i in range(layers - 1):
			up_layers += [InceptionBlock(in_channels, in_channels // 3)]
		self.up_model = nn.Sequential(*up_layers)

	def forward(self, x):
		if self.bottom:
			x1 = self.down_model(x)
			x1 = self.up_model(x1)
			return x1
		else:
			x1 = self.down_model(x)
			x2 = self.down_pool(x1)
			x2 = self.submodule(x2)
			return self.up_model(torch.cat((x1, x2), dim=1))


class UNetBlock(nn.Module):
	def __init__(self, in_channels, layers=1, submodule=None, bottom=False, top=False):
		super(UNetBlock, self).__init__()
		self.bottom = bottom

		down_layers = [nn.Conv2d(in_channels, in_channels * 2, kernel_size=(3,3), padding=1)]
		for layer in range(layers - 1):
			down_layers += [nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=(3,3), padding=1)]
		self.down_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
		self.down_model = nn.Sequential(*down_layers)

		if top:
			self.submodule = submodule
			up_layers = [nn.Conv2d(in_channels * 4, in_channels, kernel_size=(3, 3), padding=1)]
		elif not bottom:
			self.submodule = submodule
			up_layers = [nn.ConvTranspose2d(in_channels * 4, in_channels, kernel_size=(3, 3), stride=2, padding=1,
			                                 output_padding=1)]
		else:
			up_layers = [nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=(3, 3), stride=2, padding=1,
			                                 output_padding=1)]

		for i in range(layers - 1):
			up_layers += [nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1)]

		self.up_model = nn.Sequential(*up_layers)

	def forward(self, x):
		if self.bottom:
			x1 = self.down_model(x)
			x1 = self.up_model(x1)
			return x1
		else:
			x1 = self.down_model(x)
			x2 = self.down_pool(x1)
			x2 = self.submodule(x2)
			return self.up_model(torch.cat((x1, x2), dim=1))


class UNet(nn.Module):
	"""
		A fairly abstract but quite efficient UNet implementation,
		allowing for a variable number of blocks and layers per block
	"""
	def __init__(self, in_channels, out_channels, start_iblock_channels=32, num_down_blocks=4, layers_per_block=1):
		super(UNet, self).__init__()
		init_conv = nn.Conv2d(in_channels, start_iblock_channels, kernel_size=(7, 7), padding=3)

		unet_block = UNetBlock(start_iblock_channels * (2 ** (num_down_blocks - 1)), bottom=True, layers=layers_per_block)
		for i in range(num_down_blocks - 2):
			unet_block = UNetBlock(start_iblock_channels * (2 ** (num_down_blocks - i - 2)), submodule=unet_block, layers=layers_per_block)
		unet_block = UNetBlock(start_iblock_channels, submodule=unet_block, layers=layers_per_block, top=True)
		final_conv = nn.Conv2d(start_iblock_channels, out_channels, kernel_size=(7, 7), padding=3)
		self.model = nn.Sequential(init_conv, unet_block, final_conv, nn.Sigmoid())

	def forward(self, x):
		return self.model(x)


class UNetInception(nn.Module):
	def __init__(self, in_channels, out_channels, start_iblock_channels=32, num_down_blocks=4, layers_per_block=1):
		super(UNetInception, self).__init__()
		init_conv = nn.Conv2d(in_channels, start_iblock_channels, kernel_size=(7, 7), padding=3)

		unet_block = UNetInceptionBlock(start_iblock_channels * (3 ** (num_down_blocks - 1)),
		                                start_iblock_channels * (3 ** (num_down_blocks - 1)),
		                                bottom=True, layers=layers_per_block)
		for i in range(num_down_blocks - 2):
			unet_block = UNetInceptionBlock(start_iblock_channels * (3 ** (num_down_blocks - i - 2)),
			                                start_iblock_channels * (3 ** (num_down_blocks - i - 2)),
			                                submodule=unet_block,
			                                layers=layers_per_block)
		unet_block = UNetInceptionBlock(start_iblock_channels, start_iblock_channels, submodule=unet_block, top=True,
		                                layers=layers_per_block)
		final_conv = nn.Conv2d(start_iblock_channels, out_channels, kernel_size=(7, 7), padding=3)
		self.model = nn.Sequential(init_conv, unet_block, final_conv, nn.Sigmoid())

	def forward(self, x):
		return self.model(x)