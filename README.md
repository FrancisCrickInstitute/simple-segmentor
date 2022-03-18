### Usage
`python experiment.py --config [config-file]`  

e.g. `python experiment.py --config sequential.yml`  

When training is complete a results file will be written with
performance metrics and speed reports.

### Data preparation
Get the MitoEM-H images and labels from https://mitoem.grand-challenge.org/  

They should be stacked in order. The images should be 0-255 8 bit values
and the labels converted to binary values (any nonzero value in the original
labels should become a 1). Give the names and folder structure according to
the references in the config file.

### Tunable parameters
In the config file several parameters can be changed, and may
be possible to set to larger values depending on hardware.

patch_shape: controls how large the viewing window of the model is  
batch_size: controls how many samples are used in each optimizer step  
model: several parameters here can be changed to increase model complexity

