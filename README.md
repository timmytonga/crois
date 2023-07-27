# Classifier Retraining On Independent Split (CROIS)
This repo is built on top of the [Just Train Twice repo](https://github.com/anniesch/jtt). 
The requirements (located in `requirements.txt`) are:
```
matplotlib==3.3.1
torch==1.6.0
torchvision==0.7.0
Pillow==7.2.0
pytorch-transformers==1.2.0
numpy==1.19.4
pandas==1.1.2
tqdm==4.51.0
wandb  # optional
```


## Configuring the Dataset
Download and configure the 4 datasets following instructions from [Just Train Twice repo](https://github.com/anniesch/jtt).
More datasets can be configured by examining the code in the `data` folder. 

Be sure to configure the path to the data file in `data/utils.py` by setting 
`ROOT_DIR_PATH` appropriately. The default options expect the datasets to be in 
`ROOT_DIR_PATH/datasets/<dataset_name>`.


## Running CROIS
 One can run CROIS generically using `run_expt.py` by configuring
the arguments in the file. More help can be obtained by running 
```python run_expt.py --help```.


### Run default datasets with default scripts
We provide scripts to simplify running each of the dataset and run the 
two phases of CROIS using one command as configuring each dataset and its 
default parameters can be cumbersome. The files for running these scripts are in
`scripts/<dataset>_sweep.py`. The possible configurations for these scripts can be 
inspected in `script_utils/example_args.py`. The full details can also 
be obtained by running `python scripts/<dataset>_sweep.py --help`.

Some options like `-p` and `--part1_model_epochs` accept multiple inputs and the script will sweep over all the combinations
of those inputs. 
Below are example commands to run each of the dataset.

#### Waterbird
The following command will run both phases of CROIS (called "part1" and "part2" here) using the default hyperparameters.

```python scripts/cub_sweep.py -p 0.7 --seed 0 --part1_model_epochs -1 --part1_save_best --part2_loss_type group_dro --part2_reweight --run_part1```

The user can set different hyperparameters by setting options `--part1_n_epochs` and`--part2_lr` and so on. 
Here, only the loss type of `part2` is set to GDRO and reweighting is also used since it's part of the GDRO algorithm. 

Next, `p` denotes the split proportion. The default behavior is to split the train set. One can set the flag
`--val_split` to split the validation set instead. The flag on means the split proportion `p` is used to determine 
the proportion of examples to be used in the classifier retraining phase (`part2`) 
and the rest to be used for validation. 

The option `--part1_model_epochs` tells which epoch from `part1` should the feature extractor for `part2` load from. 
In the example, the option `-1` indicates that the *best* model (in terms of validation average accuracy) from `part1` 
will be used as the feature extractor for `part2`. In this case, the flag `--part1_save_best` should be set in order to
save the best model from `part1`. Similarly, if `--part1_model_epoch` is an actual epoch, the option `--part1_save_every`
should be set appropriately for the desired model to be saved. 

The user can explore other options and setting by calling the `--help` flag in any sweep script. 

#### CelebA, MultiNLI, and CivilComments
Once the user can run Waterbird as above, running the rest of the datasets is simple. Simply swap out `scripts/cub_sweep.py` to 
either `scripts/celebA_sweep.py` or `scripts/multinli_sweep.py` or `scripts/civilComments_sweep.py`. 
For example, running the below run both parts of the CivilComments dataset using CROIS with GDRO for `part2` with the best
feature extractor from `part1` and the split to be half of the validation set. 

```python scripts/civilComments_sweep.py -p 0.5 --val_split --seed 0 --part1_model_epochs -1 --part1_save_best --part2_loss_type group_dro --part2_reweight --run_part1```
