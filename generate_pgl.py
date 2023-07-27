import os
from train import run_epoch
from loss import LossComputer
import numpy as np
import pandas as pd
import torch
from utils import Logger, CSVBatchLogger
from data import dro_dataset
from torch.utils.data.sampler import WeightedRandomSampler
from copy import deepcopy


def generate_pgl(part1_model_path, part2_data, train_data, args):
    n_classes = train_data.n_classes
    part2eval_log_dir = os.path.join(args.log_dir, "part2_eval")
    if not os.path.exists(part2eval_log_dir):
        os.makedirs(part2eval_log_dir)
    part2eval_logger = Logger(os.path.join(part2eval_log_dir, "log.txt"), 'w')
    part2eval_logger.flush()

    # first load the previous model
    modeleval = torch.load(part1_model_path)
    modeleval.cuda()
    # initialize logger and loader for part2
    part2eval_csv_logger = CSVBatchLogger(os.path.join(part2eval_log_dir, f"part2_eval.csv"),
                                          part2_data.n_groups,
                                          mode='w')
    loader_kwargs = {  # setting for args
        "batch_size": 128,
        "num_workers": 4,
        "pin_memory": True,
    }

    part2_loader = dro_dataset.get_loader(part2_data,
                                          train=False,
                                          reweight_groups=None,
                                          **loader_kwargs)
    adjustments = np.array([0] * 4)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    part2eval_loss_computer = LossComputer(
        criterion,
        loss_type=args.loss_type,
        dataset=part2_data,
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
    )

    # then run an epoch on part2 and during that run, generate a csv containing the status of each example
    run_epoch(
        epoch=0,
        model=modeleval,
        optimizer=None,
        loader=part2_loader,
        loss_computer=part2eval_loss_computer,
        logger=part2eval_logger,
        csv_logger=part2eval_csv_logger,
        args=args,
        is_training=False,
        show_progress=True,
        log_every=50,
        scheduler=None,
        csv_name="pseudogroup_eval",
        wandb_group="part2_eval",
        wandb=None,
    )
    part2_df = pd.read_csv(os.path.join(part2eval_log_dir, 'output_part2_eval_epoch_0.csv'))
    n_groups = n_classes*2
    true_y = part2_df['y_true_pseudogroup_eval_epoch_0_val']
    pred_y = part2_df['y_pred_pseudogroup_eval_epoch_0_val']
    # true_g = part2_df['g_true_pseudogroup_eval_epoch_0_val']
    misclassified = true_y != pred_y
    # sampler, upsampled_part2 = None, None
    group_array = (misclassified + true_y * n_classes).astype("int")  # times 2 to make group (true_y, status)
    part2_pgl_data = deepcopy(part2_data)
    part2_pgl_data.set_group_array(group_array, n_groups)
    # print(f"DEBUG: part2_pgl_data group count {part2_pgl_data.group_counts()} "
    #       f"vs. real group count {get_group_counts(group_array, n_groups)}")
    if args.upweight == 0:  # this means we do equal sampling
        upsampled_part2 = None
        # group_weights = len(part2_data) / group_counts
        # weights = group_weights[group_array]
        # sampler = WeightedRandomSampler(weights, len(part2_data), replacement=True)
    else:  # this means we upweight
        assert args.upweight == -1 or args.upweight > 0
        aug_indices = part2_df['indices_pseudogroup_eval_epoch_0_val'][misclassified]
        upweight_factor = len(part2_df) // len(aug_indices) if args.upweight == -1 else args.upweight
        print(f"UPWEIGHT FACTOR = {upweight_factor}")
        combined_indices = list(aug_indices) * upweight_factor + list(part2_df['indices_pseudogroup_eval_epoch_0_val'])
        upsampled_part2 = torch.utils.data.Subset(train_data.dataset.dataset, combined_indices)

    return part2_pgl_data, upsampled_part2


def get_group_counts(group_array, n_groups):
    group_counts = (torch.arange(n_groups).unsqueeze(1) == torch.tensor(group_array)).sum(1).float()
    return group_counts

def analyze_pgl_quality(pgl, true_g):
    """
    This means we have to match minority class with none. Implement some group mapping logic here...
    """
    pass
