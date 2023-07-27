"""
Implements tau-normalization following [Kang' 20]: rescaling the weights of the last layer so that
    each column (representing the direction of each class in the feature space) has equal norm.
"""

import os
import argparse
from utils import set_seed, Logger, CSVBatchLogger, log_args, get_model, hinge_loss, split_data, check_args
import utils
import torch
from train import run_epoch
import numpy as np
import wandb
from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data, log_single_data
from data import dro_dataset
from loss import LossComputer


def pnorm(weights, p):
    """
    Tau_norm: Given weights, rescale each row by its norm to the pth power. (p=tau)
    """
    normB = torch.norm(weights, 2, 1)
    # print("pre-normalize: \n", weights)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    # print("post_normalize: ", ws)
    return ws


def run_eval(model, reweighted_lastlayer,
             dataloader, data, tau, logger,
             csv_logger, args, criterion,
             wandb, wandb_group='val'):
    """
    Run the model with the reweighted last layer on the dataset
    """
    old_weight = model.fc.weight.clone()
    model.fc.weight = torch.nn.Parameter(reweighted_lastlayer)
    # print("[run_eval debug] model fc weight: ", model.fc.weight)
    # adjustment and loss_computer stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(",")]
    assert len(adjustments) in (1, data.n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * data.n_groups)
    else:
        adjustments = np.array(adjustments)
    loss_computer = LossComputer(
            criterion,
            loss_type=args.loss_type,
            dataset=data,
            alpha=args.alpha,
            gamma=args.gamma,
            adj=adjustments,
            step_size=args.robust_step_size,
            normalize_loss=args.use_normalized_loss,
            btl=args.btl,
            min_var_weight=args.minimum_variational_weight,
            joint_dro_alpha=args.joint_dro_alpha,
        )
    # then run_epoch with the modified model
    run_epoch(
        epoch=tau,
        model=model,
        optimizer=None,
        loader=dataloader,
        loss_computer=loss_computer,
        logger=logger,
        csv_logger=csv_logger,
        args=args,
        is_training=False,
        show_progress=False,
        log_every=50,
        scheduler=None,
        csv_name=None,
        wandb_group=wandb_group,
        wandb=wandb,
    )
    # run_epoch(tau, x, device, None, dataset, None, writer, logger, is_training=False)
    model.fc.weight = torch.nn.Parameter(old_weight)


def main(args):
    """
        args is not completely done... we are reusing args from part2.
    """
    # first, set some configurations
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    #######################################
    #     Setup logging and wandb         #
    #######################################
    if args.wandb:
        only_last_layer = "oll" if args.part2_only_last_layer else "full"
        which_old_model = f"old-e{args.part1_model_epoch}" if args.part2_use_old_model else "new"
        real_group_labels = "_tgl" if args.use_real_group_labels else ""
        rw = "rw" if args.reweight_groups else ""
        group_name = f"part{args.part}{real_group_labels}_{rw}{args.loss_type}_{which_old_model}-{only_last_layer}" \
                     f"_wd{args.weight_decay}_lr{args.lr}"
        tags = ['tau_norm']
        run = wandb.init(project=f"{args.project_name}_{args.dataset}",
                         group=group_name,
                         tags=tags,
                         job_type=f"tau_norm",
                         name=f"p{args.part1_split_proportion}_seed{args.seed}")
        wandb.config.update(args)
    # Initialize logs
    if os.path.exists(args.log_dir) and args.resume:
        resume = True
        mode = "a"
    else:
        resume = False
        mode = "w"
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        print(f"******* Created dir {args.log_dir} **********")
    logger = Logger(os.path.join(args.log_dir, "log.txt"), 'w')
    # Record args
    logger.flush()
    log_args(args, logger)
    logger.flush()

    #######################################
    #     Prepare data                    #
    #######################################
    train_data, val_data, test_data = prepare_data(args, train=True)
    # Prepare data loader and csv_logger for each data
    loader_kwargs = {  # setting for args
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }
    data = {}
    val_loader = dro_dataset.get_loader(val_data,
                                        train=False,
                                        reweight_groups=None,
                                        **loader_kwargs)
    val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"val.csv"),
                                    val_data.n_groups,
                                    mode=mode)
    # potentially set test data
    test_csv_logger = None
    test_loader = None
    if args.run_test:
        test_loader = dro_dataset.get_loader(test_data,
                                             train=False,
                                             reweight_groups=None,
                                             **loader_kwargs)
        data["test_data"] = test_data
        data["test_loader"] = test_loader
        test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, f"test.csv"),
                                         test_data.n_groups,
                                         mode=mode)
    data["train_data"] = None
    data["val_data"] = val_data
    data["val_loader"] = val_loader
    log_data(data, logger)
    logger.flush()

    # Define the objective
    if args.hinge:
        assert args.dataset in ["CelebA", "CUB"]  # Only supports binary
        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
    #######################################
    #     Get model                       #
    #######################################
    # log file is as follows model_dir/tau_norm ... we need to go back one layer to get model path
    root_log_dir = os.path.dirname(
        args.log_dir)  # if log_dir is autogenerated, we save files where both stages use 1 level above
    model_path = os.path.join(root_log_dir, f"{args.model_epoch}_model.pth")
    # load saved model
    model = torch.load(model_path)
    logger.write(f"**** Loaded model from {model_path} ****\n")
    if args.wandb:
        wandb.watch(model)
    fc_weights, fc_bias = model.fc.weight, model.fc.bias

    #######################################
    #     Rescale last layer              #
    #######################################
    for p in np.linspace(args.min_tau, args.max_tau, args.step):
        ws = pnorm(fc_weights, p)
        run_eval(model, ws, val_loader, data=val_data,
                 tau=p, logger=logger,
                 csv_logger=val_csv_logger,
                 args=args, criterion=criterion,
                 wandb=wandb if args.wandb else None,
                 wandb_group='val')
        if args.run_test:
            run_eval(model, ws, test_loader, data=test_data,
                     tau=p, logger=logger,
                     csv_logger=test_csv_logger,
                     args=args, criterion=criterion,
                     wandb=wandb if args.wandb else None,
                     wandb_group='test')

    val_csv_logger.close()
    if args.run_test:
        test_csv_logger.close()
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    #######################################
    #     Load args/data/model            #
    #######################################
    # wandb, dataset, project_name, model_epoch,
    # group should be similar to part2 just jobtype = taunorm
    # log dir should be the dir that contain the models path
    parser = argparse.ArgumentParser()
    # add some common params
    utils.parser_add_settings(parser)
    utils.parser_add_objective_args(parser)
    utils.parser_add_data_args(parser)
    # Misc args: seed, log_dir, gpu, save_best, save_last, etc.
    utils.parser_add_misc_args(parser)
    # tau_norm specific
    parser.add_argument('--model_epoch', type=int, default=None,
                        help="which epoch of the model should we load")
    parser.add_argument('--min_tau', type=float, default=0.0, help="minimum tau_value to check")
    parser.add_argument('--max_tau', type=float, default=5.0, help="Maximum tau value to check")
    parser.add_argument('--step', type=int, default=51,
                        help="we check tau values in np.linspace(min_tau, max_tau, step)")
    parser.add_argument('--run_test', action='store_true', default=False)

    args = parser.parse_args()
    assert args.min_tau < args.max_tau, "min must less than max"
    main(args)
