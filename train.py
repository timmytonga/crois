import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import utils
from utils import AverageMeter, accuracy, save_onnx_model, get_subsampled_indices
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule

import pandas as pd
import os

# device = torch.device('cuda:0')


def run_epoch(
    epoch,
    model,
    optimizer,
    loader,
    loss_computer,
    logger,
    csv_logger,
    args,
    is_training,
    show_progress=False,
    log_every=50,
    scheduler=None,
    csv_name=None,
    wandb_group=None,
    wandb=None,
):
    """
    scheduler is only used inside this function if model is bert.
    """
    device = torch.device(f"cuda:{args.gpu}")
    if is_training:  # set model to train or eval
        model.train()
        if (args.model.startswith("bert") and args.use_bert_params):  # or (args.model == "bert"):
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):  # main train loop
            # get data
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            data_idx = batch[3]
            
            if args.model.startswith("bert"):  # handles bert:
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y,
                )[1]  # [1] returns logits
            else:
                # outputs.shape: (batch_size, num_classes)
                outputs = model(x)

            # running loss_computer computes other stats like group accuracy and such
            loss_main = loss_computer.loss(outputs, y, g, is_training)
            # update model
            if is_training:
                if (args.model.startswith("bert") and args.use_bert_params): 
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            output_df = pd.DataFrame()  # dataframe to save to output_{train/val/test}_epoch

            # Calculate stats -- get the prediction and compare with groundtruth -- save to output df
            if batch_idx == 0:
                acc_y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                acc_y_true = y.detach().cpu().numpy()
                acc_g_true = g.detach().cpu().numpy()
                indices = data_idx.detach().cpu().numpy()

                probs = outputs.detach().cpu().numpy()
            else:  # concatenate
                acc_y_pred = np.concatenate([
                    acc_y_pred,
                    np.argmax(outputs.detach().cpu().numpy(), axis=1)
                ])
                acc_y_true = np.concatenate([acc_y_true, y.detach().cpu().numpy()])
                acc_g_true = np.concatenate([acc_g_true, g.detach().cpu().numpy()])
                indices = np.concatenate([indices, data_idx.detach().cpu().numpy()])
                probs = np.concatenate([probs, outputs.detach().cpu().numpy()], axis=0)

            assert probs.shape[0] == indices.shape[0]
            run_name = f"{csv_name}_epoch_{epoch}_val"
            output_df[f"y_pred_{run_name}"] = acc_y_pred
            output_df[f"y_true_{run_name}"] = acc_y_true
            output_df[f"indices_{run_name}"] = indices
            output_df[f"g_true_{run_name}"] = acc_g_true

            for class_ind in range(probs.shape[1]):
                output_df[f"pred_prob_{run_name}_{class_ind}"] = probs[:, class_ind]
            # update csv logs and wandb
            if is_training and (batch_idx + 1) % log_every == 0:
                # this csv logger generates the train/val/test.csv that contains aggregate info per epoch
                run_stats = loss_computer.get_stats(model, args)
                csv_logger.log(epoch, batch_idx, run_stats)

                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
                if wandb is not None:  # log into wandb
                    wandb_stats = {
                        wandb_group + "/" + key: run_stats[key] for key in run_stats.keys()
                    }
                    wandb_stats["epoch"] = epoch
                    wandb_stats["batch_idx"] = batch_idx
                    wandb.log(wandb_stats)
        # save the model's classification on the dataset for this epoch
        # concern: the model changes as the data gets classified. Wonder how much impact this has
        if run_name is not None:
            save_dir = "/".join(csv_logger.path.split("/")[:-1])
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            output_df.to_csv(
                os.path.join(save_dir, 
                                f"output_{wandb_group}_epoch_{epoch}.csv"))
            print("Saved", os.path.join(save_dir, 
                                f"output_{wandb_group}_epoch_{epoch}.csv"))
        # log the final epoch -- might be repetitive: can just move code up there around?
        if (not is_training) or loss_computer.batch_count > 0:
            run_stats = loss_computer.get_stats(model, args)
            if wandb is not None:
                assert wandb_group is not None
                wandb_stats = {
                    wandb_group + "/" + key: run_stats[key] for key in run_stats.keys()
                }
                wandb_stats["epoch"] = epoch
                wandb_stats["batch_idx"] = batch_idx
                wandb.log(wandb_stats)
                print("logged to wandb")
            # this is the main thing that differ?
            csv_logger.log(epoch, batch_idx, run_stats)
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def train(
    model,
    criterion,
    dataset,
    logger,
    train_csv_logger,
    val_csv_logger,
    test_csv_logger,
    args,
    epoch_offset,
    csv_name=None,
    wandb=None,
    wandb_root_group=""
):
    if args.dataset == 'jigsaw':
        print("Jigsaw dataset... Performing validation on overlapping groups. ")
    # device = torch.device(f"cuda:{args.gpu}")
    # torch.cuda.set_device(args.gpu)
    model = model.cuda()  # the device should've been set universally in the beginning...

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(",")]
    assert len(adjustments) in (1, dataset["train_data"].n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * dataset["train_data"].n_groups)
    else:
        adjustments = np.array(adjustments)

    # setup train loss computer  -- this carries across epochs
    train_loss_computer = LossComputer(
        criterion,
        loss_type=args.loss_type,
        dataset=dataset["train_data"],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,
        joint_dro_alpha=args.joint_dro_alpha,
    )

    # BERT uses its own scheduler and optimizer
    if (args.model.startswith("bert") and args.use_bert_params): 
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr,
                          eps=args.adam_epsilon)
        t_total = len(dataset["train_loader"]) * args.n_epochs
        print(f"\nt_total is {t_total}\n")
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=args.warmup_steps,
                                         t_total=t_total)
    # not BERT. Sets up SGD optimizer with learning rate, weight decay and momentum.
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08,
            )
        else:
            scheduler = None

    best_val_acc = 0
    best_val_wg_acc = 0
    best_val_wg_epoch = 0
    # main epoch loop
    train_loader = dataset['train_loader']
    if args.multi_subsample:
        old_idxs = train_loader.dataset.dataset.indices

    print(f"Training for {args.n_epochs-epoch_offset} epochs...")
    for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
        logger.write("\nEpoch [%d]:\n" % epoch)
        logger.write(f"Training:\n")

        if args.multi_subsample:  # if we are subsampling per epoch, we modify the train_loader
            train_loader.dataset.dataset.indices = old_idxs  # initialize with old idxs
            subsampled_idxs = get_subsampled_indices(dataset['train_data'])
            train_loader.dataset.dataset.indices = subsampled_idxs

        # run train epoch
        run_epoch(
            epoch,
            model,
            optimizer,
            train_loader,
            train_loss_computer,
            logger,
            train_csv_logger,
            args,
            is_training=True,
            csv_name=csv_name,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
            wandb_group=f"{wandb_root_group}train",
            wandb=wandb,
        )

        logger.write(f"\nValidation:\n")
        val_loss_computer = LossComputer(
            criterion,
            loss_type=args.loss_type,
            dataset=dataset["val_data"],
            alpha=args.alpha,
            gamma=args.gamma,
            adj=adjustments,
            step_size=args.robust_step_size,
            normalize_loss=args.use_normalized_loss,
            btl=args.btl,
            min_var_weight=args.minimum_variational_weight,
            joint_dro_alpha=args.joint_dro_alpha,
        )
        run_epoch(
            epoch,
            model,
            optimizer,
            dataset["val_loader"],
            val_loss_computer,
            logger,
            val_csv_logger,
            args,
            is_training=False,
            csv_name=csv_name,
            wandb_group=f"{wandb_root_group}val",
            wandb=wandb,
        )
        if args.dataset == 'jigsaw':
            output_loc = os.path.join(args.log_dir, f"output_val_epoch_{epoch}.csv")
            utils.get_civil_comments_stats(epoch, output_loc, valortest='val', wandb=wandb, logger=None)

        # Test set; don't print to avoid peeking
        if dataset["test_data"] is not None:
            print(f"[{epoch}] Running test...")
            test_loss_computer = LossComputer(
                criterion,
                loss_type=args.loss_type,
                dataset=dataset["test_data"],
                step_size=args.robust_step_size,
                alpha=args.alpha,
                gamma=args.gamma,
                adj=adjustments,
                normalize_loss=args.use_normalized_loss,
                btl=args.btl,
                min_var_weight=args.minimum_variational_weight,
                joint_dro_alpha=args.joint_dro_alpha,
            )
            run_epoch(
                epoch,
                model,
                optimizer,
                dataset["test_loader"],
                test_loss_computer,
                None,
                test_csv_logger,
                args,
                is_training=False,
                csv_name=csv_name,
                wandb_group=f"{wandb_root_group}test",
                wandb=wandb,
            )
            print(f"[{epoch}] Done running test...")
            if args.dataset == 'jigsaw':
                output_loc = os.path.join(args.log_dir, f"output_test_epoch_{epoch}.csv")
                utils.get_civil_comments_stats(epoch, output_loc, valortest='test', wandb=wandb, logger=None)

        # Inspect learning rates
        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group["lr"]
                logger.write("Current lr: %f\n" % curr_lr)

        if args.scheduler and args.model != "bert":
            if args.loss_type == "group_dro":
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(
                    val_loss_computer.avg_group_loss,
                    val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(
                val_loss)  # scheduler step to update lr at the end of epoch

        if epoch < 5 or epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir,
                                           "%d_model.pth" % epoch))

        if args.save_last:
            # for saving
            x, _ = next(iter(dataset['train_loader']))
            torch.save(model, os.path.join(args.log_dir, "last_model.pth"))
            save_onnx_model(model, x, os.path.join(args.log_dir, "last_model.pth"))

        curr_val_wg_acc = min(val_loss_computer.avg_group_acc)
        if curr_val_wg_acc > best_val_wg_acc:
            best_val_wg_epoch = epoch
            best_val_wg_acc = curr_val_wg_acc
            logger.write(f"[e={best_val_wg_epoch}] Current Best Val Wg Acc = {best_val_wg_acc}")
            if wandb is not None:
                wandb.log({'val/best_wg_acc': best_val_wg_acc})
            if args.save_best:
                torch.save(model, os.path.join(args.log_dir, "best_wg_acc_model.pth"))

        if args.save_best:
            if args.loss_type == "group_dro" or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f"Current validation accuracy: {curr_val_acc}\n")
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, "best_model.pth"))
                logger.write(f"Best model saved at epoch {epoch}\n")

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(
                train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write("Adjustments updated\n")
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f"  {train_loss_computer.get_group_name(group_idx)}:\t"
                    f"adj = {train_loss_computer.adj[group_idx]:.3f}\n")
        logger.write("\n")
