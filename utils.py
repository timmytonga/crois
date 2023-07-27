import sys
import os
import torch
import numpy as np
import csv

import torch
import torch.nn as nn
import torchvision
from models import model_attributes
from data.folds import Subset, ConcatDataset
from data.data import dataset_attributes, shift_types
import data
from data.utils import ROOT_DIR_PATH
import pandas as pd
from pprint import pformat

pd.options.mode.chained_assignment = None  # default='warn'  suppress pesky warning


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Logger(object):
    def __init__(self, fpath=None, mode="w"):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    # def __del__(self):
    #     self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode="w"):
        columns = ["epoch", "batch"]
        for idx in range(n_groups):
            columns.append(f"avg_loss_group:{idx}")
            columns.append(f"exp_avg_loss_group:{idx}")
            columns.append(f"avg_acc_group:{idx}")
            columns.append(f"processed_data_count_group:{idx}")
            columns.append(f"update_data_count_group:{idx}")
            columns.append(f"update_batch_count_group:{idx}")
        columns.append("avg_actual_loss")
        columns.append("avg_per_sample_loss")
        columns.append("avg_acc")
        columns.append("model_norm_sq")
        columns.append("reg_loss")
        columns.append("wg_acc")

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode == "w":
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict["epoch"] = epoch
        stats_dict["batch"] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_args(args):
    assert 1 >= args.part1_split_proportion >= 0 and 1 >= args.val_split_proportion >= 0, \
        f"split proportion must be in [0,1]. Part1_p = {args.part1_split_proportion} and val_p = {args.val_split_proportion}"

    if args.val_split_proportion > 0 and args.part1_split_proportion < 1:
        print("Warning: ONLY USING VAL_SPLIT (since it's set) and not TRAIN_SPLIT...")

    if args.shift_type == "confounder":
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith("label_shift"):
        assert args.minority_fraction
        assert args.imbalance_ratio


def split_data(dataset, part1_proportion=0.5, seed=None):
    """
        Split data into 2 parts given a ratio. Return part1, part2
    """
    random = np.random.RandomState(seed)
    n_exs = len(dataset)
    n_part1 = int(n_exs*part1_proportion)
    # n_part2 = n_exs - n_part1
    if type(dataset) == Subset:  # note that this Subset is data.folds.Subset
        idxs = np.random.permutation(dataset.indices)
        data = dataset.dataset
    else:  # this is not a Subset i.e. just an ordinary dataset
        idxs = np.random.permutation(np.arange(len(dataset)))
        data = dataset
    part1, part2 = Subset(data, idxs[:n_part1]), Subset(data, idxs[n_part1:])  # this is not torch subset
    return part1, part2


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_args(args, logger):
    if type(args) is DotDict:
        argdict = args.items()
    else:
        argdict = vars(args).items()
    for argname, argval in argdict:
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write("\n")


def hinge_loss(yhat, y):
    # The torch loss takes in three arguments so we need to split yhat
    # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
    # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
    # so we need to swap yhat[:, 0] and yhat[:, 1]...
    torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction="none")
    y = (y.float() * 2.0) - 1.0
    return torch_loss(yhat[:, 1], yhat[:, 0], y)


def parser_add_objective_args(parser):
    """
        helper to add certain parameters to args
    """
    # Objective
    parser.add_argument("--loss_type", default="erm",
                        choices=["erm", "group_dro", "joint_dro"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--generalization_adjustment", default="0.0")
    parser.add_argument("--automatic_adjustment",
                        default=False,
                        action="store_true")
    parser.add_argument("--robust_step_size", default=0.01, type=float)
    parser.add_argument("--joint_dro_alpha", default=1, type=float,
                        help=("Size param for CVaR joint DRO."
                              " Only used if loss_type is joint_dro"))
    parser.add_argument("--use_normalized_loss",
                        default=False,
                        action="store_true")
    parser.add_argument("--btl", default=False, action="store_true")
    parser.add_argument("--hinge", default=False, action="store_true")


def parser_add_settings(parser):
    # wandb: wandb.ai
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="splitpgl", help="wandb project name")
    # Resume?
    parser.add_argument("--resume", default=False, action="store_true")


def parser_add_data_args(parser):
    # Settings
    parser.add_argument("-d",
                        "--dataset",
                        choices=dataset_attributes.keys(),
                        required=True)
    parser.add_argument("-s",
                        "--shift_type",
                        choices=shift_types,
                        required=True)
    # Confounders
    parser.add_argument("-t", "--target_name")
    parser.add_argument("-c", "--confounder_names", nargs="+")
    # Label shifts
    parser.add_argument("--minority_fraction", type=float)
    parser.add_argument("--imbalance_ratio", type=float)
    # Data
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--reweight_groups", action="store_true",
                        default=False,
                        help="set to True if loss_type is group DRO")
    parser.add_argument("--augment_data", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0.1)


def parser_add_misc_args(parser):
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--save_last", action="store_true", default=False)
    parser.add_argument("--use_bert_params", type=int, default=1)
    parser.add_argument("--num_folds_per_sweep", type=int, default=5)
    parser.add_argument("--num_sweeps", type=int, default=4)
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        "--metadata_csv_name",
        type=str,
        default="metadata.csv",
        help="name of the csv data file (dataset csv has to be placed in dataset folder).",
    )
    parser.add_argument("--fold", default=None)


def parser_add_optimization_args(parser):
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--minimum_variational_weight", type=float, default=0)


def get_model(model, pretrained, resume, n_classes, dataset, log_dir, train_data=None):
    if resume:
        model = torch.load(os.path.join(log_dir, "last_model.pth"))
        d = train_data.input_size()[0]
    elif model_attributes[model]["feature_type"] in (
            "precomputed",
            "raw_flattened",
    ):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif model == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "wideresnet50":
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model.startswith('bert'):
        if dataset == "MultiNLI":
            
            assert dataset == "MultiNLI"

            from pytorch_transformers import BertConfig, BertForSequenceClassification

            config_class = BertConfig
            model_class = BertForSequenceClassification

            config = config_class.from_pretrained("bert-base-uncased",
                                                num_labels=3,
                                                finetuning_task="mnli")
            model = model_class.from_pretrained("bert-base-uncased",
                                                from_tf=False,
                                                config=config)
        elif dataset == "jigsaw":
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                model,
                num_labels=n_classes)
            print(f'n_classes = {n_classes}')
        else: 
            raise NotImplementedError
    else:
        raise ValueError(f"{model} Model not recognized.")

    return model


def save_onnx_model(model, an_input, path, wandb_=None):
    torch.onnx.export(model, an_input, path)
    if wandb_ is not None:
        wandb_.save(path)


def get_subsampled_indices(train_data: data.dro_dataset.DRODataset) -> np.array:
    """ Given a DRODataset, return a np.array of indices of the dataset after subsampling"""
    smallest_group_size = np.min(np.array(train_data.group_counts().cpu(), dtype=int))
    # print('smallest group size = ', smallest_group_size)
    indices = np.array([], dtype=int)
    group_array = train_data.get_group_array()
    for g in np.arange(train_data.n_groups):
        group_indices = np.where((group_array == g))[0]
        indices = np.concatenate((
            indices, np.sort(np.random.permutation(group_indices)[:smallest_group_size])))
        # print(f'g indices (len {len(group_indices)})', group_indices)
    # print("final len", len(indices))
    return indices


def _cc_get_output_and_meta_df(epoch, file_path):
    meta_data_path = os.path.join(ROOT_DIR_PATH, 'datasets/jigsaw/data/all_data_with_identities.csv')
    # file_path = root_dir + f'{part}_s{seed}/output_{val_or_test}_epoch_{epoch}.csv'
    output_df = pd.read_csv(file_path)
    metadata_df = pd.read_csv(meta_data_path)
    test_df = metadata_df.iloc[output_df[f'indices_None_epoch_{epoch}_val']]
    test_df['labels'] = (test_df['toxicity'] >= 0.5).astype(int)  # this gives a warning...
    test_df.reset_index(inplace=True)

    return output_df, test_df


def _cc_analyze_accs(output_df, test_df, epoch, valortest=None):
    all_groups = [
        'male',
        'female',
        'christian',
        'muslim',
        'other_religion',
        'black',
        'white',
        'LGBTQ'
    ]
    pred_col_name = f'y_pred_None_epoch_{epoch}_val'
    group_acc_dict = {}
    group_n_dict = {}
    for toxic in range(2):  # in 0 or 1
        for g in range(len(all_groups)):
            group_key = f"{valortest}/{(all_groups[g], toxic)}"
            group_idx = toxic * len(all_groups) + g
            idxs = (test_df['labels'] == toxic) & (test_df[all_groups[g]] == 1)
            total_n_g = sum(idxs)
            group_n_dict[group_key] = total_n_g
            if total_n_g <= 0:
                group_acc_dict[group_key] = 1.1  # vacuously perfect... but set to 1.1 to distinguish
                continue
            correct_pred = sum(test_df[idxs]['labels'] == output_df[idxs][pred_col_name])
            group_acc_dict[group_key] = correct_pred / total_n_g
    return group_acc_dict, group_n_dict


def get_civil_comments_stats(epoch, file_path, valortest=None, wandb=None, logger=None):
    output_df, test_df = _cc_get_output_and_meta_df(epoch, file_path)
    group_acc_dict, group_n_dict = _cc_analyze_accs(output_df, test_df, epoch, valortest)
    group_acc_dict[f"{valortest}/true_wg_acc"] = min(v for k, v in group_acc_dict.items())
    group_acc_dict["epoch"] = epoch
    # avg_acc = sum(output_df[pred_col_name] == output_df[true_col_name]) / len(output_df)
    if logger is not None:
        logger.write(pformat(group_acc_dict)+"\n")
        logger.flush()
    if wandb is not None:
        wandb.log(group_acc_dict)
    print(f"finish get_civil_comments_stats for {valortest} epoch {epoch}")
