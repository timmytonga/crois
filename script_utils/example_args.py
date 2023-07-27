import os
import run_expt
import tau_norm
import argparse
from utils import DotDict, ROOT_DIR_PATH

DEFAULT_BATCH_SIZE = 32  # default is 32
PROJECT_NAME = "split_pgl"


def get_standard_args(dataset, model, lr, wd, gpu, seed, wandb, log_dir, n_epochs,
                      part1_save_every, part1_use_all_data,
                      metadata_path, split_proportion, confounder_names, target_name,
                      metadata_csv_name, part,
                      data_root_dir=f"{ROOT_DIR_PATH}/datasets/",
                      batch_size=DEFAULT_BATCH_SIZE, project_name=PROJECT_NAME, show_progress=False, log_every=151):
    return DotDict({
        "q": 0.7,
        "lr": lr,
        "btl": False,
        "gpu": gpu,
        "fold": None,
        "seed": seed,
        "alpha": 0.2,
        "gamma": 0.1,
        "hinge": False,
        "model": model,
        "wandb": wandb,
        "resume": False,
        "aug_col": "None",
        "dataset": dataset,
        "log_dir": log_dir,
        "fraction": 1,
        "n_epochs": n_epochs,
        "root_dir": data_root_dir,
        "run_test": False,
        "log_every": log_every,
        "loss_type": "erm",
        "save_best": False,
        "save_last": False,
        "save_step": part1_save_every,
        "scheduler": False,
        # "up_weight": 0,
        "batch_size": batch_size,
        "num_sweeps": 4,
        "shift_type": "confounder",
        "target_name": target_name,
        "augment_data": False,
        "project_name": project_name,
        "val_fraction": 0.1,
        "weight_decay": wd,
        "subsample_minority": False,
        "multi_subsample": False,
        "metadata_path": metadata_path,
        "show_progress": show_progress,
        "imbalance_ratio": None,
        "joint_dro_alpha": 1,
        "reweight_groups": False,
        "use_bert_params": 1,
        "confounder_names": confounder_names,
        "robust_step_size": 0.01,
        "metadata_csv_name": metadata_csv_name,
        "minority_fraction": None,
        "train_from_scratch": False,
        "num_folds_per_sweep": 5,
        "use_normalized_loss": False,
        "automatic_adjustment": False,
        "generalization_adjustment": "0.0",
        "minimum_variational_weight": 0,
        "part": part,
        "part1_split_proportion": split_proportion,
        "val_split_proportion": 0,
        "part1_use_all_data": part1_use_all_data,
        "part1_model_epoch": 10,
        "part1_pgl_model_epoch": None,
        "part2_only_last_layer": False,
        "part2_use_old_model": False,
        "upweight": 0
    })


class TwoPartArgs:
    def __init__(self, dataset_name, model, lr, wd, gpu, seed, wandb, n_epochs,
                 part1_save_every, root_log, metadata_path, metadata_csv_path, split_proportion,
                 confounder_names, target_name, project_name, show_progress,
                 data_root_dir=f"{ROOT_DIR_PATH}/datasets/", part1_use_all_data=False,
                 log_every=151):
        self.part1_args = get_standard_args(
            part=1,
            dataset=dataset_name,
            model=model, lr=lr, wd=wd, gpu=gpu, seed=seed, wandb=wandb,
            log_dir=f"{root_log}/part1",
            n_epochs=n_epochs,
            part1_save_every=part1_save_every,
            part1_use_all_data=part1_use_all_data,
            data_root_dir=data_root_dir,
            metadata_path=metadata_path,
            metadata_csv_name=metadata_csv_path,
            split_proportion=split_proportion,
            confounder_names=confounder_names,
            target_name=target_name,
            project_name=project_name,
            show_progress=show_progress,
            log_every=log_every)

        self.part2_args = get_standard_args(
            part=2,
            dataset=dataset_name,
            model=model, lr=lr, wd=wd, gpu=gpu, seed=seed, wandb=wandb,
            log_dir=f"{root_log}/part2",
            n_epochs=n_epochs,
            part1_save_every=part1_save_every,
            part1_use_all_data=part1_use_all_data,
            data_root_dir=data_root_dir,
            metadata_path=metadata_path,
            metadata_csv_name=metadata_csv_path,
            split_proportion=split_proportion,
            confounder_names=confounder_names,
            target_name=target_name,
            project_name=project_name,
            show_progress=show_progress,
            log_every=log_every)


# --aug_col None --log_dir results/jigsaw/jigsaw_sample_exp/ERM_upweight_0_epochs_3_lr_1e-05_weight_decay_0.01/model_outputs
# --metadata_path results/jigsaw/jigsaw_sample_exp/metadata_aug.csv --lr 1e-05 --weight_decay 0.01 --up_weight 0
# --metadata_csv_name all_data_with_identities.csv --model bert-base-uncased --use_bert_params 0 --wandb --loss_type erm
class MyCivilCommentsArgs(TwoPartArgs):
    choices = [
        'male',
        'female',
        'christian',
        'muslim',
        'other_religion',
        'black',
        'white',
        'LGBTQ',
        'any_identity',
        'all'
    ]

    def __init__(self, n_epochs=6, wd=1e-2, lr=1e-5, part1_use_all_data=False,
                 upweight=0, run_name='civilComments_run', project_name='splitpgl',
                 only_last_layer=True, seed=0, wandb=True, show_progress=True,
                 split_proportion=0.5, gpu=0, part1_save_every=10, use_group="identity_any"):
        self.upweight = upweight
        self.only_last_layer = only_last_layer
        self.root_log = f"{ROOT_DIR_PATH}/pseudogroups/CivilComments/splitpgl_sweep_logs"
        self.ROOT_LOG = os.path.join(self.root_log,
                                     f"/SPGL_proportion{split_proportion}_epochs{n_epochs}_lr{lr}_weightdecay{wd}")
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

        confounder_names = all_groups if use_group == 'all' else [use_group]
        dataset_name = "jigsaw"
        target_name = "toxicity"
        metadata_csv_path = "all_data_with_identities.csv"
        model = "bert-base-uncased"
        metadata_path = None

        super().__init__(dataset_name, model, lr, wd, gpu, seed, wandb,
                         n_epochs, part1_save_every, self.ROOT_LOG,
                         metadata_path, metadata_csv_path, split_proportion,
                         confounder_names, target_name,
                         project_name, show_progress, part1_use_all_data=part1_use_all_data,
                         data_root_dir=f'{ROOT_DIR_PATH}/datasets/jigsaw')


class MyMultinliArgs(TwoPartArgs):
    def __init__(self, n_epochs=21, wd=0, lr=2e-5, part1_use_all_data=False,
                 upweight=0, run_name='multiNLI_run', project_name='splitpgl',
                 only_last_layer=True, seed=0, wandb=True, show_progress=True,
                 split_proportion=0.5, gpu=0, part1_save_every=10):
        self.upweight = upweight
        self.only_last_layer = only_last_layer
        self.root_log = f"{ROOT_DIR_PATH}/pseudogroups/MultiNLI/splitpgl_sweep_logs"
        self.ROOT_LOG = os.path.join(self.root_log,
                                     f"/SPGL_proportion{split_proportion}_epochs{n_epochs}_lr{lr}_weightdecay{wd}")
        confounder_names = ["sentence2_has_negation"]
        dataset_name = "MultiNLI"
        target_name = "gold_label_random"
        metadata_csv_path = "metadata_random.csv"
        model = "bert"
        metadata_path = None

        super().__init__(dataset_name, model, lr, wd, gpu, seed, wandb,
                         n_epochs, part1_save_every, self.ROOT_LOG,
                         metadata_path, metadata_csv_path, split_proportion,
                         confounder_names, target_name,
                         project_name, show_progress, part1_use_all_data=part1_use_all_data)


class MyCelebaArgs(TwoPartArgs):
    def __init__(self, n_epochs=51, wd=1e-5, lr=1e-5, part1_use_all_data=False,
                 upweight=0, run_name='celebA_run', project_name='noname',
                 only_last_layer=True, seed=0, wandb=True, show_progress=True,
                 split_proportion=0.5, gpu=0, part1_save_every=10, log_every=601):
        self.upweight = upweight
        self.only_last_layer = only_last_layer
        self.root_log = f"{ROOT_DIR_PATH}/pseudogroups/CelebA/splitpgl_sweep_logs"
        self.ROOT_LOG = os.path.join(self.root_log,
                                     f"/SPGL_proportion{split_proportion}_epochs{n_epochs}_lr{lr}_weightdecay{wd}")
        confounder_names = ["Male"]
        dataset_name = "CelebA"
        model = "resnet50"
        target_name = "Blond_Hair"
        metadata_csv_path = "list_attr_celeba.csv"
        metadata_path = f"myresults/celebA/{run_name}/metadata_aug.csv"

        super().__init__(dataset_name, model, lr, wd, gpu, seed, wandb,
                         n_epochs, part1_save_every, self.ROOT_LOG,
                         metadata_path, metadata_csv_path, split_proportion,
                         confounder_names, target_name,
                         project_name, show_progress, part1_use_all_data=part1_use_all_data,
                         log_every=log_every)


class MyCUBArgs(TwoPartArgs):
    def __init__(self, n_epochs=51, wd=1e-4, lr=1e-3, part1_use_all_data=False,
                 upweight=0, run_name='waterbird_newrun', project_name='splitpgl',
                 only_last_layer=True, seed=0, wandb=True, show_progress=True,
                 split_proportion=0.5, gpu=0, part1_save_every=10):
        self.upweight = upweight
        self.only_last_layer = only_last_layer
        self.root_log = f"{ROOT_DIR_PATH}/pseudogroups/CUB/splitpgl_sweep_logs"
        self.ROOT_LOG = os.path.join(self.root_log,
                                     f"/SPGL_proportion{split_proportion}_epochs{n_epochs}_lr{lr}_weightdecay{wd}")
        dataset_name = "CUB"
        model = "resnet50"
        target_name = "waterbird_complete95"
        confounder_names = ["forest2water2"]
        metadata_csv_path = "metadata.csv"
        metadata_path = f"myresults/CUB/{run_name}/metadata_aug.csv"
        data_root_dir = f"{ROOT_DIR_PATH}/datasets/cub"

        super().__init__(dataset_name, model, lr, wd, gpu, seed, wandb,
                         n_epochs, part1_save_every, self.ROOT_LOG,
                         metadata_path, metadata_csv_path, split_proportion,
                         confounder_names, target_name,
                         project_name, show_progress, data_root_dir=data_root_dir,
                         part1_use_all_data=part1_use_all_data)

    def set_param_both(self, param, value):
        self.part2_args[param] = value
        self.part1_args[param] = value
        # need to update log dir
        if param in ['n_epochs', 'weight_decay', 'lr']:
            N_EPOCHS = self.part1_args['n_epochs']
            LR = self.part1_args['lr']
            WEIGHT_DECAY = self.part1_args['weight_decay']
            log_root = os.path.join(self.root_log,
                                    f"ERM_upweight_0_epochs_{N_EPOCHS}_lr_{LR}_weight_decay_{WEIGHT_DECAY}")
            self.part1_args['log_dir'] = os.path.join(self.root_log, "model_outputs")
            self.part2_args['log_dir'] = os.path.join(log_root,
                                                      f"{'retrain' if self.only_last_layer else 'last_layer'}_part2_upweight{self.upweight}")


# TAU_NORM_ARGS
RUN_TAU_NORM = False
MIN_TAU, MAX_TAU, TAU_STEP = 1.0, 10.0, 101


def set_args_and_run_sweep(mainargsConstructor, args, PART2_USE_OLD_MODEL=True):

    project_name = "ValRgl" if args.val_split else f"{'Rgl' if not args.part2_use_pgl else 'Pgl'}"

    if args.jigsaw_use_group is not "any_identity":
        mainargs = mainargsConstructor(wandb=not args.no_wandb,
                                       seed=args.seed,
                                       show_progress=args.show_progress,
                                       project_name=project_name,
                                       gpu=args.gpu,
                                       part1_save_every=args.part1_save_every,
                                       part1_use_all_data=args.part1_use_all_data,
                                       use_group=args.jigsaw_use_group)
    else:
        mainargs = mainargsConstructor(wandb=not args.no_wandb,
                         seed=args.seed,
                         show_progress=args.show_progress,
                         project_name=project_name,
                         gpu=args.gpu,
                         part1_save_every=args.part1_save_every,
                         part1_use_all_data=args.part1_use_all_data)

    main_part1_args = mainargs.part1_args
    main_part2_args = mainargs.part2_args

    # part1 args
    main_part1_args.loss_type = args.part1_loss_type
    main_part1_args.lr, main_part1_args.weight_decay = args.part1_lr, args.part1_wd
    main_part1_args.n_epochs, main_part2_args.n_epochs = args.part1_n_epochs, args.part2_n_epochs
    main_part1_args.reweight_groups = args.part1_reweight
    main_part1_args.save_best = args.part1_save_best
    main_part1_args.run_test = args.run_test
    main_part1_args.batch_size = args.batch_size

    part1_log_lr = args.part1_lr  # this is to help with resuming the correct model
    if args.part1_resume_epoch >= 0:
        main_part1_args.resume = True
        main_part1_args.resume_epoch = args.part1_resume_epoch
        if args.part1_resume_lr is not None:
            part1_log_lr = args.part1_resume_lr

    # part 2 args
    main_part2_args.lr, main_part2_args.weight_decay = args.part2_lr, args.part2_wd
    main_part2_args.part2_only_last_layer = not args.part2_train_full
    main_part2_args.use_real_group_labels = not args.part2_use_pgl
    main_part2_args.part1_pgl_model_epoch = args.part1_pgl_model_epoch
    main_part2_args.loss_type = args.part2_loss_type
    main_part2_args.reweight_groups = args.part2_reweight
    main_part2_args.subsample_minority = args.part2_subsample
    main_part2_args.part2_use_old_model = PART2_USE_OLD_MODEL
    main_part2_args.multi_subsample = args.part2_multi_subsample
    main_part2_args.run_test = args.run_test
    main_part2_args.generalization_adjustment = args.part2_group_adjustment
    main_part2_args.batch_size = args.batch_size
    RUN_PART2 = not args.no_part2

    part2_log_lr = args.part2_lr  # this is to help with resuming the correct model
    if args.part2_resume_epoch >= 0:
        main_part2_args.resume = True
        main_part2_args.resume_epoch = args.part2_resume_epoch
        if args.part2_resume_lr is not None:
            part2_log_lr = args.part2_resume_lr

    # some log dir for part 1
    extra_part1 = f"{'_rw' if main_part1_args.reweight_groups else ''}" \
                  f"{main_part1_args.loss_type if main_part1_args.loss_type != 'erm' else ''}"

    # some log dir for part 2
    oll_part2 = "oll" if not args.part2_train_full else "full"
    extra_part2 = f"{'rw' if main_part2_args.reweight_groups else ''}" \
                  f"{'_subsample' if main_part2_args.subsample_minority else ''}" \
                  f"{'_rgl' if main_part2_args.use_real_group_labels else f'_pgl{args.part1_pgl_model_epoch}'}"
    extra_part2 += f'_ga{main_part2_args.generalization_adjustment}' if main_part2_args.generalization_adjustment != '0.0' else ''

    # tau norm args
    tau_norm_args = DotDict(main_part2_args.copy())
    tau_norm_args['model_epoch'] = main_part2_args.n_epochs - 1
    tau_norm_args['min_tau'], tau_norm_args['max_tau'], tau_norm_args['step'] = MIN_TAU, MAX_TAU, TAU_STEP
    tau_norm_args['run_test'] = True

    if args.part1_model_epochs is None:
        p1me = [args.part1_n_epochs - 1]
    else:
        p1me = args.part1_model_epochs

    print(f"Run with p1me {p1me} and p {args.p} and {'val split' if args.val_split else 'train split'}")
    pname_stem = 'valp' if args.val_split else 'p'
    for p in args.p:
        # prep part 1 and 2 proportions params
        if args.val_split:
            main_part1_args.val_split_proportion = p
            main_part2_args.val_split_proportion = p
            tau_norm_args.val_split_proportion = p
        else:
            main_part1_args.val_split_proportion = 0  # Don't really wanna set this to anything non-zero here...
            main_part2_args.val_split_proportion = 0
            tau_norm_args.val_split_proportion = 0

            main_part1_args.part1_split_proportion = p
            main_part2_args.part1_split_proportion = p
            tau_norm_args.part1_split_proportion = p

        # make some log dirs
        pname = pname_stem + str(p)
        stem = f"{'all' if (args.part1_use_all_data or args.val_split) else pname}{extra_part1}" \
               f"_wd{args.part1_wd}_lr{part1_log_lr}"
        root_log = os.path.join(mainargs.root_log, stem)
        main_part1_args.log_dir = os.path.join(root_log, f"part1_s{args.seed}")

        # run part1
        if args.run_part1:  # ensure we have already run part 1 if this is set to False
            run_expt.main(main_part1_args)
            if args.part1_use_all_data or args.val_split:
                print(f"******** [PART1_USE_ALL_DATA] SKIPPING TRAINING PART1 FOR p={p} SINCE ALREADY TRAINED ON ALL "
                      f"DATA *******")
                args.run_part1 = False  # since we will be training the same model again

        # now run part2
        for part1_model_epoch in p1me:
            main_part2_args.part1_model_epoch = part1_model_epoch
            print(f"Running {pname} and p1me={part1_model_epoch}")
            main_part2_args.log_dir = os.path.join(root_log, f"part2_{oll_part2}{part1_model_epoch}{extra_part2}_"
                                                             f"{args.part2_loss_type}_{pname}_wd{args.part2_wd}"
                                                             f"_lr{part2_log_lr}"
                                                             f"_s{args.seed}")
            tau_norm_args.log_dir = os.path.join(main_part2_args.log_dir, "tau_norm")
            if RUN_PART2:
                run_expt.main(main_part2_args)
            if RUN_TAU_NORM:
                tau_norm.main(tau_norm_args)


def set_two_parts_args(seed=0, p=(0.3, 0.5, 0.7), gpu=0,
                       part1_wd=1e-4, part1_lr=1e-4, part1_n_epochs=51,
                       part2_wd=1e-4, part2_lr=1e-4, part2_n_epochs=51, batch_size=DEFAULT_BATCH_SIZE):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("-p", nargs="+", type=float, default=p)
    parser.add_argument("--val_split", action="store_true", default=False,
                        help="Set this to use p on the validation set. "
                             "Here, p determines the proportion of the validation set to finetune part2.")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=gpu)
    parser.add_argument("--show_progress", action="store_true", default=False)
    parser.add_argument("--run_test", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    # part1 args
    parser.add_argument("--part1_wd", type=float, default=part1_wd)
    parser.add_argument("--part1_lr", type=float, default=part1_lr)
    parser.add_argument("--part1_loss_type", default="erm",
                        choices=["erm", "group_dro", "joint_dro"])
    parser.add_argument("--part1_reweight", action="store_true", default=False)
    parser.add_argument("--run_part1", action="store_true", default=False)
    parser.add_argument("--part1_save_every", type=int, default=10)
    parser.add_argument("--part1_save_best", action="store_true", default=False)
    parser.add_argument("--part1_n_epochs", type=int, default=part1_n_epochs)
    parser.add_argument("--part1_use_all_data", action="store_true", default=False)
    parser.add_argument("--part1_resume_epoch", type=int, default=-1)
    parser.add_argument("--part1_resume_lr", type=float, default=None,
                        help="Resume lr is used when we want to resume the run from a particular epoch "
                             "but with a different lr")
    # python scripts/cub_sweep.py -p 0.7 --seed 0 --part1_model_epochs -1
    # part2 args
    parser.add_argument("--part1_model_epochs", nargs="+", type=int, default=None,
                        help="Which model epoch to retrain part2 on. Use -1 to ")
    parser.add_argument("--part2_use_pgl", action="store_true", default=False)
    parser.add_argument("--part1_pgl_model_epoch", type=int, default=None)
    parser.add_argument("--part2_loss_type", default="erm",
                        choices=["erm", "group_dro", "joint_dro"])
    parser.add_argument("--part2_subsample", action="store_true", default=False)
    parser.add_argument("--part2_reweight", action="store_true", default=False)
    parser.add_argument("--part2_lr", type=float, default=part2_lr)
    parser.add_argument("--part2_wd", type=float, default=part2_wd)
    parser.add_argument("--part2_n_epochs", type=int, default=part2_n_epochs)
    parser.add_argument("--no_part2", action="store_true", default=False)
    parser.add_argument("--part2_resume_epoch", type=int, default=-1,
                        help="Which epoch are we resuming from")
    parser.add_argument("--part2_resume_lr", type=float, default=None,
                        help="Resume lr is used when we want to resume the run from a particular epoch "
                             "but with a different lr")
    parser.add_argument("--part2_multi_subsample", action="store_true", default=False)
    parser.add_argument("--part2_group_adjustment", type=str, default="0.0",
                        help="This set the group adjustment parameter for retraining")
    parser.add_argument("--tau_norm_after_part2", action="store_true", default=False)
    parser.add_argument("--part2_train_full", action="store_true", default=False,
                        help="By default we are only retraining the last layer for part2. "
                             "Set this if want to retrain all.")

    parser.add_argument("--jigsaw_use_group", choices=MyCivilCommentsArgs.choices, default='any_identity',
                        help="Specify which group to use. Can specify multiple groups.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    test_args = MyCUBArgs()
    part1_args = test_args.part1_args
    part2_args = test_args.part2_args
