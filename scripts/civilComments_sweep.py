from script_utils.example_args import MyCivilCommentsArgs, set_args_and_run_sweep, set_two_parts_args


args = set_two_parts_args(seed=0,
                          p=[0.3, 0.5, 0.7],
                          gpu=0,
                          part1_lr=1e-5,
                          part1_wd=1e-2,
                          part1_n_epochs=6,
                          part2_lr=1e-5,
                          part2_wd=0,
                          part2_n_epochs=6,
                          batch_size=16)

# run with args
set_args_and_run_sweep(MyCivilCommentsArgs, args)
