from script_utils.example_args import MyMultinliArgs, set_args_and_run_sweep, set_two_parts_args


args = set_two_parts_args(seed=0,
                          p=[0.3, 0.5, 0.7],
                          gpu=0,
                          part1_lr=2e-5,
                          part1_wd=0,
                          part1_n_epochs=21,
                          part2_lr=2e-5,
                          part2_wd=0,
                          part2_n_epochs=21)

# run with args
set_args_and_run_sweep(MyMultinliArgs, args)
