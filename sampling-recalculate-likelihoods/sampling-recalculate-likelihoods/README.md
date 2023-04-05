# Sampling


## Environment

The code in this repository requires the exact environment specified in `environment.yml` to run. It is not guaranteed that it will run or produce correct results with later versions of respy or other packages.

## Run command

To test that everything runs, run:

`python create_sample.py`

The command has the following options:

- `--start_index`: The start parameter index, Default `--start_index 0`
- `--stop_index`: The stop parameter index, Default `--start_index 1`
- `--proc`: The number of cores. Default `--proc 1`, i.e. no parallelization
- `--periods` The number of model periods. Must be 50 for all actual results. Can be lower for faster debugging. Default `--periods 50`
- `--skip-existing`: Whether results that are found on disks are skipped or re-calculated. Default `--skip-existing False`


## What will be produced

When running the command for the first time, a directory called `results` will be
produced. This will contain a file called `params_sample.pickle`. This file contains all sampled parameters as numpy arrays. The first entry, is the point estimate. All other entries are randomly drawn from the asymptotic distribution.

**The old params sample file did not contain the point estimate!**

The results directory will also contain one pickle file per parameter vector. This pickle file contains the log likelihood of that parameter vector. The name of the pickle file is the sha1 hash of the parameter vector.

The names of the likelihood pickle files can be matched to the names of the qoi files in Philipp's results.
