"""This script creates samples via distributed / parallelized computing.

The main function from this script, ``create_samples``, can either be imported from this
module or this script can be called directly from the command line with the respective
arguments. The later is necessary if one wishes to use distributed computing with mpi.
If it suffices to run the code in parallel on a single machine, then it is often more
convenient to import the function in another script, which still allows for
parallelization, but not distribution.


How to use this script from the command line:
---------------------------------------------

Example "Creating 1000 samples using 10 processes on a distributed machine":

    ``mpiexec -n 1 -usize 10 python create_sample.py -p 10 -s 1000``

For descriptions of the argument run

    ``python create_sample.py --help

"""
import os

# ensure automatic parallelism is turned off
n_threads = 1
update = dict.fromkeys(
    [
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MKL_NUM_THREADS",
    ],
    str(n_threads),
)
os.environ.update(update)

import click

import pandas as pd
from library import load_mean_and_cov
from library import distribute_tasks_ll
from library import load_model_specifications
from library import temporary_working_directory
from library import RESULTS_DIR
from batch_evaluators import joblib_batch_evaluator


def create_sample(
    start_index=0,
    stop_index=1,
    n_proc=1,
    n_periods=50,
    skip_existing=False,
):
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir()


    if n_proc == 1:
        _create_sample(
            start_index=start_index,
            stop_index=stop_index,
            n_periods=n_periods,
            skip_existing=skip_existing,
        )
    else:
        n_samples = stop_index - start_index
        batch_size = n_samples // n_proc
        args_list = []
        for i in range(start_index, stop_index, batch_size):
            args = {
                "start_index": i,
                "stop_index": min(i + batch_size, stop_index),
                "n_periods": n_periods,
                "skip_existing": skip_existing,
            }
            args_list.append(args)

        joblib_batch_evaluator(
            func=_create_sample,
            arguments=args_list,
            n_cores=n_proc,
            unpack_symbol="**",
        )



    out = [pd.read_pickle(path) for path in RESULTS_DIR.iterdir() if not "params_sample" in str(path)]

    return out


def _create_sample(
    start_index,
    stop_index,
    n_periods,
    skip_existing,
):
    # ==================================================================================
    # Restrict some options to the values we use in the paper so we can replace
    # the code for sampling the parameters by loading the sampled parameters.
    model = "kw_97_extended"
    n_proc = 1
    # ==================================================================================
    params, options, constraints, data = load_model_specifications(
        model=model,
        monte_carlo_sequence="sobol",
        simulation_agents=50_000,
        solution_draws=500,
        n_periods=n_periods,
    )

    # ==================================================================================
    # load the old parameter sample; Sampling freshly yields very similar parameters
    # but they differ after 11 decimal places. The old seed was 1234.
    mean, _ = load_mean_and_cov(model)
    full_sample = [mean.to_numpy()] + pd.read_pickle("old_results/params_sample.pickle")
    sampled_params = full_sample[start_index: stop_index]

    # ==================================================================================

    # collect data that has to be shared via mpi
    prob_info = {
        "skip_existing": skip_existing,
        "num_params": len(sampled_params[0]),
        "constraints": constraints,
        "n_periods": n_periods,
        "options": options,
        "params": params,
        "n_proc": n_proc,
        "model": model,
        "data": data,
    }

    # save the params sample; this is what produced the sample we are loading here.
    sample_dir = RESULTS_DIR / "params_sample.pickle"
    if not sample_dir.exists():
        pd.to_pickle(full_sample, sample_dir)

    with temporary_working_directory("tmp"):
        distribute_tasks_ll(sampled_params, prob_info)



@click.command()
@click.option(
    "--start_index", type=int, default=0, help="Start index."
)
@click.option(
    "--stop_index", type=int, default=1, help="Stop index."
)
@click.option(
    "--proc",
    "-p",
    type=int,
    default=1,
    help="Number of cores to parallelize the sampling.",
)
@click.option(
    "--periods",
    type=int,
    default=50,
    help="Number of periods for the model. Use a small value for debugging.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Whether qois that are already saved on disk are skipped or re-calculated.",
)
def main(
    start_index,
    stop_index,
    proc,
    periods,
    skip_existing,
):

    create_sample(
        start_index=start_index,
        stop_index=stop_index,
        n_proc=proc,
        n_periods=periods,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
