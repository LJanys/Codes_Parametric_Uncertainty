import contextlib
import os
import shutil
import sys
import warnings
from enum import IntEnum
from functools import partial
from hashlib import sha1
from pathlib import Path
from time import time
from traceback import format_exception

import chaospy as cp
import numpy as np
import pandas as pd
import respy as rp

from decorators import numpy_interface

TAGS = IntEnum("TAGS", "RUN EXIT")
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data"

IMPLEMENTED_MODELS = ["kw_97_extended", "kw_97_basic"]


CHAOSPY_SAMPLING_METHODS = {
    "random",
    "grid",
    "chebyshev",
    "korobov",
    "sobol",
    "halton",
    "hammersley",
    "latin_hypercube",
}


def compute_ll(params, loglike_func, skip_existing):
    """Compute and save quantity of interest, given params and options."""
    params_hash = sha1(params["value"].to_numpy().tobytes()).hexdigest()
    save_path = RESULTS_DIR / f"{params_hash}.pickle"
    if skip_existing and save_path.exists():
        pass
    else:
        try:
            out = loglike_func(params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            tb = get_traceback()
            out = {"params": params, "exception": tb}
            warnings.warn(tb)

        pd.to_pickle(out, save_path)



def compute_qoi(params, simulate_func, skip_existing):
    """Compute and save quantity of interest, given params and options."""
    params_hash = sha1(params["value"].to_numpy().tobytes()).hexdigest()
    save_path = RESULTS_DIR / f"{params_hash}.pickle"

    if skip_existing and save_path.exists():
        pass
    else:
        with temporary_working_directory(params_hash):
            try:
                qois = _compute_qoi(params, simulate_func)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                tb = get_traceback()
                qois = {"params": params, "exception": tb}
                warnings.warn(tb)

        pd.to_pickle(qois, save_path)


def _compute_qoi(params, simulate_func):
    """Simulate model and evaluate quantity of interest.

    This function does the actual computations. Its public counterpart adds error
    handling, calls it from a temporary working directory, cleans up cached files
    and saves the result to disk.

    Note: This is only tested for kw_97_extended and reflects the quantities of
    interest that were calculated for that model.

    Args:
        params (pd.DataFrame): Parameter data frame used to simulate a respy model.
        options (dict): respy options dictionary corresponding to model

    Returns:
        dict: Nested dictionary with all relevant quantities of interest for a keane
            wolpin paper.

    """
    params = params[["value"]]

    # base case
    base_params = params.copy()
    base_df = simulate_func(base_params)

    # policy case
    subsidy_params = params.copy()
    subsidy_params.loc[("nonpec_school", "hs_graduate"), "value"] += 2000
    policy_df = simulate_func(subsidy_params)

    # preparations
    qois = {}
    base_df = base_df.copy()
    policy_df = policy_df.copy()
    vf_cols = [c for c in base_df.columns if "Value_Function" in c]
    base_df["Value_Function_Max"] = base_df[vf_cols].max(axis=1)
    policy_df["Value_Function_Max"] = policy_df[vf_cols].max(axis=1)
    last_period = base_df.index.get_level_values("Period").max()

    # calculate overall effect of tuition subsidy on years of education
    base_last_period = base_df.query(f"Period == {last_period}")
    policy_last_period = policy_df.query(f"Period == {last_period}")

    base_education = base_last_period["Experience_School"].mean()
    policy_education = policy_last_period["Experience_School"].mean()
    policy_effect = policy_education - base_education
    qois["subsidy_effect_on_years"] = {
        "difference": policy_effect,
        "baseline": base_education,
        "with_subsidy": policy_education,
    }

    # calculate effect of tuition subsidy on years of education by type
    base_education = base_last_period.groupby("Type")["Experience_School"].mean()
    policy_education = policy_last_period.groupby("Type")["Experience_School"].mean()
    policy_effect = policy_education - base_education
    qois["subsidy_effect_on_years_by_type"] = {
        "difference": policy_effect,
        "baseline": base_education,
        "with_subsidy": policy_education,
    }

    # average wages by period; for testing purposes because not degenerate even in
    # models with very few periods
    qois["average_wage_by_period"] = base_df.groupby("Period")[
        ["Wage_Blue_Collar", "Wage_White_Collar"]
    ].mean()

    # calculate overall effect of tuition subsidy on high school graduation
    base_high_school_share = (base_last_period["Experience_School"] >= 12).mean()
    policy_high_school_share = (policy_last_period["Experience_School"] >= 12).mean()
    qois["subsidy_effect_on_hs_graduation"] = {
        "difference": policy_high_school_share - base_high_school_share,
        "baseline": base_high_school_share,
        "with_subsidy": policy_high_school_share,
    }

    # calculate overall effect of tuition subsidy on college graduation
    base_college_share = (base_last_period["Experience_School"] >= 16).mean()
    policy_college_share = (policy_last_period["Experience_School"] >= 16).mean()
    qois["subsidy_effect_on_col_graduation"] = {
        "difference": policy_college_share - base_college_share,
        "baseline": base_college_share,
        "with_subsidy": policy_college_share,
    }

    # calculate subsidy effect on lifetime utility by type
    base_avg_utility = (
        base_df.query("Period == 0").groupby("Type")["Value_Function_Max"].mean()
    )
    policy_avg_utility = (
        policy_df.query("Period == 0").groupby("Type")["Value_Function_Max"].mean()
    )
    qois["subsidy_effect_on_utility_by_type"] = {
        "difference": policy_avg_utility - base_avg_utility,
        "baseline": base_avg_utility,
        "with_subsidy": policy_avg_utility,
    }

    # calculate choice shares after schooling until age 40
    # Note that we approximate "after schooling" by restricting the data to period 8 and larger
    qois["choice_shares_by_period"] = (
        base_df.groupby("Period")
        .Choice.value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    reduced = base_df.query("Period >= 8 & Period <= 24")
    if len(reduced) > 0:
        qois["average_choice_shares_by_type_until_age_40"] = (
            reduced.groupby(["Type"])
            .Choice.value_counts(normalize=True)
            .unstack()
            .fillna(0)
            .T
        )
    else:
        qois["average_choice_shares_by_type_until_age_40"] = None

    # calculate max schooling by type
    qois["years_educ_by_type"] = (
        base_df.query(f"Period == {last_period}")
        .groupby("Type")["Experience_School"]
        .mean()
    )

    # calculate 90 % result
    df_sample = base_df.groupby("Identifier").first()

    rslt_utilty = df_sample.groupby(["Type"])["Value_Function_Max"].mean()
    rslt_counts = df_sample.groupby(["Type"]).size()

    df_groups = pd.concat(
        [rslt_utilty, rslt_counts], axis=1, keys=["Group_Mean", "Count"]
    )
    df_groups["Grand_Mean"] = df_sample["Value_Function_Max"].mean()

    ssb = (
        df_groups["Count"] * (df_groups["Group_Mean"] - df_groups["Grand_Mean"]) ** 2
    ).sum()
    sst = np.var(df_sample["Value_Function_Max"]) * df_groups["Count"].sum()

    qois["variance_decomposition_of_lifetime_utility"] = 100 * ssb / sst

    # calculate average wages at age 50
    reduced = base_df.query("Period == 34")
    if len(reduced) > 0:
        qois["average_wage_at_age_50"] = reduced.groupby("Choice")[
            ["Wage_Blue_Collar", "Wage_White_Collar"]
        ].mean()
    else:
        qois["average_wage_at_age_50"] = None

    qois["params"] = params

    return qois


def load_model_specifications(
    model,
    monte_carlo_sequence,
    simulation_agents,
    solution_draws,
    n_periods,
):
    """Load model specifications from implemented respy model.

    Args:
        model (str): Type of model to simulate.
        monte_carlo_sequence (str): Type of sequence to use for Monte Carlo evaluations
            inside ``respy``.
        simulation_agents (int): Number of agents to simulate.
        solution_draws (int): Number of draws for the solution. Higher draws correspond
            to higher precision.

    Returns:
        params (pd.DataFrame): Model data frame.
        options (dict): Options used in respy stored in dictionary.
        constraints (list): List of constraints.
        n_periods (int): Number of periods to simulate.

    """
    _, options, data = rp.get_example_model(model, with_data=True)

    options["monte_carlo_sequence"] = monte_carlo_sequence
    options["simulation_agents"] = simulation_agents
    options["solution_draws"] = solution_draws
    options["n_periods"] = n_periods

    params = pd.read_pickle(DATA_PATH / f"mean_{model}.pkl")
    if "lower" not in params.columns:
        params["lower"] = -np.inf
        params["upper"] = np.inf

    constraints = rp.get_parameter_constraints(model)
    try:
        constraints.remove({"loc": "shocks_sdcorr", "type": "sdcorr"})
    except ValueError:
        pass

    return params, options, constraints, data


def distribute_tasks_ll(tasks, prob_info):

    num_proc = prob_info["n_proc"]

    if num_proc == 1:

        skip_existing = prob_info["skip_existing"]
        constraints = prob_info["constraints"]
        options = prob_info["options"]
        params = prob_info["params"]
        data = prob_info["data"]

        loglike_func = rp.get_log_like_func(params, options, data)
        ll = numpy_interface(params, constraints)(compute_ll)
        ll = partial(ll, loglike_func=loglike_func, skip_existing=skip_existing)

        [ll(task) for task in tasks]

    else:

        raise NotImplementedError("parallelization not implemented for ll")



def distribute_tasks(tasks, prob_info):

    num_proc = prob_info["n_proc"]

    if num_proc == 1:

        skip_existing = prob_info["skip_existing"]
        constraints = prob_info["constraints"]
        options = prob_info["options"]
        params = prob_info["params"]

        simulate_func = rp.get_simulate_func(params, options)
        qoi = numpy_interface(params, constraints)(compute_qoi)
        qoi = partial(qoi, simulate_func=simulate_func, skip_existing=skip_existing)

        [qoi(task) for task in tasks]

    else:

        num_children = min(len(tasks), num_proc)

        if "PMI_SIZE" not in os.environ.keys():
            raise AssertionError("requires MPI access")
        from mpi4py import MPI

        info = MPI.Info.Create()
        info.update({"wdir": os.getcwd()})

        file_ = os.path.dirname(os.path.realpath(__file__)) + "/child.py"
        comm = MPI.COMM_SELF.Spawn(
            sys.executable, args=[file_], maxprocs=num_children, info=info
        )

        # We send all problem-specific information once and for all.
        comm.bcast(prob_info, root=MPI.ROOT)

        status = MPI.Status()
        for task in tasks:
            comm.recv(status=status)
            rank_sender = status.Get_source()

            comm.send(None, tag=TAGS.RUN, dest=rank_sender)

            task = np.array(task, dtype="float64")
            comm.Send([task, MPI.DOUBLE], dest=rank_sender)

        # We are done and now terminate all child processes properly and finally the turn off
        # the communicator. We need for all to acknowledge the receipt to make sure we do
        # not continue here before all tasks are not only started but actually finished.
        [comm.send(None, tag=TAGS.EXIT, dest=rank) for rank in range(num_children)]
        [comm.recv() for _ in range(num_children)]
        comm.Disconnect()


def load_mean_and_cov(model):
    """Return mean and covariance for given model. The parameters in tha DATA_PATH refer to the
    updated parameters based on the new estimations using respy.

    Args:
        model (str): Which model to sample from. Possible models are in ['kw_97_basic',
            'kw_97_extended']

    Returns:
        mean, cov (pd.DataFrame, pd.DataFrame): Mean and covariance of the (normal)
            distribution.

    """
    cov = pd.read_pickle(DATA_PATH / f"cov_{model}.pkl")
    mean = pd.read_pickle(DATA_PATH / f"mean_{model}.pkl")
    mean = mean.reindex(cov.index)["value"]

    return mean, cov


def get_sampled_parameters(mean, cov, n_samples, method, seed):
    """Create a list of numpy arrays with sampled internal parameter vectors.

    Args:
        mean (pd.DataFrame or np.ndarray): The mean, of shape (k, ).
        cov (pd.DataFrame or np.ndarrary): The covariance, has to be of shape (k, k).

        n_samples (int): Number of samples to draw from ``distribution``.
        method (str): Which method to use for the sampling strategy. Possible values are
            given above. Examples include 'random' for standard monte carlo random
            sampling or 'latin_hypercube' for latin_hypercube sampling.
        seed (int): Seed for the random number generators.

    Returns:
        list: the sampled parameter vectors

    """
    np.random.seed(seed)
    distribution = cp.MvNormal(loc=mean, scale=cov)
    if method in CHAOSPY_SAMPLING_METHODS:
        sample_array = distribution.sample(size=n_samples, rule=method)
    else:
        raise ValueError(f"Argument 'method' is not in {CHAOSPY_SAMPLING_METHODS}.")

    return list(sample_array.T)


def check_arguments_of_create_samples(model, num_samples, method, num_proc, seed):
    """Assert input arguments of function `create_samples`.

    Args:
        model (str): Which model to sample from. Possible models are in ['kw_94_one',
            'kw_97_basic', 'kw_97_extended']
        num_samples (int): Number of samples to draw.
        method (str): Specifies which sampling method should be employed. Possible
            arguments are in {"random", "grid", "chebyshev", "korobov", "sobol",
            "halton", "hammersley", "latin_hypercube"}
        internal (bool): Should the internal or external parameters be used. Defaults
            to True.
        num_proc (int): Number of cores to use for the parallelization step.
        seed (int): Seed for the random number generator.

    Returns:
        None

    """
    assert (
        model in IMPLEMENTED_MODELS
    ), f"Argument 'model' must be in {IMPLEMENTED_MODELS}."
    assert num_samples > 0 and isinstance(
        num_samples, int
    ), "Argument 'num_samples' must be a positive integer."

    assert num_proc > 0 and isinstance(
        num_proc, int
    ), "Argument 'num_samples' must be a positive integer."

    assert (
        method in CHAOSPY_SAMPLING_METHODS
    ), f"Argument 'method' must be in {CHAOSPY_SAMPLING_METHODS}."

    assert (
        isinstance(seed, int) and seed >= 1
    ), "Argument 'seed' must be a positive integer."


@contextlib.contextmanager
def temporary_working_directory(snippet):
    """Changes working directory and returns to previous on exit.

    The name of the temporary directory is 'temp_snippet_process-id_timestamp'

    The directory is deleted upon exit.

    Args:
        snippet (str): String that will be part of the temporary directory

    """
    snippet = str(snippet)
    folder_name = f"temp_{snippet}_{os.getpid()}_{str(time()).replace('.', '')}"
    path = Path(".").resolve() / folder_name
    path.mkdir()
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(path)


def get_traceback():
    tb = format_exception(*sys.exc_info())
    if isinstance(tb, list):
        tb = "".join(tb)
    return tb
