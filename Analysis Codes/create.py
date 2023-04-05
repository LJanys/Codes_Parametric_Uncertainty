from functools import partial
import subprocess as sp
import pickle as pkl
import glob
import os

from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
import respy as rp

from library_decision_theory import get_within_confidence_set
from library_decision_theory import get_decisions_uncertainty
from library_decision_theory import get_decisions_certainty


def extract_quantity(results, label):
    stats = list()
    for result in results:
        if "decomposition" in label or "choice" in label:
            stats.append(result[label])
        elif "for_type_" in label:
            key_ = "subsidy_effect_on_years_by_type"
            type_ = int(label[-1])
            stats.append(result[key_]["difference"][type_])
        else:
            stats.append(result[label]["difference"])
    return stats


def get_params_df():

    params_list = pkl.load(open("results/params_sample.pickle", "rb"))

    mean_base = pd.read_pickle("mean_kw_97_extended.pkl")
    param_index = mean_base.index[~mean_base["se"].isnull()]

    df_params = pd.DataFrame(columns=param_index, index=range(len(params_list)))
    df_params.index.name = "Sample"

    for i, values in enumerate(params_list):
        df_params.loc[i, param_index] = values
    df_params.to_pickle("df-params.pkl")

    return df_params


def get_quantities_df():
    results = list()
    for fname in glob.glob("results/*.pickle"):
        if "params_sample" in fname:
            continue
        result = pkl.load(open(fname, "rb"))
        results.append(result)
    pkl.dump(results, open("results-quantities.pkl", "wb"))

    results = pkl.load(open("results-quantities.pkl", "rb"))
    p_extract_quantity = partial(extract_quantity, results)

    labels = list()
    labels += ["choice_shares_by_period"]
    labels += ["subsidy_effect_on_years", "subsidy_effect_on_hs_graduation"]
    labels += ["subsidy_effect_on_col_graduation", "variance_decomposition_of_lifetime_utility"]
    labels += [f"subsidy_effect_on_years_for_type_{type_}" for type_ in range(4)]

    df_quantities = pd.DataFrame()
    for label in labels:
        df_quantities[label] = p_extract_quantity(label)
    df_quantities.index.name = "Sample"

    df_quantities.to_pickle("df-quantities.pkl")

    return df_quantities


def get_final_sample(df_quantities, df_params):
    def get_rmse(probs_sim, probs_obs):
        x = probs_sim.loc[:10, columns].unstack()
        y = probs_obs.loc[:10, columns].unstack()

        return np.square(np.subtract(x, y)).mean()

    # We want to exclude extreme draws that do not have anything to do with the observed sample. We
    # compute the RMSE of the implied choice probabilities and simply remove the 0.5% with the
    # highest statistic:
    df_obs = rp.get_example_model("kw_97_extended")[2]
    probs_obs = df_obs.groupby("Period")["Choice"].value_counts(normalize=True)
    probs_obs = probs_obs.unstack().fillna(0).sort_index(axis=1)

    columns = ["blue_collar", "military", "white_collar", "school", "home"]
    rmses = list()
    for count in df_quantities.index.get_level_values(0):

        probs_sim = pd.DataFrame(data=0.0, columns=columns, index=range(49))
        probs_sim.index.names = ["Period"]

        info = df_quantities["choice_shares_by_period"].loc[count]
        probs_sim.update(info)

        rmse = get_rmse(probs_sim, probs_obs)
        rmses.append(rmse)

    df_quantities_final = df_quantities.copy()

    df_rmse = pd.DataFrame(rmses, columns=["rmse"])
    cond = df_rmse["rmse"] < df_rmse["rmse"].quantile(0.995)
    idx = df_quantities_final[cond].index.get_level_values(0)
    df_quantities_final = df_quantities_final.loc[idx, :]
    df_quantities_final = df_quantities_final.sample(n=30000)

    # We need to align the samples of the quantities of interest and the parameters.
    idx = df_quantities_final.index.get_level_values(0)

    df_params_final = df_params.copy()
    df_params_final = df_params_final.loc[idx, :]
    df_params_final.reset_index(drop=True, inplace=True)
    df_params_final.to_pickle("df-params-final.pkl")

    df_quantities_final = df_quantities_final.copy()
    df_quantities_final = df_quantities_final.loc[idx, :]
    df_quantities_final.reset_index(drop=True, inplace=True)
    df_quantities_final.to_pickle("df-quantities-final.pkl")

    return df_quantities_final, df_params_final


def compute_results():
    # We need to access the asymptotic distribution of the free model parameters repeatedly and
    # this just cuts the provided foundational objects to only the relevant parts.
    mean_base = pd.read_pickle("mean_kw_97_extended.pkl")
    param_index = mean_base.index[~mean_base["se"].isnull()]
    mean = mean_base.loc[param_index, "value"].to_numpy()

    cov_base = pd.read_pickle("cov_kw_97_extended.pkl")
    cov = cov_base.loc[param_index, param_index].to_numpy()

    pkl.dump(mean, open("asymptotic-distribution-mean.pkl", "wb"))
    pkl.dump(cov, open("asymptotic-distribution-cov.pkl", "wb"))

    df_quantities = pd.read_pickle("df-quantities-final.pkl")
    df_params = pd.read_pickle("df-params-final.pkl")

    df_params["_weights"] = multivariate_normal(mean, cov, allow_singular=True).pdf(df_params)
    df_weights = df_params[["_weights"]].copy()
    df_params.drop("_weights", axis=1, inplace=True)

    for alpha in [0.10, 0.05, 0.01]:
        fname_ext = f"{alpha:0.3f}"

        df_params_subset = get_within_confidence_set(df_params, mean, cov, alpha)
        df_params_subset.to_pickle(f"df-params-{fname_ext}.pkl")

        idx = df_params_subset.index.get_level_values(0)
        df_quantities_subset = df_quantities.loc[idx, :]
        df_quantities_subset.to_pickle(f"df-quantities-{fname_ext}.pkl")

        df_weights_subset = df_weights.loc[idx, :]
        df_weights_subset.to_pickle(f"df-weights-{fname_ext}.pkl")

        # We need to restrict the data to the competing policies and not the other quantities of
        # interest such as the importance of initial type heterogeneity and an overall subsidy.
        df_quantities_subset = df_quantities_subset.filter(regex="subsidy_effect_on_years_for_*")

        args = (df_params_subset, df_quantities_subset, mean, cov, 0.0, df_weights_subset)
        df_uncertainty = get_decisions_uncertainty(*args)

        df_info = pd.read_pickle("qoi-at-mean.pickle")
        df_certainty = get_decisions_certainty(df_quantities_subset, df_info, mean)

        pd.concat([df_uncertainty, df_certainty]).to_pickle(f"df-decisions-{fname_ext}.pkl")


if __name__ == "__main__":

    if not os.path.exists("results"):
        src = "https://uni-bonn.sciebo.de/s/ORscTCWmWruvTu4/download"

        cmds = list()
        cmds += ["wget " + src + " --output-document results.zip"]
        cmds += ["unzip -o results.zip > /dev/null"]

        [sp.check_call(cmd, shell=True) for cmd in cmds]

        df_quantities = get_quantities_df()
        df_params = get_params_df()

    df_quantities = pd.read_pickle("df-quantities.pkl")
    df_params = pd.read_pickle("df-params.pkl")

    get_final_sample(df_quantities, df_params)

    compute_results()
