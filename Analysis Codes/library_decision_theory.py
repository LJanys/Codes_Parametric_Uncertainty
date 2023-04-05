from scipy.stats import chi2
import pandas as pd
import numpy as np


def get_within_confidence_set(df_params, mean, cov, alpha):
    """This functions returns all elements within the confidence set."""
    num_sample, num_params = df_params.shape

    df_params_subset = df_params.copy()
    cov_inv = np.linalg.inv(cov)

    args = (mean, cov_inv)
    df_params_subset["_stat"] = df_params_subset.apply(_compute_stat, axis=1, args=args)

    crit_val = chi2(num_params).ppf(1.0 - alpha)
    idx_inside = df_params_subset["_stat"] <= crit_val
    df_params_subset = df_params_subset.loc[idx_inside, :]

    df_params_subset.drop("_stat", axis=1, inplace=True)

    return df_params_subset


def get_decisions_certainty(df_policy, df_info, mean):
    df_as_if = _create_frame(df_policy, "as_if")

    info = df_info["subsidy_effect_on_years_by_type"]["difference"]
    for rank, policy in enumerate(info.sort_values(ascending=False).index):
        value = info[policy]
        index = ("as_if", f"subsidy_effect_on_years_for_type_{policy}")
        df_as_if.loc[index, :] = np.array([rank, value, mean], dtype="object")

    return df_as_if.sort_values("Rank")


def get_decisions_uncertainty(df_params, df_policy, mean, cov, alpha, df_weights=None):

    # Impose uniform weighting if nothing else specified.
    if df_weights is None:
        df_weights = pd.DataFrame(1.0, index=df_params.index, columns=["_weights"])
        df_weights /= df_weights.sum()

    # Restrict attention to parameter draws inside the confidence set
    if alpha > 0:
        df_params_subset = get_within_confidence_set(df_params, mean, cov, alpha)
    else:
        df_params_subset = df_params.copy()

    # Restrict attention to policy effects inside the confidence set
    idx = df_params_subset.index.get_level_values(0)
    df_policy_subset = df_policy.loc[idx, :]

    # Rescale weights to sum to one inside support.
    df_weights_subset = df_weights.loc[idx, :]
    df_weights_subset /= df_weights_subset.sum()

    # Determine decision based on competing criteria
    args = (df_policy_subset, df_weights_subset)
    df_subjective_bayes = _subjective_bayes_decision(*args)

    args = (df_policy_subset, df_params)
    df_minimax_regret = _minimax_regret_decision(*args)
    df_maximin = _maximin_decision(*args)

    decision = pd.concat([df_subjective_bayes, df_minimax_regret, df_maximin])

    return decision


def _maximin_decision(df_policy_subset, df_params):
    df_decision = _create_frame(df_policy_subset, "maximin")
    df_policy_internal = df_policy_subset.copy()

    info = df_policy_internal.min()
    for rank, policy in enumerate(info.sort_values(ascending=False).index):
        value = info[policy]
        index = df_policy_internal.index[df_policy_internal[policy] == value]
        params = df_params.loc[index, :].values[0]

        df_decision.loc[("maximin", policy), :] = np.array([rank, value, params], dtype="object")

    return df_decision.sort_values("Rank")


def _minimax_regret_decision(df_policy_subset, df_params):
    df_decision = _create_frame(df_policy_subset, "minimax_regret")
    df_policy_internal = df_policy_subset.copy()

    for label in df_policy_internal:
        regret = df_policy_internal.max(axis=1) - df_policy_internal[label]
        df_policy_internal[f"_regret_{label}"] = regret
    df_policy_regret = df_policy_internal.filter(like="_regret")

    info = df_policy_regret.max()
    for rank, policy in enumerate(info.sort_values(ascending=True).index):
        value = info[policy]
        index = df_policy_regret.index[df_policy_regret[policy] == value]
        params = df_params.loc[index, :].values[0]

        policy = policy.replace("_regret_", "")
        df_decision.loc[("minimax_regret", policy), :] = np.array(
            [rank, value, params], dtype="object"
        )

    return df_decision.sort_values("Rank")


def _subjective_bayes_decision(df_policy_subset, df_weights_subset):
    df_decision = _create_frame(df_policy_subset, "subjective_bayes")
    df_policy_internal = df_policy_subset.copy()

    for column in df_policy_internal.columns:
        df_policy_internal[column] *= df_weights_subset["_weights"]

    info = df_policy_internal.sum()
    for rank, policy in enumerate(info.sort_values(ascending=False).index):
        value = info[policy]
        df_decision.loc[("subjective_bayes", policy), :] = np.array(
            [rank, value, None], dtype="object"
        )

    return df_decision.sort_values("Rank")


def _compute_stat(row, mean, cov_inv):
    stat = (mean - row).T @ cov_inv @ (mean - row)
    return stat


def _create_frame(df_policy, criterion):
    names = ["Criterion", "Policy"]
    index = pd.MultiIndex.from_product([[criterion], df_policy.columns], names=names)
    df_decision = pd.DataFrame(columns=["Rank", "Value", "Params"], index=index)
    return df_decision
