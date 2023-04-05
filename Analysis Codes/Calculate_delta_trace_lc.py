###This creates the

import pickle as pkl
import pandas as pd
import numpy as np
import respy as rp




num_periods, num_points = 50, 10

delta_se = np.sqrt(pkl.load(open("asymptotic-distribution-cov.pkl", "rb"))[0, 0])

params_base, options, df = rp.get_example_model("kw_97_extended_respy")
options["n_periods"] = num_periods

log_like = rp.get_log_like_func(params=params_base, options=options, df=df)
log_l_base = log_like(params_base)

simulate_func = rp.get_simulate_func(params_base, options)
delta = params_base.loc[("delta", "delta"), "value"]

lower, upper = delta - delta_se * 1.96, delta + delta_se * 1.96
df_rslt = pd.DataFrame(columns=["Impact", "Delta","logl","ratio_ll_base"], index=range(num_points))

delta_grid = np.linspace(lower, upper, num=num_points)


for i, delta in enumerate(delta_grid):
        params = params_base.copy()

        params.loc[("delta", "delta"), "value"] = delta
        df_base = simulate_func(params)

        params.loc[("nonpec_school", "hs_graduate"), "value"] += 2000
        df_policy = simulate_func(params)
        log_like = rp.get_log_like_func(params=params, options=options, df=df)
        log_l = log_like(params)
       
        
        stat_policy = df_policy.groupby("Identifier")["Experience_School"].last().mean()
        stat_base = df_base.groupby("Identifier")["Experience_School"].last().mean()

        df_rslt.loc[i, :] = [stat_policy - stat_base, delta, log_l,log_l/log_l_base]

df_rslt.to_pickle("df-delta-trace.pkl")





