import respy as rp

params, options, data_obs = rp.get_example_model("kw_97_extended_respy")

# How does a simulated sample at our estimates look like.
options["simulation_agents"] = 10000
data_sim = rp.get_simulate_func(params, options)(params)
data_sim.to_pickle("df-simulated-sample.pkl")
