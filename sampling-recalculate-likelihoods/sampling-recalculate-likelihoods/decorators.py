"""Decorators to call functions using external params with internal params.
"""
import functools

import pandas as pd
from estimagic.optimization.process_constraints import process_constraints
from estimagic.optimization.reparametrize import reparametrize_from_internal


def numpy_interface(params, constraints=None):
    """Convert x to params.

    This decorator receives a NumPy array of parameters and converts it to a
    :class:`pandas.DataFrame` which can be handled by the user's code. If the
    input is already a :class:`pandas.DataFrame` with the correct structure,
    then the input will just be passed through.

    Args:
        params (pandas.DataFrame): See :ref:`params`.
        constraints (list of dict): Contains constraints.

    """
    if "_internal_free" not in params.columns:
        constraints, params = process_constraints(constraints, params)

    def decorator_numpy_interface(func):
        @functools.wraps(func)
        def wrapper_numpy_interface(x, *args, **kwargs):
            # Handle usage in :func:`internal_function` for gradients.
            if constraints is None:
                p = params.copy()
                p["value"] = x

            # Handle case of "external" params input
            if isinstance(x, pd.DataFrame) and "value" in x.columns:
                p = x

            # Handle usage in :func:`internal_criterion`.
            else:
                p = reparametrize_from_internal(
                    internal=x,
                    fixed_values=params["_internal_fixed_value"].to_numpy(),
                    pre_replacements=params["_pre_replacements"].to_numpy().astype(int),
                    processed_constraints=constraints,
                    post_replacements=(
                        params["_post_replacements"].to_numpy().astype(int)
                    ),
                    processed_params=params,
                )

            criterion_value = func(p, *args, **kwargs)

            return criterion_value

        return wrapper_numpy_interface

    return decorator_numpy_interface
