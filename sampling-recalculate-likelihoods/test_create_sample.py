import os

"""Test that mpi-distributed and serial execution give the same result.

For this to run, pytest hast to be invoked as follows:

``mpirun -n 1 python -m pytest --with-mpi``


"""

import pytest
from numpy.testing import assert_array_almost_equal as aaae

from create_sample import create_sample
import warnings


@pytest.mark.skipif(
    "CI" in os.environ.keys(), reason="no multiprocessing on GitHub Actions"
)
def test_result_stays_same_with_different_numbers_of_processes():
    processes = [1, 2, 4]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="PY_SSIZE_T_CLEAN will be required for '#' formats"
        )

        results = [
            create_sample(start_index=0, stop_index=4, n_proc=n_proc, n_periods=12)
            for n_proc in processes
        ]

    serial_res = results[0]
    for parallel_res in results[1:]:
        aaae(serial_res, parallel_res)