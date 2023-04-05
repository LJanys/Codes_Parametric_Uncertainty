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

from functools import partial
import shutil


from decorators import numpy_interface
from library import compute_qoi
from library import TAGS


if __name__ == "__main__":
    # import in if condition so we don't get errors if mpi is not installed
    from mpi4py import MPI
    import numpy as np
    import respy as rp

    comm, status = MPI.Comm.Get_parent(), MPI.Status()
    num_children, rank = comm.Get_size(), comm.Get_rank()

    subdir = f"subdir_child_{rank}"
    if os.path.exists(subdir):
        shutil.rmtree(subdir)
    os.mkdir(subdir)
    os.chdir(subdir)

    # We need some additional task-specific information.
    prob_info = comm.bcast(None)
    skip_existing = prob_info["skip_existing"]
    constraints = prob_info["constraints"]
    num_params = prob_info["num_params"]
    n_periods = prob_info["n_periods"]
    options = prob_info["options"]
    params = prob_info["params"]
    model = prob_info["model"]

    simulate_func = rp.get_simulate_func(params, options)
    qoi = numpy_interface(params, constraints)(compute_qoi)
    qoi = partial(qoi, simulate_func=simulate_func, skip_existing=skip_existing)

    while True:

        comm.send(None, dest=0)

        comm.recv(status=status)
        tag = status.Get_tag()

        if tag == TAGS.RUN:
            sample = np.empty(num_params, dtype="float64")
            comm.Recv([sample, MPI.DOUBLE])
            qoi(sample)

        elif tag == TAGS.EXIT:
            comm.send(None, dest=0)
            os.chdir("../")
            shutil.rmtree(subdir)
            comm.Disconnect()
            break

        else:
            raise AssertionError
