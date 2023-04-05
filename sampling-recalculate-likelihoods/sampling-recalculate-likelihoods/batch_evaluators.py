from joblib import Parallel, delayed
import functools



def joblib_batch_evaluator(
    func,
    arguments,
    *,
    n_cores,
    unpack_symbol,
):
    """Batch evaluator based on joblib's Parallel.
    Args:
        func (Callable): The function that is evaluated.
        arguments (Iterable): Arguments for the functions. Their interperation
            depends on the unpack argument.
        n_cores (int): Number of cores used to evaluate the function in parallel.
            Value below one are interpreted as one. If only one core is used, the
            batch evaluator disables everything that could cause problems, i.e. in that
            case func and arguments are never pickled and func is executed in the main
            process.
        error_handling (str): Can take the values "raise" (raise the error and stop all
            tasks as soon as one task fails) and "continue" (catch exceptions and set
            the output of failed tasks to the traceback of the raised exception.
            KeyboardInterrupt and SystemExit are always raised.
        unpack_symbol (str or None). Can be "**", "*" or None. If None, func just takes
            one argument. If "*", the elements of arguments are positional arguments for
            func. If "**", the elements of arguments are keyword arguments for func.
    Returns:
        list: The function evaluations.
    """
    _check_inputs(func, arguments, n_cores, unpack_symbol)
    n_cores = int(n_cores) if int(n_cores) >= 2 else 1

    @unpack(symbol=unpack_symbol)
    def internal_func(*args, **kwargs):
        return func(*args, **kwargs)

    if n_cores == 1:
        res = [internal_func(arg) for arg in arguments]
    else:
        res = Parallel(n_jobs=n_cores)(delayed(internal_func)(arg) for arg in arguments)

    return res



def _check_inputs(func, arguments, n_cores, unpack_symbol):
    if not callable(func):
        raise TypeError("func must be callable.")

    try:
        arguments = list(arguments)
    except Exception as e:
        raise ValueError("arguments must be list like.") from e

    try:
        int(n_cores)
    except Exception:
        ValueError("n_cores must be an integer.")

    if unpack_symbol not in (None, "*", "**"):
        raise ValueError(
            f"unpack_symbol must be None, '*' or '**', not {unpack_symbol}"
        )


def unpack(func=None, symbol=None):
    def decorator_unpack(func):
        if symbol is None:

            @functools.wraps(func)
            def wrapper_unpack(arg):
                return func(arg)

        elif symbol == "*":

            @functools.wraps(func)
            def wrapper_unpack(arg):
                return func(*arg)

        elif symbol == "**":

            @functools.wraps(func)
            def wrapper_unpack(arg):
                return func(**arg)

        return wrapper_unpack

    if callable(func):
        return decorator_unpack(func)
    else:
        return decorator_unpack
