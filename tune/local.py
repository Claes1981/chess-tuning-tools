import logging
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime
from logging import Logger
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)
from prettytable import PrettyTable

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bask import Optimizer #, acquisition
#from bask.priors import make_roundflat
from matplotlib.transforms import Bbox
from numpy.random import RandomState
import random
from athena.active import ActiveSubspaces
from athena.utils import Normalizer
from scipy.optimize import OptimizeResult
from scipy.stats import dirichlet #, halfnorm
from skopt.space import Categorical, Dimension, Integer, Real, Space
from skopt.utils import normalize_dimensions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from tune.plots import (
    plot_objective,
    plot_objective_1d,
    plot_optima,
    plot_performance,
    plot_activesubspace_eigenvalues,
    plot_activesubspace_eigenvectors,
    plot_activesubspace_sufficient_summary,
)
from tune.summary import confidence_intervals
from tune.utils import TimeControl, confidence_to_mult, expected_ucb

__all__ = [
    "counts_to_penta",
    "initialize_optimizer",
    "run_match",
    "check_if_pause",
    "is_debug_log",
    "check_log_for_errors",
    "parse_experiment_result",
    "print_results",
    "plot_results",
    "inputs_uniform",
    "reduce_ranges",
    "update_model",
    "elo_to_prob",
    "prob_to_elo",
    "setup_logger",
]

LOGGER = "ChessTuner"

#ACQUISITION_FUNC = {
    #"ei": acquisition.ExpectedImprovement(),
    #"lcb": acquisition.LCB(),
    #"mean": acquisition.Expectation(),
    #"mes": acquisition.MaxValueSearch(),
    #"pvrs": acquisition.PVRS(),
    #"ts": acquisition.ThompsonSampling(),
    #"ttei": acquisition.TopTwoEI(),
    #"vr": acquisition.VarianceReduction(),
#}

def elo_to_prob(elo: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Convert an Elo score (logit space) to a probability.

    Parameters
    ----------
    elo : float
        A real-valued Elo score.
    k : float, optional (default=4.0)
        Scale of the logistic distribution.

    Returns
    -------
    float
        Win probability

    Raises
    ------
    ValueError
        if k <= 0

    """
    if k <= 0:
        raise ValueError("k must be positive")
    return np.atleast_1d(1 / (1 + np.power(10, -elo / k)))


def prob_to_elo(p: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Convert a win probability to an Elo score (logit space).

    Parameters
    ----------
    p : float
        The win probability of the player.
    k : float, optional (default=4.0)
        Scale of the logistic distribution.

    Returns
    -------
    float
        Elo score of the player

    Raises
    ------
    ValueError
        if k <= 0

    """
    if k <= 0:
        raise ValueError("k must be positive")
    return np.atleast_1d(k * np.log10(-p / (p - 1)))


def counts_to_penta(
    counts: np.ndarray,
    prior_counts: Optional[Iterable[float]] = None,
    n_dirichlet_samples: int = 1000000,
    score_scale: float = 4.0,
    random_state: Union[int, RandomState, None] = None,
    **kwargs,
) -> Tuple[float, float]:
    """Compute mean Elo score and variance of the pentanomial model for a count array.

    Parameters
    ----------
    counts : np.ndarray
        Array of counts for WW, WD, WL/DD, LD and LL
    prior_counts : np.ndarray or None, default=None
        Pseudo counts to use for WW, WD, WL/DD, LD and LL in the
        pentanomial model.
    n_dirichlet_samples : int, default = 1 000 000
        Number of samples to draw from the Dirichlet distribution in order to
        estimate the standard error of the score.
    score_scale : float, optional (default=4.0)
        Scale of the logistic distribution used to calculate the score. Has to be a
        positive real number
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    kwargs : dict
        Additional keyword arguments
    Returns
    -------
    tuple (float, float)
        Mean Elo score and corresponding variance
    """
    if prior_counts is None:
        prior_counts = np.array([0.14, 0.19, 0.34, 0.19, 0.14]) * 2.5
    else:
        prior_counts = np.array(prior_counts)
        if len(prior_counts) != 5:
            raise ValueError("Argument prior_counts should contain 5 elements.")
    dist = dirichlet(alpha=counts + prior_counts)
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    score = float(prob_to_elo(dist.mean().dot(scores), k=score_scale))
    error = prob_to_elo(
        dist.rvs(n_dirichlet_samples, random_state=random_state).dot(scores),
        k=score_scale,
    ).var()
    return score, error


def setup_logger(verbose: int = 0, logfile: str = "log.txt") -> Logger:
    """Setup logger with correct verbosity and file handler.

    Parameters
    ----------
    verbose : int
        Verbosity level. If verbose = 0, use INFO level, otherwise DEBUG.
    logfile : str
        Desired path to the logfile.

    Returns
    -------
    Logger
        Logger to be used for logging.
    """
    log_level = logging.DEBUG if verbose > 0 else logging.INFO
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    logger = logging.getLogger(LOGGER)
    logger.setLevel(log_level)
    logger.propagate = False

    file_logger = logging.FileHandler(logfile)
    file_logger.setFormatter(log_format)
    logger.addHandler(file_logger)
    console_logger = logging.StreamHandler(sys.stdout)
    console_logger.setFormatter(log_format)
    logger.addHandler(console_logger)
    return logger


def load_points_to_evaluate(
    space: Space, csv_file: Optional[TextIO] = None, rounds: int = 10
) -> List[Tuple[List, int]]:
    """Load extra points to evaluate from a csv file.

    Parameters
    ----------
    space : Space
        Optimization space containing the parameters to tune.
    csv_file : TextIO, optional (default=None)
        Comma-separated text file containing the points to evaluate. If an extra column
        is present, it will be used as the number of rounds to play.
    rounds : int, optional (default=10)
        Number of rounds to play for each point. Will be used if no extra column is
        present.

    Returns
    -------
    List[Tuple[List, int]]
        List of points to evaluate and number of rounds to play each point.

    Raises
    ------
    ValueError
        If the values in the csv file are out of the optimization bounds or if the
        number of columns is not equal to the number of parameters (+1 in the case the
        sample size is provided).
    """

    if csv_file is None:
        # No file given, do not load any points:
        return []

    # Open csv file using pandas:
    df = pd.read_csv(csv_file, header=None)

    # First check if number of columns is correct:
    n_dim = len(space.dimensions)
    if len(df.columns) not in (n_dim, n_dim + 1):
        raise ValueError(
            f"Number of columns in csv file ({len(df.columns)}) does not match number"
            f" of dimensions ({n_dim})."
        )

    # Check if the given points are within the lower and upper bounds of the space:
    for i, dim in enumerate(space.dimensions):
        if not (df.iloc[:, i].between(dim.low, dim.high).all()):
            raise ValueError(
                "Some points in the csv file are outside of the specified bounds."
                " Please check the csv file and your configuration."
            )

    # If there is an extra column, extract it as the number of rounds for each point:
    if len(df.columns) == n_dim + 1:
        rounds_column = df.iloc[:, n_dim].values
        df = df.iloc[:, :n_dim]
    else:
        rounds_column = np.full(len(df), rounds)

    # All points are within the bounds, add them to the list of points to evaluate:
    return [(x, r) for x, r in zip(df.values.tolist(), rounds_column)]


def reduce_ranges(
    X: Sequence[list], y: Sequence[float], noise: Sequence[float], space: Space
) -> Tuple[bool, List[list], List[float], List[float]]:
    """Return all data points consistent with the new restricted space.

    Parameters
    ----------
    X : Sequence of lists
        Contains n_points many lists, each representing one configuration.
    y : Sequence of floats
        Contains n_points many scores, one for each configuration.
    noise : Sequence of floats
        Contains n_points many variances, one for each score.
    space : skopt.space.Space
        Space object specifying the new optimization space.

    Returns
    -------
    Tuple (bool, list, list, list)
        Returns a boolean indicating if a reduction of the dataset was needed and the
        corresponding new X, y and noise lists.
    """
    X_new = []
    y_new = []
    noise_new = []
    reduction_needed = False
    for row, yval, nval in zip(X, y, noise):
        include_row = True
        for dim, value in zip(space.dimensions, row):
            if isinstance(dim, Integer) or isinstance(dim, Real):
                lb, ub = dim.bounds
                if value < lb or value > ub:
                    include_row = False
            elif isinstance(dim, Categorical):
                if value not in dim.bounds:
                    include_row = False
            else:
                raise ValueError(f"Parameter type {type(dim)} unknown.")
        if include_row:
            X_new.append(row)
            y_new.append(yval)
            noise_new.append(nval)
        else:
            reduction_needed = True
    return reduction_needed, X_new, y_new, noise_new


def initialize_data(
    parameter_ranges: Sequence[Union[Sequence, Dimension]],
    data_path: Optional[str] = None,
    intermediate_data_path: Optional[str] = None,
    resume: bool = True,
) -> Tuple[list, list, list, int, list, list, int, list, list]:
    """Initialize data structures needed for tuning. Either empty or resumed from disk.

    Parameters
    ----------
    parameter_ranges : Sequence of Dimension objects or tuples
        Parameter range specifications as expected by scikit-optimize.
    data_path : str or None, default=None
        Path to the file containing the data structures used for resuming.
        If None, no resuming will be performed.
    intermediate_data_path : str or None, default=None
        Path to the file containing the data structures used for resuming an unfinished experiment.
        If None, no resuming will be performed.
    resume : bool, default=True
        If True, fill the data structures with the the data from the given data_path.
        Otherwise return empty data structures.

    Returns
    -------
    tuple consisting of list, list, list and int
        Returns the initialized data structures X, y, noise and iteration number.

    Raises
    ------
    ValueError
        If the number of specified parameters is not matching the existing number of
        parameters in the data.
    """
    logger = logging.getLogger()
    X = []
    y = []
    noise = []
    optima = []
    performance = []
    point = []
    iteration = 0
    round = 0
    counts_array = np.array([0, 0, 0, 0, 0])
    if data_path is not None and resume:
        space = normalize_dimensions(parameter_ranges)
        path = pathlib.Path(data_path)
        intermediate_path = pathlib.Path(intermediate_data_path)
        if intermediate_path.exists():
            with np.load(intermediate_path) as importa:
                round = importa["arr_0"]
                counts_array = importa["arr_1"]
                point = importa["arr_2"].tolist()
                for i in range(len(point)):
                    if isinstance(point[i], float) and point[i].is_integer():
                        point[i] = int(point[i])
        if path.exists():
            with np.load(path) as importa:
                X = importa["arr_0"].tolist()
                y = importa["arr_1"].tolist()
                noise = importa["arr_2"].tolist()
                if "arr_3" in importa:
                    optima = importa["arr_3"].tolist()
                if "arr_4" in importa:
                    performance = importa["arr_4"].tolist()
                if "arr_5" in importa:
                    iteration = importa["arr_5"]
                else:
                    iteration = len(X)
            if len(X[0]) != space.n_dims:
                raise ValueError(
                    f"Number of parameters ({len(X[0])}) are not matching "
                    f"the number of dimensions ({space.n_dims})."
                )
            reduction_needed, X_reduced, y_reduced, noise_reduced = reduce_ranges(
                X, y, noise, space
            )
            if reduction_needed:
                backup_path = path.parent / (
                    path.stem + f"_backup_{int(time.time())}" + path.suffix
                )
                logger.warning(
                    f"The parameter ranges are smaller than the existing data. "
                    f"Some points will have to be discarded. "
                    f"The original {len(X)} data points will be saved to "
                    f"{backup_path}"
                )
                np.savez_compressed(
                    backup_path, np.array(X), np.array(y), np.array(noise)
                )
                X = X_reduced
                y = y_reduced
                noise = noise_reduced
            #iteration = len(X)
    return X, y, noise, iteration, optima, performance, round, counts_array, point


def setup_random_state(seed: int) -> np.random.RandomState:
    """Return a seeded RandomState object.

    Parameters
    ----------
    seed : int
        Random seed to be used to seed the RandomState.

    Returns
    -------
    numpy.random.RandomState
        RandomState to be used to generate random numbers.
    """
    ss = np.random.SeedSequence(seed)
    return np.random.RandomState(np.random.MT19937(ss.spawn(1)[0]))


def initialize_optimizer(
    X: Sequence[list],
    y: Sequence[float],
    noise: Sequence[float],
    parameter_ranges: Sequence[Union[Sequence, Dimension]],
    noise_scaling_coefficient: float = 1.0,
    random_seed: int = 0,
    warp_inputs: bool = True,
    normalize_y: bool = True,
    #kernel_lengthscale_prior_lower_bound: float = 0.1,
    #kernel_lengthscale_prior_upper_bound: float = 0.5,
    #kernel_lengthscale_prior_lower_steepness: float = 2.0,
    #kernel_lengthscale_prior_upper_steepness: float = 1.0,
    n_points: int = 500,
    n_initial_points: int = 16,
    acq_function: str = "mes",
    acq_function_samples: int = 1,
    acq_function_lcb_alpha: float = 1.96,
    resume: bool = True,
    fast_resume: bool = True,
    model_path: Optional[str] = None,
    gp_initial_burnin: int = 100,
    gp_initial_samples: int = 300,
    gp_priors: Optional[List[Callable[[float], float]]] = None,
) -> Optimizer:
    """Create an Optimizer object and if needed resume and/or reinitialize.

    Parameters
    ----------
    X : Sequence of lists
        Contains n_points many lists, each representing one configuration.
    y : Sequence of floats
        Contains n_points many scores, one for each configuration.
    noise : Sequence of floats
        Contains n_points many variances, one for each score.
    parameter_ranges : Sequence of Dimension objects or tuples
        Parameter range specifications as expected by scikit-optimize.
    random_seed : int, default=0
        Random seed for the optimizer.
    warp_inputs : bool, default=True
        If True, the optimizer will internally warp the input space for a better model
        fit. Can negatively impact running time and required burnin samples.
    n_points : int, default=500
        Number of points to evaluate the acquisition function on.
    n_initial_points : int, default=16
        Number of points to pick quasi-randomly to initialize the the model, before
        using the acquisition function.
    acq_function : str, default="mes"
        Acquisition function to use.
    acq_function_samples : int, default=1
        Number of hyperposterior samples to average the acquisition function over.
    resume : bool, default=True
        If True, resume optimization from existing data. If False, start with a
        completely fresh optimizer.
    fast_resume : bool, default=True
        If True, restore the optimizer from disk, avoiding costly reinitialization.
        If False, reinitialize the optimizer from the existing data.
    model_path : str or None, default=None
        Path to the file containing the existing optimizer to be used for fast resume
        functionality.
    gp_initial_burnin : int, default=100
        Number of burnin samples to use for reinitialization.
    gp_initial_samples : int, default=300
        Number of samples to use for reinitialization.
    gp_priors : list of callables, default=None
        List of priors to be used for the kernel hyperparameters. Specified in the
        following order:
        - signal magnitude prior
        - lengthscale prior (x number of parameters)
        - noise magnitude prior

    Returns
    -------
    bask.Optimizer
        Optimizer object to be used in the main tuning loop.
    """
    logger = logging.getLogger(LOGGER)
    # Create random generator:
    random_state = setup_random_state(random_seed)
    #space = normalize_dimensions(parameter_ranges)

    gp_kwargs = dict(
        normalize_y=normalize_y,
        warp_inputs=warp_inputs,
    )
    if acq_function == "rand":
        current_acq_func = random.choice(["ts", "lcb", "pvrs", "mes", "ei", "mean"])
    else:
        current_acq_func = acq_function

    if acq_function_lcb_alpha == float("inf"):
        acq_function_lcb_alpha = str(
            acq_function_lcb_alpha
        )  # Bayes-skopt expect alpha as a string, "inf", in case of infinite alpha.
    acq_func_kwargs = dict(
        alpha=acq_function_lcb_alpha,
        n_thompson=500,
    )

    #roundflat = make_roundflat(
                #kernel_lengthscale_prior_lower_bound,
                #kernel_lengthscale_prior_upper_bound,
                #kernel_lengthscale_prior_lower_steepness,
                #kernel_lengthscale_prior_upper_steepness,
            #)
    #priors = [
        # Prior distribution for the signal variance:
        #lambda x: halfnorm(scale=2.).logpdf(np.sqrt(np.exp(x))) + x / 2.0 - np.log(2.0),
        # Prior distribution for the length scales:
        #*[lambda x: roundflat(np.exp(x)) + x for _ in range(space.n_dims)],
        # Prior distribution for the noise:
        #lambda x: halfnorm(scale=2.).logpdf(np.sqrt(np.exp(x))) + x / 2.0 - np.log(2.0)
        #]

    opt = Optimizer(
        dimensions=parameter_ranges,
        n_points=n_points,
        n_initial_points=n_initial_points,
        # gp_kernel=kernel,  # TODO: Let user pass in different kernels
        gp_kwargs=gp_kwargs,
        #gp_priors=priors,
        gp_priors=gp_priors,
        acq_func=current_acq_func,
        acq_func_kwargs=acq_func_kwargs,
        random_state=random_state,
    )

    if not resume:
        return opt

    reinitialize = True
    if model_path is not None and fast_resume:
        path = pathlib.Path(model_path)
        if path.exists():
            with open(model_path, mode="rb") as model_file:
                old_opt = dill.load(model_file)
                logger.info(f"Resuming from existing optimizer in {model_path}.")
            if opt.space == old_opt.space:
                old_opt.acq_func = opt.acq_func
                old_opt.acq_func_kwargs = opt.acq_func_kwargs
                old_opt.n_points = opt.n_points
                opt = old_opt
                reinitialize = False
            else:
                logger.info(
                    "Parameter ranges have been changed and the "
                    "existing optimizer instance is no longer "
                    "valid. Reinitializing now."
                )
            if gp_priors is not None:
                opt.gp_priors = gp_priors

    if reinitialize and len(X) > 0:
        logger.info(
            f"Importing {len(X)} existing datapoints. " f"This could take a while..."
        )
        if acq_function == "rand":
            logger.debug(f"Current random acquisition function: {current_acq_func}")
        opt.tell(
            X,
            y,
            #noise_vector=noise,
            noise_vector=[i * noise_scaling_coefficient for i in noise],
            gp_burnin=gp_initial_burnin,
            gp_samples=gp_initial_samples,
            n_samples=acq_function_samples,
            progress=True,
        )
        logger.info("Importing finished.")
    #root_logger.debug(f"noise_vector: {[i*noise_scaling_coefficient for i in noise]}")
    logger.debug(f"GP kernel_: {opt.gp.kernel_}")
    #logger.debug(f"GP priors: {opt.gp_priors}")
    #logger.debug(f"GP X_train_: {opt.gp.X_train_}")
    #logger.debug(f"GP alpha: {opt.gp.alpha}")
    #logger.debug(f"GP alpha_: {opt.gp.alpha_}")
    #logger.debug(f"GP y_train_: {opt.gp.y_train_}")
    #logger.debug(f"GP y_train_std_: {opt.gp.y_train_std_}")
    #logger.debug(f"GP y_train_mean_: {opt.gp.y_train_mean_}")

    #if warp_inputs and hasattr(opt.gp, "warp_alphas_"):
        #warp_params = dict(
            #zip(
                #parameter_ranges.keys(),
                #zip(
                    #np.around(np.exp(opt.gp.warp_alphas_), 3),
                    #np.around(np.exp(opt.gp.warp_betas_), 3),
                #),
            #)
        #)
        #logger.debug(
            #f"Input warping was applied using the following parameters for "
            #f"the beta distributions:\n"
            #f"{warp_params}"
        #)

    return opt


def print_results(
    optimizer: Optimizer,
    result_object: OptimizeResult,
    parameter_names: Sequence[str],
    confidence: float = 0.9,
) -> Tuple[np.ndarray, float, float]:
    """Log the current results of the optimizer.

    Parameters
    ----------
    optimizer : bask.Optimizer
        Fitted Optimizer object.
    result_object : scipy.optimize.OptimizeResult
        Result object containing the data and the last fitted model.
    parameter_names : Sequence of str
        Names of the parameters to use for printing.
    confidence : float, default=0.9
        Confidence used for the confidence intervals.

    Raises
    ------
    ValueError
        If computation of the optimum was not successful due to numerical reasons.
    """
    logger = logging.getLogger(LOGGER)
    try:
        best_point, best_value = expected_ucb(result_object, alpha=0.0)
        best_point_dict = dict(zip(parameter_names, best_point))
        with optimizer.gp.noise_set_to_zero():
            _, best_std = optimizer.gp.predict(
                optimizer.space.transform([best_point]), return_std=True
            )
        logger.info(f"Current optimum:\n{best_point_dict}")
        estimated_elo = -best_value * 100
        logger.info(
            f"Estimated Elo: {np.around(estimated_elo, 4)} +- "
            f"{np.around(best_std * 100, 4).item()}"
        )
        confidence_mult = confidence_to_mult(confidence)
        lower_bound = np.around(
            -best_value * 100 - confidence_mult * best_std * 100, 4
        ).item()
        upper_bound = np.around(
            -best_value * 100 + confidence_mult * best_std * 100, 4
        ).item()
        logger.info(
            f"{confidence * 100}% confidence interval of the Elo value: "
            f"({lower_bound}, "
            f"{upper_bound})"
        )
        confidence_out = confidence_intervals(
            optimizer=optimizer,
            param_names=parameter_names,
            hdi_prob=confidence,
            opt_samples=1000,
            space_samples=5000,
            multimodal=True,
            only_mean=True,
        )
        logger.info(
            f"{confidence * 100}% confidence intervals of the parameters:"
            f"\n{confidence_out}"
        )
        return best_point, estimated_elo, float(best_std * 100)
    except ValueError as e:
        logger.info(
            "Computing current optimum was not successful. "
            "This can happen in rare cases and running the "
            "tuner again usually works."
        )
        raise e


def plot_results(
    optimizer: Optimizer,
    result_object: OptimizeResult,
    iterations: np.ndarray,
    elos: np.ndarray,
    optima: np.ndarray,
    plot_path: str,
    parameter_names: Sequence[str],
    confidence: float = 0.9,
    current_iteration: Optional[int] = None,
) -> None:
    """Plot the current results of the optimizer.

    Parameters
    ----------
    optimizer : bask.Optimizer
        Fitted Optimizer object.
    result_object : scipy.optimize.OptimizeResult
        Result object containing the data and the last fitted model.
    iterations : np.ndarray
        Array containing the iterations at which optima were collected.
    elos : np.ndarray, shape=(n_iterations, 2)
        Array containing the estimated Elo of the optima and the standard error.
    optima : np.ndarray
        Array containing the predicted optimal parameters.
    plot_path : str
        Path to the directory to which the plots should be saved.
    parameter_names : Sequence of str
        Names of the parameters to use for plotting.
    confidence : float
        The confidence level of the normal distribution to plot in the 1d plot.
    current_iteration : int, default=None
        The current iteration of the optimization process.
        If None, the current iteration is assumed to be the amount of points collected.
    """
    logger = logging.getLogger(LOGGER)

    plt.rcdefaults()
    logger.debug("Starting to compute the next partial dependence plot.")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    dark_gray = "#36393f"

    if current_iteration is None:
        current_iteration = len(optimizer.Xi)

    # First save the landscape:
    save_params = dict()
    if optimizer.space.n_dims == 1:
        fig, ax = plot_objective_1d(
            result=result_object,
            parameter_name=parameter_names[0],
            confidence=confidence,
        )
        save_params["bbox_inches"] = Bbox([[0.5, -0.2], [9.25, 5.5]])
    else:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(
            nrows=optimizer.space.n_dims,
            ncols=optimizer.space.n_dims,
            figsize=(3 * optimizer.space.n_dims, 3 * optimizer.space.n_dims),
        )
        for i in range(optimizer.space.n_dims):
            for j in range(optimizer.space.n_dims):
                ax[i, j].set_facecolor("xkcd:dark grey")
        fig.patch.set_facecolor("xkcd:dark grey")
        plot_objective(
            result_object,
            regression_object=None,
            polynomial_features_object=None,
            dimensions=parameter_names,
            next_point=optimizer._next_x,
            plot_standard_deviation=False,
            plot_polynomial_regression=False,
            fig=fig,
            ax=ax,
        )
    plotpath = pathlib.Path(plot_path)
    for subdir in ["landscapes", "elo", "optima"]:
        (plotpath / subdir).mkdir(parents=True, exist_ok=True)
    full_plotpath = (
        plotpath / f"landscapes/partial_dependence-{timestr}-{current_iteration}.png"
    )
    dpi = 150 if optimizer.space.n_dims == 1 else 300
    plt.savefig(
        full_plotpath,
        dpi=dpi,
        facecolor="xkcd:dark grey",
        **save_params,
    )
    logger.info(f"Saving a partial dependence plot to {full_plotpath}.")
    plt.close(fig)

    logger.debug(
        "Starting to compute the next polynomial regression partial dependence plot."
    )

    timestr = time.strftime("%Y%m%d-%H%M%S")

    polynomial_features = PolynomialFeatures(degree=2)
    samples_polynomial_features_transformed = polynomial_features.fit_transform(
        optimizer.space.transform(np.asarray(optimizer.Xi))
    )

    logger.debug(
        f"polynomial_features.n_output_features_= {polynomial_features.n_output_features_}"
    )

    LinearRegression_polynomial = LinearRegression()
    LinearRegression_polynomial.fit(
        samples_polynomial_features_transformed,
        np.asarray(optimizer.yi),
        1 / np.asarray(optimizer.noisei),
    )

    logger.debug(
        f"LinearRegression_polynomial.score= {LinearRegression_polynomial.score(samples_polynomial_features_transformed, np.asarray(optimizer.yi), 1 / np.asarray(optimizer.noisei))}"
    )

    # breakpoint()

    LinearRegression_polynomial_predicted_scores = LinearRegression_polynomial.predict(
        samples_polynomial_features_transformed
    )
    LinearRegression_polynomial_residuals = (
        np.asarray(optimizer.yi) - LinearRegression_polynomial_predicted_scores
    )
    LinearRegression_polynomial_weighted_residuals = (
        LinearRegression_polynomial_residuals
        * np.sqrt(1 / np.asarray(optimizer.noisei))
    )

    # Save the landscape:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(
        nrows=optimizer.space.n_dims,
        ncols=optimizer.space.n_dims,
        figsize=(3 * optimizer.space.n_dims, 3 * optimizer.space.n_dims),
    )
    for i in range(optimizer.space.n_dims):
        for j in range(optimizer.space.n_dims):
            ax[i, j].set_facecolor("xkcd:dark grey")
    fig.patch.set_facecolor("xkcd:dark grey")
    plot_objective(
        result_object,
        regression_object=LinearRegression_polynomial,
        polynomial_features_object=polynomial_features,
        dimensions=parameter_names,
        next_point=optimizer._next_x,
        plot_standard_deviation=False,
        plot_polynomial_regression=True,
        fig=fig,
        ax=ax,
    )
    plotpath = pathlib.Path(plot_path)
    for subdir in ["landscapes", "elo", "optima"]:
        (plotpath / subdir).mkdir(parents=True, exist_ok=True)
    full_plotpath = (
        plotpath
        / f"landscapes/partial_dependence_polynomial_regression-{timestr}-{current_iteration}.png"
    )
    dpi = 150 if optimizer.space.n_dims == 1 else 300
    plt.savefig(
        full_plotpath,
        dpi=dpi,
        facecolor="xkcd:dark grey",
        **save_params,
    )
    logger.info(
        f"Saving a polynomial regression partial dependence plot to {full_plotpath}."
    )
    plt.close(fig)

    # Plot the history of optima:
    fig, ax = plot_optima(
        iterations=iterations,
        optima=optima,
        space=optimizer.space,
        parameter_names=parameter_names,
    )
    full_plotpath = plotpath / f"optima/optima-{timestr}-{current_iteration}.png"
    fig.savefig(full_plotpath, dpi=150, facecolor="xkcd:dark grey")
    plt.close(fig)

    # Plot the predicted Elo performance of the optima:
    fig, ax = plot_performance(
        performance=np.hstack([iterations[:, None], elos]), confidence=confidence
    )
    full_plotpath = plotpath / f"elo/elo-{timestr}-{current_iteration}.png"
    fig.savefig(full_plotpath, dpi=150, facecolor="xkcd:dark grey")
    plt.close(fig)

    logger.debug("Starting to compute the next standard deviation plot.")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.style.use("dark_background")
    standard_deviation_figure, standard_deviation_axes = plt.subplots(
        nrows=optimizer.space.n_dims,
        ncols=optimizer.space.n_dims,
        figsize=(3 * optimizer.space.n_dims, 3 * optimizer.space.n_dims),
    )
    standard_deviation_figure.patch.set_facecolor("xkcd:dark grey")
    for i in range(optimizer.space.n_dims):
        for j in range(optimizer.space.n_dims):
            standard_deviation_axes[i, j].set_facecolor("xkcd:dark grey")

    plot_objective(
        result_object,
        regression_object=None,
        polynomial_features_object=None,
        dimensions=parameter_names,
        next_point=optimizer._next_x,
        plot_standard_deviation=True,
        plot_polynomial_regression=False,
        fig=standard_deviation_figure,
        ax=standard_deviation_axes,
    )
    standard_deviation_full_plotpath = (
        plotpath / f"landscapes/standard_deviation-{timestr}-{current_iteration}.png"
    )
    plt.savefig(
        standard_deviation_full_plotpath,
        dpi=300,
        facecolor="xkcd:dark grey",
        **save_params,
    )
    logger.info(
        f"Saving a standard deviation plot to {standard_deviation_full_plotpath}."
    )
    plt.close(standard_deviation_figure)
    plt.rcdefaults()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    number_of_random_active_subspace_samples = 10000 - len(result_object.x_iters)
    number_of_input_dimensions = optimizer.space.n_dims
    # active_subspace_samples_y_values = []
    # active_subspace_samples_gradients = []

    # Uniformly distributed inputs
    lb = 0 * np.ones(number_of_input_dimensions)  # lower bounds
    ub = 1 * np.ones(number_of_input_dimensions)  # upper bounds

    active_subspace_samples_x_raw = inputs_uniform(
        number_of_random_active_subspace_samples, lb, ub
    )
    active_subspace_samples_x_raw = np.append(
        active_subspace_samples_x_raw,
        optimizer.space.transform(np.asarray(result_object.x_iters)),
        axis=0,
    )

    active_subspaces_input_normalizer = Normalizer(lb, ub)
    active_subspace_samples_normalized_x = (
        active_subspaces_input_normalizer.fit_transform(active_subspace_samples_x_raw)
    )
    active_subspace_samples_y_values = np.zeros(np.shape(active_subspace_samples_x_raw)[0])
    active_subspace_samples_gradients = np.zeros(np.shape(active_subspace_samples_x_raw))

    if optimizer.gp.kernel.k1.k2.nu >= 1.5:
        for row_number, x_row in enumerate(active_subspace_samples_x_raw):
            y_row, grad_row = optimizer.gp.predict(
                np.reshape(x_row, (1, -1)), return_mean_grad=True
            )
            # if active_subspace_samples_gradients == []:
            #     active_subspace_samples_gradients = grad_row
            #     active_subspace_samples_y_values = y_row
            # else:
            #     active_subspace_samples_gradients = np.vstack(
            #         [active_subspace_samples_gradients, grad_row]
            #     )
            #     active_subspace_samples_y_values = np.vstack(
            #         [active_subspace_samples_y_values, y_row]
            #     )
            # active_subspace_samples_y_values.append(y_row)
            # active_subspace_samples_gradients.append(grad_row)
            active_subspace_samples_y_values[row_number]=y_row
            active_subspace_samples_gradients[row_number]=grad_row

        active_subspaces_object = ActiveSubspaces(dim=2, method="exact", n_boot=1000)
        active_subspaces_object.fit(gradients=active_subspace_samples_gradients)
    else:
        for x_row in active_subspace_samples_x_raw:
            y_row = optimizer.gp.predict(np.reshape(x_row, (1, -1)))
            if active_subspace_samples_y_values == []:
                active_subspace_samples_y_values = y_row
            else:
                active_subspace_samples_y_values = np.vstack(
                    [active_subspace_samples_y_values, y_row]
                )

        active_subspaces_object = ActiveSubspaces(dim=2, method="local", n_boot=1000)
        active_subspaces_object.fit(
            inputs=active_subspace_samples_normalized_x,
            outputs=active_subspace_samples_y_values,
        )

    plt.style.use("dark_background")

    active_subspace_figure = plt.figure(
        constrained_layout=True,
        figsize=(20, 18 + active_subspaces_object.evects.shape[1] * 6),
    )
    active_subspace_subfigures = active_subspace_figure.subfigures(
        nrows=3,
        ncols=1,
        wspace=0.07,
        height_ratios=[1, active_subspaces_object.evects.shape[1], 3],
    )

    #active_subspace_figure.tight_layout()
    #active_subspace_figure.patch.set_facecolor("xkcd:dark grey")
    active_subspace_eigenvalues_axes = active_subspace_subfigures[0].subplots(1, 1)
    active_subspace_eigenvalues_axes = plot_activesubspace_eigenvalues(
        active_subspaces_object,
        active_subspace_figure=active_subspace_figure,
        active_subspace_eigenvalues_axes=active_subspace_eigenvalues_axes,
        #figsize=(6, 4),
    )
    logger.debug(
        f"Active subspace eigenvalues: {np.squeeze(active_subspaces_object.evals)}"
    )

    active_subspace_eigenvectors_axes = active_subspace_subfigures[1].subplots(
        active_subspaces_object.evects.shape[1], 1
    )
    active_subspace_eigenvectors_axes = plot_activesubspace_eigenvectors(
        active_subspaces_object,
        active_subspace_figure=active_subspace_figure,
        active_subspace_eigenvectors_axes=active_subspace_eigenvectors_axes,
        #n_evects=number_of_input_dimensions,
        n_evects=active_subspaces_object.evects.shape[1],
        labels=parameter_names,
        #figsize=(6, 4),
    )

    activity_scores_table = PrettyTable()
    activity_scores_table.add_column("Parameter", parameter_names)
    activity_scores_table.add_column(
        "Activity score", np.squeeze(active_subspaces_object.activity_scores)
    )
    activity_scores_table.sortby = "Activity score"
    activity_scores_table.reversesort = True
    logger.debug(f"Active subspace activity scores:\n{activity_scores_table}")
    #logger.debug(f"Active subspace activity scores: {np.squeeze(active_subspaces_object.activity_scores)}")

    active_subspace_sufficient_summary_axes = active_subspace_subfigures[2].subplots(
        1, 1
    )
    active_subspace_sufficient_summary_axes = plot_activesubspace_sufficient_summary(
        active_subspaces_object,
        active_subspace_samples_normalized_x,
        active_subspace_samples_y_values,
        result_object,
        next_point=optimizer._next_x,
        active_subspace_figure=active_subspace_figure,
        active_subspace_sufficient_summary_axes=active_subspace_sufficient_summary_axes,
        #figsize=(6, 4),
    )

    active_subspace_figure.suptitle("Active subspace")

    #plt.show()
    active_subspace_full_plotpath = (
        plotpath / f"landscapes/active_subspace-{timestr}-{current_iteration}.png"
    )
    active_subspace_figure.savefig(
        active_subspace_full_plotpath,
        dpi=300,
        facecolor="xkcd:dark grey",
        **save_params,
    )
    logger.info(f"Saving an active subspace plot to {active_subspace_full_plotpath}.")
    plt.close(active_subspace_figure)
    plt.rcdefaults()


def inputs_uniform(n_samples, lb, ub):
    return np.vstack(
        np.array(
            [np.random.uniform(lb[i], ub[i], n_samples) for i in range(lb.shape[0])]
        ).T
    )


def run_match(
    cutechesscli_command: Optional[str] = "cutechess-cli",
    rounds: int = 1,
    engine1_tc: Optional[Union[str, TimeControl]] = None,
    engine2_tc: Optional[Union[str, TimeControl]] = None,
    engine1_st: Optional[Union[str, int]] = None,
    engine2_st: Optional[Union[str, int]] = None,
    engine1_npm: Optional[Union[str, int]] = None,
    engine2_npm: Optional[Union[str, int]] = None,
    engine1_depth: Optional[Union[str, int]] = None,
    engine2_depth: Optional[Union[str, int]] = None,
    engine1_ponder: bool = False,
    engine2_ponder: bool = False,
    engine1_restart: str = "on",
    engine2_restart: str = "on",
    timemargin: Optional[Union[str, int]] = None,
    opening_file: Optional[str] = None,
    tuning_config_name: str = None,
    adjudicate_draws: bool = False,
    draw_movenumber: int = 1,
    draw_movecount: int = 10,
    draw_score: int = 8,
    adjudicate_resign: bool = False,
    resign_movecount: int = 3,
    resign_score: int = 550,
    resign_twosided: bool = False,
    adjudicate_tb: bool = False,
    tb_path: Optional[str] = None,
    concurrency: int = 1,
    debug_mode: bool = False,
    **kwargs: Any,
) -> Iterator[str]:
    """Run a cutechess-cli match of two engines with paired random openings.

    Parameters
    ----------
    cutechesscli_command : str, default="cutechess-cli"
        Command (with or without path) to start the cutecess-cli executable.
    rounds : int, default=1
        Number of rounds to play in the match (each round consists of 2 games).
    engine1_tc : str or TimeControl object, default=None
        Time control to use for the first engine. If str, it can be a
        non-increment time control like "10" (10 seconds) or an increment
        time control like "5+1.5" (5 seconds total with 1.5 seconds increment).
        If None, it is assumed that engine1_npm, engine1_st or engine1_depth is
        provided.
    engine2_tc : str or TimeControl object, default=None
        See engine1_tc.
    engine1_st : str or int, default=None
        Time limit in seconds for each move.
        If None, it is assumed that engine1_tc, engine1_npm or engine1_depth is
        provided.
    engine2_st : str or TimeControl object, default=None
        See engine1_tc.
    engine1_npm : str or int, default=None
        Number of nodes per move the engine is allowed to search.
        If None, it is assumed that engine1_tc, engine1_st or engine1_depth is provided.
    engine2_npm : str or int, default=None
        See engine1_npm.
    engine1_depth : str or int, default=None
        Depth the engine is allowed to search.
        If None, it is assumed that engine1_tc, engine1_st or engine1_npm is provided.
    engine2_depth : str or int, default=None
        See engine1_depth.
    engine1_ponder : bool, default=False
        If True, allow engine1 to ponder.
    engine2_ponder : bool, default=False
        See engine1_ponder.
    engine1_restart : str, default="on"
        Restart mode for engine1. Can be "auto" (engine decides), "on" (default, engine
        is always restarted between games), "off" (engine is never restarted).
    engine2_restart : str, default="on"
        See engine1_restart.
    timemargin : str or int, default=None
        Allowed number of milliseconds the engines are allowed to go over the time
        limit. If None, the margin is 0.
    opening_file : str, default=None
        Path to the file containing the openings. Can be .epd or .pgn.
        Make sure that the file explicitly has the .epd or .pgn suffix, as it
        is used to detect the format.
    tuning_config_name : str, default=None
        Filename of the tuning configuration.
    adjudicate_draws : bool, default=False
        Specify, if cutechess-cli is allowed to adjudicate draws, if the
        scores of both engines drop below draw_score for draw_movecount number
        of moves. Only kicks in after draw_movenumber moves have been played.
    draw_movenumber : int, default=1
        Number of moves to play after the opening, before draw adjudication is
        allowed.
    draw_movecount : int, default=10
        Number of moves below the threshold draw_score, without captures and
        pawn moves, before the game is adjudicated as draw.
    draw_score : int, default=8
        Score threshold of the engines in centipawns. If the score of both
        engines drops below this value for draw_movecount consecutive moves,
        and there are no captures and pawn moves, the game is adjudicated as
        draw.
    adjudicate_resign : bool, default=False
        Specify, if cutechess-cli is allowed to adjudicate wins/losses based on
        the engine scores. If one engine’s score drops below -resign_score for
        resign_movecount many moves, the game is considered a loss for this
        engine.
    resign_movecount : int, default=3
        Number of consecutive moves one engine has to output a score below
        the resign_score threshold for the game to be considered a loss for this
        engine.
    resign_score : int, default=550
        Resign score threshold in centipawns. The score of the engine has to
        stay below -resign_score for at least resign_movecount moves for it to
        be adjudicated as a loss.
    resign_twosided : bool, default=False
        If True, the absolute score for both engines has to above resign_score before
        the game is adjudicated.
    adjudicate_tb : bool, default=False
        Allow cutechess-cli to adjudicate games based on Syzygy tablebases.
        If true, tb_path has to be set.
    tb_path : str, default=None
        Path to the folder containing the Syzygy tablebases.
    concurrency : int, default=1
        Number of games to run in parallel. Be careful when running time control
        games, since the engines can negatively impact each other when running
        in parallel.

    Yields
    -------
    out : str
        Results of the cutechess-cli match streamed as str.
    """
    string_array = ["/usr/bin/nice"]
    string_array.append("--5")
    string_array.append(cutechesscli_command)
    string_array.extend(("-concurrency", str(concurrency)))

    if (
        engine1_npm is None
        and engine1_tc is None
        and engine1_st is None
        and engine1_depth is None
    ) or (
        engine2_npm is None
        and engine2_tc is None
        and engine2_st is None
        and engine2_depth is None
    ):
        raise ValueError("A valid time control or nodes configuration is required.")
    string_array.extend(
        _construct_engine_conf(
            id=1,
            engine_npm=engine1_npm,
            engine_tc=engine1_tc,
            engine_st=engine1_st,
            engine_depth=engine1_depth,
            engine_ponder=engine1_ponder,
            engine_restart=engine1_restart,
            timemargin=timemargin,
        )
    )
    string_array.extend(
        _construct_engine_conf(
            id=2,
            engine_npm=engine2_npm,
            engine_tc=engine2_tc,
            engine_st=engine2_st,
            engine_depth=engine2_depth,
            engine_ponder=engine2_ponder,
            engine_restart=engine2_restart,
            timemargin=timemargin,
        )
    )

    if opening_file is not None:
        opening_path = pathlib.Path(opening_file)
        if not opening_path.exists():
            raise FileNotFoundError(
                f"Opening file the following path was not found: {opening_path}"
            )
        opening_format = opening_path.suffix
        if opening_format not in {".epd", ".pgn"}:
            raise ValueError(
                "Unable to determine opening format. "
                "Make sure to add .epd or .pgn to your filename."
            )
        string_array.extend(
            (
                "-openings",
                f"file={str(opening_path)}",
                f"format={opening_format[1:]}",
                "order=random",
            )
        )

    if adjudicate_draws:
        string_array.extend(
            (
                "-draw",
                f"movenumber={draw_movenumber}",
                f"movecount={draw_movecount}",
                f"score={draw_score}",
            )
        )
    if adjudicate_resign:
        string_array.extend(
            (
                "-resign",
                f"movecount={resign_movecount}",
                f"score={resign_score}",
                f"twosided={str(resign_twosided).lower()}",
            )
        )
    if adjudicate_tb:
        if tb_path is None:
            raise ValueError("No path to tablebases provided.")
        tb_path_object = pathlib.Path(tb_path)
        if not tb_path_object.exists():
            raise FileNotFoundError(
                f"No folder found at the following path: {str(tb_path_object)}"
            )
        string_array.extend(("-tb", str(tb_path_object)))

    string_array.extend(("-rounds", "1"))
    string_array.extend(("-games", "2"))
    string_array.append("-repeat")
    string_array.append("-recover")
    string_array.append("-debug")
    string_array.extend(
        ("-pgnout", f"{pathlib.Path(tuning_config_name).with_suffix('.pgn')}")
    )

    with subprocess.Popen(
        string_array, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as popen:
        if popen.stdout is not None:
            for line in iter(popen.stdout.readline, ""):
                yield line
        else:
            raise ValueError("No stdout found.")


def check_if_pause() -> None:
    # Read the start and end pause times from a file.
    # pause_times.txt contains intervals in the format 'HH:MM-HH:MM'
    # e g 05:00-10:00
    #     14:30-18:30
    # and is in the same directory as the program
    with open("pause_times.txt", "r") as file:
        intervals = file.read().strip().split("\n")

    for interval in intervals:
        start_time, end_time = interval.split("-")
        start_hour, start_minute = map(int, start_time.split(":"))
        end_hour, end_minute = map(int, end_time.split(":"))
        # Check if it is time to pause the program and
        # check if the interval spans over midnight
        if end_hour < start_hour or (
            end_hour == start_hour and end_minute < start_minute
        ):
            # Split the interval into two separate intervals, one for each day
            pause_between_times(start_hour, start_minute, 23, 59)
            pause_between_times(0, 0, end_hour, end_minute)
        else:
            pause_between_times(start_hour, start_minute, end_hour, end_minute)


def pause_between_times(start_hour, start_minute, end_hour, end_minute):
    current_time = time.localtime()
    current_hour, current_minute = current_time.tm_hour, current_time.tm_min

    if (
        current_hour >= start_hour
        and (current_hour != start_hour or current_minute >= start_minute)
    ) and (
        current_hour < end_hour
        or (current_hour == end_hour and current_minute < end_minute)
    ):
        target_time = time.struct_time(
            (
                current_time.tm_year,
                current_time.tm_mon,
                current_time.tm_mday,
                end_hour,
                end_minute,
                0,
                current_time.tm_wday,
                current_time.tm_yday,
                current_time.tm_isdst,
            )
        )
        target_timestamp = time.mktime(target_time)
        current_timestamp = time.mktime(current_time)
        sleep_time = target_timestamp - current_timestamp
        if sleep_time > 0:
            print(f"Pause until {target_time}.")
        time.sleep(sleep_time)


def is_debug_log(
    cutechess_line: str,
) -> bool:
    """Check if the provided cutechess log line is a debug mode line.

    Parameters
    ----------
    cutechess_line : str
        One line from a cutechess log.

    Returns
    -------
    bool
        True, if the given line is a debug mode line, False otherwise.
    """
    if re.match(r"[0-9]+ [<>]", cutechess_line) is not None:
        return True
    return False


def check_log_for_errors(
    cutechess_output: List[str],
) -> None:
    """Parse the log output produced by cutechess-cli and scan for important errors.

    Parameters
    ----------
    cutechess_output : list of str
        String containing the log output produced by cutechess-cli.
    """
    logger = logging.getLogger(LOGGER)
    for line in cutechess_output:
        # Check for forwarded errors:
        pattern = r"[0-9]+ [<>].+: error (.+)"
        match = re.search(pattern=pattern, string=line)
        if match is not None:
            logger.warning(f"cutechess-cli error: {match.group(1)}")

        # Check for unknown UCI option
        pattern = r"Unknown (?:option|command): (.+)"
        match = re.search(pattern=pattern, string=line)
        if match is not None:
            logger.error(
                f"UCI option {match.group(1)} was unknown to the engine. "
                f"Check if the spelling is correct."
            )
            continue

        # Check for loss on time
        pattern = (
            r"Finished game [0-9]+ \((.+) vs (.+)\): [0-9]-[0-9] {(\S+) "
            r"(?:loses on time)}"
        )
        match = re.search(pattern=pattern, string=line)
        if match is not None:
            engine = match.group(1) if match.group(3) == "White" else match.group(2)
            logger.warning(f"Engine {engine} lost on time as {match.group(3)}.")
            continue

        # Check for connection stall:
        pattern = (
            r"Finished game [0-9]+ \((.+) vs (.+)\): [0-9]-[0-9] {(\S+)'s "
            r"(?:connection stalls)}"
        )
        match = re.search(pattern=pattern, string=line)
        if match is not None:
            engine = match.group(1) if match.group(3) == "White" else match.group(2)
            logger.error(
                f"{engine}'s connection stalled as {match.group(3)}. "
                f"Game result is unreliable."
            )


def parse_experiment_result(
    outstr: str,
    prior_counts: Optional[Sequence[float]] = None,
    n_dirichlet_samples: int = 1000000,
    score_scale: float = 4.0,
    random_state: Union[int, RandomState, None] = None,
    **kwargs: Any,
) -> Tuple[float, float, float]:
    """Parse cutechess-cli result output to extract mean score and error.

    Here we use a simple pentanomial model to exploit paired openings.
    We distinguish the outcomes WW, WD, WL/DD, LD and LL and apply the
    following scoring (note, that the optimizer always minimizes the score):

    +------+------+-------+-----+-----+
    | WW   | WD   | WL/DD | LD  | LL  |
    +======+======+=======+=====+=====+
    | -1.0 | -0.5 | 0.0   | 0.5 | 1.0 |
    +------+------+-------+-----+-----+

    Note: It is important that the match output was produced using
    cutechess-cli using paired openings, otherwise the returned score is
    useless.

    Parameters
    ----------
    output : string (utf-8)
        Match output of cutechess-cli. It assumes the output was coming from
        a head-to-head match with paired openings.
    prior_counts : list-like float or int, default=None
        Pseudo counts to use for WW, WD, WL/DD, LD and LL in the
        pentanomial model.
    n_dirichlet_samples : int, default = 1 000 000
        Number of samples to draw from the Dirichlet distribution in order to
        estimate the standard error of the score.
    score_scale : float, optional (default=4.0)
        Scale of the logistic distribution used to calculate the score. Has to be a
        positive real number
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    Returns
    -------
    score : float (in [-1, 1])
        Expected (negative) score of the first player (the lower the stronger)
    error : float
        Estimated standard error of the score. Estimated by repeated draws
        from a Dirichlet distribution.
    draw_rate : float
        Estimated draw rate of the match.
    """
    wdl_strings = re.findall(r"Score of.*:\s*([0-9]+\s-\s[0-9]+\s-\s[0-9]+)", outstr)
    array = np.array(
        [np.array([int(y) for y in re.findall(r"[0-9]+", x)]) for x in wdl_strings]
    )
    diffs = np.diff(array, axis=0, prepend=np.array([[0, 0, 0]]))

    # Parse order of finished games to be able to compute the correct pentanomial scores
    finished = np.array(
        [int(x) - 1 for x in re.findall(r"Finished game ([0-9]+)", outstr)]
    )
    diffs = diffs[np.argsort(finished)]

    counts = {"WW": 0, "WD": 0, "WL/DD": 0, "LD": 0, "LL": 0}
    DD = 0  # Track DD separately to compute draw rate
    for i in range(0, len(diffs) - 1, 2):
        match = diffs[i] + diffs[i + 1]
        if match[0] == 2:
            counts["WW"] += 1
        elif match[0] == 1:
            if match[1] == 1:
                counts["WL/DD"] += 1
            else:
                counts["WD"] += 1
        elif match[1] == 1:
            counts["LD"] += 1
        elif match[2] == 2:
            counts["WL/DD"] += 1
            DD += 1
        else:
            counts["LL"] += 1
    counts_array = np.array(list(counts.values()))
    score, error = counts_to_penta(
        counts=counts_array,
        prior_counts=prior_counts,
        n_dirichlet_samples=n_dirichlet_samples,
        score_scale=score_scale,
        random_state=random_state,
        **kwargs,
    )
    draw_rate = (DD + 0.5 * counts["WD"] + 0.5 * counts["LD"] + 1.0) / (
        counts_array.sum() + 3.0
    )
    return score, error, draw_rate, counts_array


def update_model(
    optimizer: Optimizer,
    point: list,
    score: float,
    variance: float,
    noise_scaling_coefficient: float = 1.0,
    acq_function_samples: int = 1,
    acq_function_lcb_alpha: float = 1.96,
    gp_burnin: int = 5,
    gp_samples: int = 300,
    gp_initial_burnin: int = 100,
    gp_initial_samples: int = 300,
) -> None:
    """Update the optimizer model with the newest data.

    Parameters
    ----------
    optimizer : bask.Optimizer
        Optimizer object which is to be updated.
    point : list
        Latest configuration which was tested.
    score : float
        Elo score the configuration achieved.
    variance : float
        Variance of the Elo score of the configuration.
    acq_function_samples : int, default=1
        Number of hyperposterior samples to average the acquisition function over.
    gp_burnin : int, default=5
        Number of burnin iterations to use before keeping samples for the model.
    gp_samples : int, default=300
        Number of samples to collect for the model.
    gp_initial_burnin : int, default=100
        Number of burnin iterations to use for the first initial model fit.
    gp_initial_samples : int, default=300
        Number of samples to collect
    """
    logger = logging.getLogger(LOGGER)
    while True:
        try:
            now = datetime.now()
            # We fetch kwargs manually here to avoid collisions:
            n_samples = acq_function_samples
            gp_burnin = gp_burnin
            gp_samples = gp_samples
            if optimizer.gp.chain_ is None:
                gp_burnin = gp_initial_burnin
                gp_samples = gp_initial_samples
            optimizer.tell(
                x=point,
                y=score,
                #noise_vector=variance,
                noise_vector=noise_scaling_coefficient * variance,
                n_samples=n_samples,
                gp_samples=gp_samples,
                gp_burnin=gp_burnin,
            )
            later = datetime.now()
            difference = (later - now).total_seconds()
            logger.info(f"GP sampling finished ({difference}s)")
            #logger.debug(f"noise_vector: {[i*noise_scaling_coefficient for i in noise]}")
            logger.debug(f"GP kernel_: {optimizer.gp.kernel_}")
            #logger.debug(f"GP priors: {opt.gp_priors}")
            #logger.debug(f"GP X_train_: {opt.gp.X_train_}")
            #logger.debug(f"GP alpha: {opt.gp.alpha}")
            #logger.debug(f"GP alpha_: {opt.gp.alpha_}")
            #logger.debug(f"GP y_train_: {opt.gp.y_train_}")
            #logger.debug(f"GP y_train_std_: {opt.gp.y_train_std_}")
            #logger.debug(f"GP y_train_mean_: {opt.gp.y_train_mean_}")
        except ValueError:
            logger.warning(
                "Error encountered during fitting. Trying to sample chain a bit. "
                "If this problem persists, restart the tuner to reinitialize."
            )
            optimizer.gp.sample(n_burnin=11, priors=optimizer.gp_priors)
        else:
            break


def _construct_engine_conf(
    id: int,
    engine_npm: Optional[Union[int, str]] = None,
    engine_tc: Optional[Union[str, TimeControl]] = None,
    engine_st: Optional[Union[int, str]] = None,
    engine_depth: Optional[Union[int, str]] = None,
    engine_ponder: bool = False,
    engine_restart: str = "on",
    timemargin: Optional[Union[int, str]] = None,
) -> List[str]:
    result = ["-engine", f"conf=engine{id}", f"restart={engine_restart}"]
    if timemargin is not None:
        result.append(f"timemargin={timemargin}")
    if engine_ponder:
        result.append("ponder")
    if engine_npm is not None:
        result.extend(("tc=inf", f"nodes={engine_npm}"))
        return result
    elif engine_st is not None:
        result.append(f"st={str(engine_st)}")
        return result
    elif engine_depth is not None:
        result.extend(("tc=inf", f"depth={str(engine_depth)}"))
        return result
    elif engine_tc is not None:
        if isinstance(engine_tc, str):
            engine_tc = TimeControl.from_string(engine_tc)
        result.append(f"tc={str(engine_tc)}")
        return result
    else:
        raise ValueError(f"No engine time control specified for engine {id}.")
