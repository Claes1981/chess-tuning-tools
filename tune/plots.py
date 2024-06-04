import itertools
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors #as matplotlibcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator
from scipy.optimize import OptimizeResult
from skopt.plots import _format_scatter_plot_axes
from skopt.space import Space
#from athena.utils import Normalizer

from tune.utils import confidence_to_mult, expected_ucb, latest_iterations

__all__ = [
    "partial_dependence",
    "plot_objective",
    "plot_objective_1d",
    "plot_optima",
    "plot_performance",
    "plot_activesubspace_eigenvalues",
    "plot_activesubspace_eigenvectors",
    "plot_activesubspace_sufficient_summary",
]


def _evenly_sample(dim, n_points):
    """Return `n_points` evenly spaced points from a Dimension.
    Parameters
    ----------
    dim : `Dimension`
        The Dimension to sample from.  Can be categorical; evenly-spaced
        category indices are chosen in order without replacement (result
        may be smaller than `n_points`).
    n_points : int
        The number of points to sample from `dim`.
    Returns
    -------
    xi : np.array
        The sampled points in the Dimension.  For Categorical
        dimensions, returns the index of the value in
        `dim.categories`.
    xi_transformed : np.array
        The transformed values of `xi`, for feeding to a model.
    """
    cats = np.array(getattr(dim, "categories", []), dtype=object)
    if len(cats):  # Sample categoricals while maintaining order
        xi = np.linspace(0, len(cats) - 1, min(len(cats), n_points), dtype=int)
        xi_transformed = dim.transform(cats[xi])
    else:
        bounds = dim.bounds
        # XXX use linspace(*bounds, n_points) after python2 support ends
        if dim.prior == "log-uniform":
            xi = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), n_points)
        else:
            xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = dim.transform(xi)
    return xi, xi_transformed


def partial_dependence(
    space,
    model,
    regression_object,
    polynomial_features_object,
    i,
    j=None,
    plot_standard_deviation=False,
    plot_polynomial_regression=False,
    sample_points=None,
    n_samples=250,
    n_points=40,
    x_eval=None,
):
    """Calculate the partial dependence for dimensions `i` and `j` with
    respect to the objective value, as approximated by `model`.
    The partial dependence plot shows how the value of the dimensions
    `i` and `j` influence the `model` predictions after "averaging out"
    the influence of all other dimensions.
    When `x_eval` is not `None`, the given values are used instead of
    random samples. In this case, `n_samples` will be ignored.
    Parameters
    ----------
    space : `Space`
        The parameter space over which the minimization was performed.
    model
        Surrogate model for the objective function.
    i : int
        The first dimension for which to calculate the partial dependence.
    j : int, default=None
        The second dimension for which to calculate the partial dependence.
        To calculate the 1D partial dependence on `i` alone set `j=None`.
    sample_points : np.array, shape=(n_points, n_dims), default=None
        Only used when `x_eval=None`, i.e in case partial dependence should
        be calculated.
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points` when using partial
        dependence.
    n_samples : int, default=100
        Number of random samples to use for averaging the model function
        at each of the `n_points` when using partial dependence. Only used
        when `sample_points=None` and `x_eval=None`.
    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.
    x_eval : list, default=None
        `x_eval` is a list of parameter values or None. In case `x_eval`
        is not None, the parsed dependence will be calculated using these
        values.
        Otherwise, random selected samples will be used.
    Returns
    -------
    For 1D partial dependence:
    xi : np.array
        The points at which the partial dependence was evaluated.
    yi : np.array
        The value of the model at each point `xi`.
    For 2D partial dependence:
    xi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.
    yi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.
    zi : np.array, shape=(n_points, n_points)
        The value of the model at each point `(xi, yi)`.
    For Categorical variables, the `xi` (and `yi` for 2D) returned are
    the indices of the variable in `Dimension.categories`.
    """
    # The idea is to step through one dimension, evaluating the model with
    # that dimension fixed and averaging either over random values or over
    # the given ones in x_val in all other dimensions.
    # (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (model) dimensions.

    # If we haven't parsed an x_eval list we use random sampled values instead
    if x_eval is None and sample_points is None:
        sample_points = space.transform(space.rvs(n_samples=n_samples))
    elif sample_points is None:
        sample_points = space.transform([x_eval])

    # dim_locs[i] is the (column index of the) start of dim i in
    # sample_points.
    # This is usefull when we are using one hot encoding, i.e using
    # categorical values
    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    if j is None:
        # We sample evenly instead of randomly. This is necessary when using
        # categorical values
        xi, xi_transformed = _evenly_sample(space.dimensions[i], n_points)
        yi_partial_dependence = []
        yi_standard_deviation = []
        for x_ in xi_transformed:
            rvs_ = np.array(sample_points)  # copy
            # We replace the values in the dimension that we want to keep
            # fixed
            rvs_[:, dim_locs[i] : dim_locs[i + 1]] = x_
            # In case of `x_eval=None` rvs conists of random samples.
            # Calculating the mean of these samples is how partial dependence
            # is implemented.
            if plot_standard_deviation:
                with model.noise_set_to_zero():
                    y, std = model.predict(rvs_, return_std=True)
                yi_partial_dependence.append(np.mean(y))
                yi_standard_deviation.append(np.mean(std))
            elif plot_polynomial_regression:
                yi_partial_dependence.append(
                    np.mean(
                        regression_object.predict(
                            polynomial_features_object.transform(rvs_)
                        )
                    )
                )
            else:
                yi_partial_dependence.append(np.mean(model.predict(rvs_)))

        return xi, yi_partial_dependence, yi_standard_deviation

    else:
        xi, xi_transformed = _evenly_sample(space.dimensions[j], n_points)
        yi, yi_transformed = _evenly_sample(space.dimensions[i], n_points)

        zi_partial_dependence = []
        zi_standard_deviation = []
        for x_ in xi_transformed:
            row_partial_dependence = []
            row_standard_deviation = []
            for y_ in yi_transformed:
                rvs_ = np.array(sample_points)  # copy
                rvs_[:, dim_locs[j] : dim_locs[j + 1]] = x_
                rvs_[:, dim_locs[i] : dim_locs[i + 1]] = y_
                if plot_standard_deviation:
                    with model.noise_set_to_zero():
                        z, std = model.predict(rvs_, return_std=True)
                    row_partial_dependence.append(np.mean(z))
                    row_standard_deviation.append(np.mean(std))
                elif plot_polynomial_regression:
                    row_partial_dependence.append(
                        np.mean(
                            regression_object.predict(
                                polynomial_features_object.transform(rvs_)
                            )
                        )
                    )
                else:
                    row_partial_dependence.append(np.mean(model.predict(rvs_)))
            zi_partial_dependence.append(row_partial_dependence)
            zi_standard_deviation.append(row_standard_deviation)

        return xi, yi, np.array(zi_partial_dependence).T, np.array(zi_standard_deviation).T


def plot_objective_1d(
    result: OptimizeResult,
    parameter_name: Optional[str] = None,
    n_points: int = 500,
    n_random_restarts: int = 100,
    confidence: float = 0.9,
    figsize: Tuple[float, float] = (10, 6),
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    colors: Optional[Sequence[Union[tuple, str]]] = None,
) -> Tuple[Figure, Axes]:
    """Plot the 1D objective function.

    Parameters
    ----------
    result : OptimizeResult
        The current optimization result.
    parameter_name : str, optional
        The name of the parameter to plot. If None, no x-axis label is shown.
    n_points : int (default=500)
        The number of points to use for prediction of the Gaussian process.
    n_random_restarts : int (default=100)
        The number of random restarts to employ to find the optima.
    confidence : float (default=0.9)
        The confidence interval to plot around the mean prediction.
    figsize : tuple (default=(10, 6))
        The size of the figure.
    fig : Figure, optional
        The figure to use. If None, a new figure is created.
    ax : Axes, optional
        The axes to use. If None, new axes are created.
    colors : Sequence of colors, optional
        The colors to use for different elements in the plot.
        Can be tuples or strings.

    Returns
    -------
    fig : Figure
        The figure.
    ax : Axes
        The axes.

    """
    if colors is None:
        colors = plt.cm.get_cmap("Set3").colors

    if fig is None:
        plt.style.use("dark_background")
        gs_kw = dict(width_ratios=(1,), height_ratios=[5, 1], hspace=0.05)
        fig, ax = plt.subplots(figsize=figsize, nrows=2, gridspec_kw=gs_kw, sharex=True)
        for a in ax:
            a.set_facecolor("xkcd:dark grey")
        fig.patch.set_facecolor("xkcd:dark grey")
    gp = result.models[-1]

    # Compute the optima of the objective function:
    failures = 0
    while True:
        try:
            with gp.noise_set_to_zero():
                min_x = expected_ucb(
                    result, alpha=0.0, n_random_starts=n_random_restarts
                )[0]
                min_ucb = expected_ucb(result, n_random_starts=n_random_restarts)[0]
        except ValueError:
            failures += 1
            if failures == 10:
                break
            continue
        else:
            break

    # Regardless of the range of the parameter to be plotted, the model always operates
    # in [0, 1]:
    x_gp = np.linspace(0, 1, num=n_points)
    x_orig = np.array(result.space.inverse_transform(x_gp[:, None])).flatten()
    with gp.noise_set_to_zero():
        y, y_err = gp.predict(x_gp[:, None], return_std=True)
    y = -y * 100
    y_err = y_err * 100
    confidence_mult = confidence_to_mult(confidence)

    (mean_plot,) = ax[0].plot(x_orig, y, zorder=4, color=colors[0])
    err_plot = ax[0].fill_between(
        x_orig,
        y - y_err * confidence_mult,
        y + y_err * confidence_mult,
        alpha=0.3,
        zorder=0,
        color=colors[0],
    )
    opt_plot = ax[0].axvline(x=min_x, zorder=3, color=colors[3])
    pess_plot = ax[0].axvline(x=min_ucb, zorder=2, color=colors[5])
    if parameter_name is not None:
        ax[1].set_xlabel(parameter_name)
    dim = result.space.dimensions[0]
    ax[0].set_xlim(dim.low, dim.high)
    match_plot = ax[1].scatter(
        x=result.x_iters,
        y=-result.func_vals * 100,
        zorder=1,
        marker=".",
        s=0.6,
        color=colors[0],
    )
    ax[0].set_ylabel("Elo")
    ax[1].set_ylabel("Elo")
    fig.legend(
        (mean_plot, err_plot, opt_plot, pess_plot, match_plot),
        ("Mean", f"{confidence:.0%} CI", "Optimum", "Conservative Optimum", "Matches"),
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
    )

    return fig, ax


def plot_objective(
    result,
    regression_object,
    polynomial_features_object,
    levels=20,
    n_points=200,
    n_samples=30,
    size=3,
    zscale="linear",
    dimensions=None,
    next_point=None,
    plot_standard_deviation=False,
    plot_polynomial_regression=False,
    n_random_restarts=100,
    alpha=0.25,
    margin=0.65,
    colors=None,
    partial_dependence_figure=None,
    partial_dependence_axes=None,
    standard_deviation_figure=None,
    standard_deviation_axes=None,
):
    """Pairwise partial dependence plot of the objective function.
    The diagonal shows the partial dependence for dimension `i` with
    respect to the objective function. The off-diagonal shows the
    partial dependence for dimensions `i` and `j` with
    respect to the objective function. The objective function is
    approximated by `result.model.`
    Pairwise scatter plots of the points at which the objective
    function was directly evaluated are shown on the off-diagonal.
    A red point indicates the found minimum.
    Note: search spaces that contain `Categorical` dimensions are
          currently not supported by this function.
    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.
    * `levels` [int, default=10]
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`.
    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension.
    * `n_samples` [int, default=250]
        Number of random samples to use for averaging the model function
        at each of the `n_points`.
    * `size` [float, default=2]
        Height (in inches) of each facet.
    * `zscale` [str, default='linear']
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.
    * `dimensions` [list of str, default=None] Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.
    * `n_random_restarts` [int, default=100]
        Number of restarts to try to find the global optimum.
    * `alpha` [float, default=0.25]
        Transparency of the sampled points.
    * `margin` [float, default=0.65]
        Margin in inches around the plot.
    * `colors` [list of tuples, default=None]
        Colors to use for the optima.
    * `partial_dependence_figure` [Matplotlib figure, default=None]
        Figure to use for plotting the partial dependence. If None, it will create one.
    * `partial_dependence_axes` [k x k axes, default=None]
        Axes on which to plot the marginals. If None, it will create appropriate
        axes.
    Returns
    -------
    * `partial_dependence_axes`: [`Axes`]:
        The matplotlib axes.
    """
    if colors is None:
        colors = plt.cm.get_cmap("Set3").colors
    space = result.space
    contour_plot = np.empty((space.n_dims, space.n_dims), dtype=object)
    samples = np.asarray(result.x_iters)
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))
    z_min = np.full((space.n_dims, space.n_dims), np.inf)
    z_max = np.full((space.n_dims, space.n_dims), np.NINF)
    z_ranges = np.zeros((space.n_dims, space.n_dims))

    if zscale == "log":
        locator = LogLocator()
    elif zscale == "linear":
        locator = None
    else:
        raise ValueError(
            "Valid values for zscale are 'linear' and 'log'," " not '%s'." % zscale
        )
    if partial_dependence_figure is None:
        partial_dependence_figure, partial_dependence_axes = plt.subplots(
            space.n_dims,
            space.n_dims,
            figsize=(size * space.n_dims, size * space.n_dims),
        )
    if plot_standard_deviation and standard_deviation_figure is None:
        standard_deviation_figure, standard_deviation_axes = plt.subplots(
            space.n_dims,
            space.n_dims,
            figsize=(size * space.n_dims, size * space.n_dims),
        )
    width, height = partial_dependence_figure.get_size_inches()

    partial_dependence_figure.subplots_adjust(
        left=margin / width,
        right=1 - margin / width,
        bottom=margin / height,
        top=1 - margin / height,
        hspace=0.1,
        wspace=0.1,
    )
    standard_deviation_figure.subplots_adjust(
        left=margin / width,
        right=1 - margin / width,
        bottom=margin / height,
        top=1 - margin / height,
        hspace=0.1,
        wspace=0.1,
    )
    failures = 0
    while True:
        try:
            with result.models[-1].noise_set_to_zero():
                min_x = expected_ucb(
                    result, alpha=0.0, n_random_starts=n_random_restarts
                )[0]
                min_ucb = expected_ucb(result, n_random_starts=n_random_restarts)[0]
        except ValueError:
            failures += 1
            if failures == 10:
                break
            continue
        else:
            break

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                xi, yi_partial_dependence, yi_standard_deviation = partial_dependence(
                    space,
                    result.models[-1],
                    regression_object,
                    polynomial_features_object,
                    i,
                    j=None,
                    plot_standard_deviation=plot_standard_deviation,
                    plot_polynomial_regression=plot_polynomial_regression,
                    sample_points=rvs_transformed,
                    n_points=n_points,
                )
                yi_min_partial_dependence, yi_max_partial_dependence = np.min(yi_partial_dependence), np.max(yi_partial_dependence)
                yi_min_standard_deviation, yi_max_standard_deviation = np.min(yi_standard_deviation), np.max(yi_standard_deviation)
                partial_dependence_axes[i, i].plot(xi, yi_partial_dependence, color=colors[1])
                standard_deviation_axes[i, i].plot(xi, yi_standard_deviation, color=colors[1])
                if failures != 10:
                    partial_dependence_axes[i, i].axvline(min_ucb[i], linestyle="--", color=colors[5], lw=1)
                    standard_deviation_axes[i, i].axvline(min_ucb[i], linestyle="--", color=colors[5], lw=1)
                    partial_dependence_axes[i, i].axvline(min_x[i], linestyle="--", color=colors[3], lw=1)
                    standard_deviation_axes[i, i].axvline(min_x[i], linestyle="--", color=colors[3], lw=1)
                    partial_dependence_axes[i, i].text(
                        min_ucb[i],
                        yi_min_partial_dependence + 0.7 * (yi_max_partial_dependence - yi_min_partial_dependence),
                        f"{np.around(min_ucb[i], 4)}",
                        color=colors[5],
                    )
                    standard_deviation_axes[i, i].text(
                        min_ucb[i],
                        yi_min_standard_deviation + 0.7 * (yi_max_standard_deviation - yi_min_standard_deviation),
                        f"{np.around(min_ucb[i], 4)}",
                        color=colors[5],
                    )
                    partial_dependence_axes[i, i].text(
                        min_x[i],
                        yi_min_partial_dependence + 0.9 * (yi_max_partial_dependence - yi_min_partial_dependence),
                        f"{np.around(min_x[i], 4)}",
                        color=colors[3],
                    )
                    standard_deviation_axes[i, i].text(
                        min_x[i],
                        yi_min_standard_deviation + 0.9 * (yi_max_standard_deviation - yi_min_standard_deviation),
                        f"{np.around(min_x[i], 4)}",
                        color=colors[3],
                    )

            # lower triangle
            elif i > j:
                xi, yi, zi_partial_dependence, zi_standard_deviation = partial_dependence(
                    space,
                    result.models[-1],
                    regression_object,
                    polynomial_features_object,
                    i,
                    j,
                    plot_standard_deviation,
                    plot_polynomial_regression,
                    rvs_transformed,
                    n_points,
                )
                contour_plot_partial_dependence[i, j] = partial_dependence_axes[i, j].contourf(
                    xi, yi, zi_partial_dependence, levels, locator=locator, cmap="viridis_r"
                )
                contour_plot_standard_deviation[i, j] = standard_deviation_axes[i, j].contourf(
                    xi, yi, zi_standard_deviation, levels, locator=locator, cmap="viridis_r"
                )
                #partial_dependence_figure.colorbar(contour_plot_partial_dependence[i, j], ax=partial_dependence_axes[i, j])
                partial_dependence_axes[i, j].scatter(
                    samples[:, j], samples[:, i], c="k", s=10, lw=0.0, alpha=alpha
                )
                standard_deviation_axes[i, j].scatter(
                    samples[:, j], samples[:, i], c="k", s=10, lw=0.0, alpha=alpha
                )
                if failures != 10:
                    partial_dependence_axes[i, j].scatter(
                        next_point[j], next_point[i], c=["xkcd:pink"], s=20, lw=0.0
                    )
                    standard_deviation_axes[i, j].scatter(
                        next_point[j], next_point[i], c=["xkcd:pink"], s=20, lw=0.0
                    )
                    partial_dependence_axes[i, j].scatter(
                        min_ucb[j], min_ucb[i], c=["xkcd:orange"], s=20, lw=0.0
                    )
                    standard_deviation_axes[i, j].scatter(
                        min_ucb[j], min_ucb[i], c=["xkcd:orange"], s=20, lw=0.0
                    )
                    partial_dependence_axes[i, j].scatter(min_x[j], min_x[i], c=["r"], s=20, lw=0.0)
                    standard_deviation_axes[i, j].scatter(min_x[j], min_x[i], c=["r"], s=20, lw=0.0)
                z_min_partial_dependence[i, j] = np.min(zi_partial_dependence)
                z_max_partial_dependence[i, j] = np.max(zi_partial_dependence)
                z_ranges_partial_dependence[i, j] = np.max(zi_partial_dependence) - np.min(zi_partial_dependence)
                partial_dependence_axes[i, j].text(
                    0.5,
                    0.5,
                    np.format_float_positional(
                        z_ranges_partial_dependence[i, j],
                        precision=2,
                        unique=False,
                        fractional=False,
                        trim="k",
                    ),
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=partial_dependence_axes[i, j].transAxes,
                )
                z_min_standard_deviation[i, j] = np.min(zi_standard_deviation)
                z_max__standard_deviation[i, j] = np.max(zi_standard_deviation)
                z_ranges_standard_deviation[i, j] = np.max(zi_standard_deviation) - np.min(zi_standard_deviation)
                standard_deviation_axes[i, j].text(
                    0.5,
                    0.5,
                    np.format_float_positional(
                        z_ranges_standard_deviation[i, j],
                        precision=2,
                        unique=False,
                        fractional=False,
                        trim="k",
                    ),
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=standard_deviation_axes[i, j].transAxes,
                )
    # Get all dimensions.
    plot_dims = []
    for row in range(space.n_dims):
        if space.dimensions[row].is_constant:
            continue
        plot_dims.append((row, space.dimensions[row]))
    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i > j:
                contour_plot_partial_dependence[i, j].set_clim(vmin=np.min(z_min_partial_dependence), vmax=np.max(z_max_partial_dependence))
                contour_plot_standard_deviation[i, j].set_clim(vmin=np.min(z_min_standard_deviation), vmax=np.max(z_max_standard_deviation))
    partial_dependence_figure.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=np.min(z_min_partial_dependence), vmax=np.max(z_max_partial_dependence)),
            cmap="viridis_r",
        ),
        ax=partial_dependence_axes[np.triu_indices(space.n_dims, k=1)],
        shrink=0.7,
    )
    standard_deviation_figure.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=np.min(z_min_standard_deviation), vmax=np.max(z_max_standard_deviation)),
            cmap="viridis_r",
        ),
        ax=standard_deviation_axes[np.triu_indices(space.n_dims, k=1)],
        shrink=0.7,
    )
    #plt.cm.ScalarMappable.set_clim(self, vmin=np.min(z_min_partial_dependence), vmax=np.max(z_max_partial_dependence))
    #partial_dependence_figure.colorbar(contour_plot_partial_dependence[1, 0], ax=partial_dependence_axes[np.triu_indices(space.n_dims, k=1)])
    # if plot_standard_deviation:
    #     return _format_scatter_plot_axes(
    #         ax,
    #         space,
    #         ylabel="Standard deviation",
    #         plot_dims=plot_dims,
    #         dim_labels=dimensions,
    #     )
    # else:
    #     return _format_scatter_plot_axes(
    #         ax,
    #         space,
    #         ylabel="Partial dependence",
    #         plot_dims=plot_dims,
    #         dim_labels=dimensions,
    #     )
    return _format_scatter_plot_axes(
            partial_dependence_axes,
            space,
            ylabel="Partial dependence",
            plot_dims=plot_dims,
            dim_labels=dimensions,
        ),
        _format_scatter_plot_axes(
            standard_deviation_axes,
            space,
            ylabel="Standard deviation",
            plot_dims=plot_dims,
            dim_labels=dimensions,
        )

def plot_optima(
    iterations: np.ndarray,
    optima: np.ndarray,
    space: Optional[Space] = None,
    parameter_names: Optional[Sequence[str]] = None,
    plot_width: float = 8,
    aspect_ratio: float = 0.4,
    fig: Optional[Figure] = None,
    ax: Optional[Union[Axes, np.ndarray]] = None,
    colors: Optional[Sequence[Union[tuple, str]]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Plot the optima found by the tuning algorithm.

    Parameters
    ----------
    iterations : np.ndarray
        The iterations at which the optima were found.
    optima : np.ndarray
        The optima found recorded at the given iterations.
    space : Space, optional
        The optimization space for the parameters. If provided, it will be used to
        scale the y-axes and to apply log-scaling, if the parameter is optimized on
        a log-scale.
    parameter_names : Sequence[str], optional
        The names of the parameters. If not provided, no y-axis labels will be shown.
    plot_width : int, optional
        The width of each plot in inches. The total width of the plot will be larger
        depending on the number of parameters and how they are arranged.
    aspect_ratio : float, optional
        The aspect ratio of the subplots. The default is 0.4, which means that the
        height of each subplot will be 40% of the width.
    fig : Figure, optional
        The figure to plot on. If not provided, a new figure in the style of
        chess-tuning-tools will be created.
    ax : np.ndarray or Axes, optional
        The axes to plot on. If not provided, new axes will be created.
        If provided, the axes will be filled. Thus, the number of axes should be at
        least as large as the number of parameters.
    colors : Sequence[Union[tuple, str]], optional
        The colors to use for the plots. If not provided, the color scheme 'Set3' of
        matplotlib will be used.

    Returns
    -------
    Figure
        The figure containing the plots.
    np.ndarray
        A two-dimensional array containing the axes.

    Raises
    ------
    ValueError
        - if the number of parameters does not match the number of parameter names
        - if the number of axes is smaller than the number of parameters
        - if the number of iterations is not matching the number of optima
        - if a fig, but no ax is passed
    """
    if optima.shape[0] != len(iterations):
        raise ValueError("Iteration array does not match optima array.")
    iterations, optima = latest_iterations(iterations, optima)
    n_points, n_parameters = optima.shape
    if parameter_names is not None and len(parameter_names) != n_parameters:
        raise ValueError(
            "Number of parameter names does not match the number of parameters."
        )
    if colors is None:
        colors = plt.cm.get_cmap("Set3").colors
    n_colors = len(colors)
    if fig is None:
        plt.style.use("dark_background")
        n_cols = int(np.floor(np.sqrt(n_parameters)))
        n_rows = int(np.ceil(n_parameters / n_cols))
        figsize = (n_cols * plot_width, aspect_ratio * plot_width * n_rows)
        fig, ax = plt.subplots(
            figsize=figsize,
            nrows=n_rows,
            ncols=n_cols,
            sharex=True,
        )

        margin_left = 1.0
        margin_right = 0.1
        margin_bottom = 0.5
        margin_top = 0.4
        wspace = 1
        hspace = 0.3
        plt.subplots_adjust(
            left=margin_left / figsize[0],
            right=1 - margin_right / figsize[0],
            bottom=margin_bottom / figsize[1],
            top=1 - margin_top / figsize[1],
            hspace=n_rows * hspace / figsize[1],
            wspace=n_cols * wspace / figsize[0],
        )
        ax = np.atleast_2d(ax).reshape(n_rows, n_cols)
        for a in ax.reshape(-1):
            a.set_facecolor("xkcd:dark grey")
            a.grid(which="major", color="#ffffff", alpha=0.1)
        fig.patch.set_facecolor("xkcd:dark grey")
        fig.suptitle(
            "Predicted best parameters over time",
            y=1 - 0.5 * margin_top / figsize[1],
            va="center",
        )
    else:
        if ax is None:
            raise ValueError("Axes must be specified if a figure is provided.")
        if not hasattr(ax, "__len__"):
            n_rows = n_cols = 1
        elif ax.ndim == 1:
            n_rows = len(ax)
            n_cols = 1
        else:
            n_rows, n_cols = ax.shape
        if n_rows * n_cols < n_parameters:
            raise ValueError("Not enough axes to plot all parameters.")
        ax = np.atleast_2d(ax).reshape(n_rows, n_cols)

    for i, (j, k) in enumerate(itertools.product(range(n_rows), range(n_cols))):
        a = ax[j, k]
        if i >= n_parameters:
            fig.delaxes(a)
            continue
        # If the axis is the last one in the current column, then set the xlabel:
        if (j + 1) * n_cols + k + 1 > n_parameters:
            a.set_xlabel("Iteration")
            # Since hspace=0, we have to fix the xaxis label and tick labels here:
            a.xaxis.set_label_coords(0.5, -0.12)
            a.xaxis.set_tick_params(labelbottom=True)

        a.plot(
            iterations,
            optima[:, i],
            color=colors[i % n_colors],
            zorder=10,
            linewidth=1.3,
        )
        a.axhline(
            y=optima[-1, i],
            color=colors[i % n_colors],
            zorder=9,
            linewidth=0.5,
            ls="--",
            alpha=0.6,
        )
        # If the user supplied an optimization space, we can use that information to
        # scale the yaxis and apply log-scaling, where necessary:
        s = f"{optima[-1, i]:.2f}"
        if space is not None:
            dim = space.dimensions[i]
            a.set_ylim(*dim.bounds)
            if dim.prior == "log-uniform":
                a.set_yscale("log")
                s = np.format_float_scientific(optima[-1, i], precision=2)

        # Label the horizontal line of the current optimal value:
        # First convert the y-value to normalized axes coordinates:
        point = a.get_xlim()[0], optima[-1, i]
        transformed_point = a.transAxes.inverted().transform(
            a.transData.transform(point)
        )
        a.text(
            x=transformed_point[0] + 0.01,
            y=transformed_point[1] - 0.02,
            s=s,
            bbox=dict(
                facecolor="xkcd:dark grey",
                edgecolor="None",
                alpha=0.5,
                boxstyle="square,pad=0.1",
            ),
            transform=a.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            color=colors[i % n_colors],
            zorder=11,
        )

        if parameter_names is not None:
            a.set_ylabel(parameter_names[i])
    return fig, ax


def plot_performance(
    performance: np.ndarray,
    confidence: float = 0.9,
    plot_width: float = 8,
    aspect_ratio: float = 0.7,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    colors: Optional[Sequence[Union[tuple, str]]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Plot the estimated Elo of the Optima predicted by the tuning algorithm.

    Parameters
    ----------
    performance : np.ndarray, shape=(n_iterations, 3)
        Array containing the iteration numbers, the estimated Elo of the predicted
        optimum, and the estimated standard error of the estimated Elo.
    confidence : float, optional (default=0.9)
        The confidence interval to plot around the estimated Elo.
    plot_width : int, optional (default=8)
        The width of each plot in inches. The total width of the plot will be larger
        depending on the number of parameters and how they are arranged.
    aspect_ratio : float, optional (default=0.7)
        The aspect ratio of the subplots. The default is 0.4, which means that the
        height of each subplot will be 40% of the width.
    fig : Figure, optional
        The figure to plot on. If not provided, a new figure in the style of
        chess-tuning-tools will be created.
    ax : np.ndarray or Axes, optional
        The axes to plot on. If not provided, new axes will be created.
        If provided, the axes will be filled. Thus, the number of axes should be at
        least as large as the number of parameters.
    colors : Sequence[Union[tuple, str]], optional
        The colors to use for the plots. If not provided, the color scheme 'Set3' of
        matplotlib will be used.

    Returns
    -------
    Figure
        The figure containing the plots.
    np.ndarray
        A two-dimensional array containing the axes.

    Raises
    ------
    ValueError
        - if the number of parameters does not match the number of parameter names
        - if the number of axes is smaller than the number of parameters
        - if the number of iterations is not matching the number of optima
        - if a fig, but no ax is passed
    """
    iterations, elo, elo_std = latest_iterations(*performance.T)
    if colors is None:
        colors = plt.cm.get_cmap("Set3").colors
    if fig is None:
        plt.style.use("dark_background")
        figsize = (plot_width, aspect_ratio * plot_width)
        fig, ax = plt.subplots(figsize=figsize)

        margin_left = 0.8
        margin_right = 0.1
        margin_bottom = 0.7
        margin_top = 0.3
        plt.subplots_adjust(
            left=margin_left / figsize[0],
            right=1 - margin_right / figsize[0],
            bottom=margin_bottom / figsize[1],
            top=1 - margin_top / figsize[1],
        )
        ax.set_facecolor("xkcd:dark grey")
        ax.grid(which="major", color="#ffffff", alpha=0.1)
        fig.patch.set_facecolor("xkcd:dark grey")
        ax.set_title("Elo of the predicted best parameters over time")
    elif ax is None:
        raise ValueError("Axes must be specified if a figure is provided.")

    ax.plot(
        iterations,
        elo,
        color=colors[0],
        zorder=10,
        linewidth=1.3,
        label="Predicted Elo",
    )
    confidence_mult = confidence_to_mult(confidence)
    ax.fill_between(
        iterations,
        elo - confidence_mult * elo_std,
        elo + confidence_mult * elo_std,
        color=colors[0],
        linewidth=0,
        zorder=9,
        alpha=0.25,
        label=f"{confidence:.0%} confidence interval",
    )
    ax.axhline(
        y=elo[-1],
        linestyle="--",
        zorder=8,
        color=colors[0],
        label="Last prediction",
        linewidth=1,
        alpha=0.3,
    )
    ax.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.08), ncol=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Elo")
    ax.set_xlim(min(iterations), max(iterations))

    return fig, ax


def plot_activesubspace_eigenvalues(
    active_subspaces_object,
    active_subspace_figure=None,
    active_subspace_eigenvalues_axes=None,
    n_evals=None,
    filename=None,
    figsize=(8, 8),
    title="",
    **kwargs,
):
    """
    Plot the eigenvalues.

    :param int n_evals: number of eigenvalues to plot. If not provided all
        the eigenvalues will be plotted.
    :param str filename: if specified, the plot is saved at `filename`.
    :param tuple(int,int) figsize: tuple in inches defining the figure size.
        Default is (8, 8).
    :param str title: title of the plot.
    :raises: TypeError

    .. warning:: `active_subspaces_object.fit` has to be called in advance.
    """
    #ax = self #or plt.gca()
    if active_subspaces_object.evals is None:
        raise TypeError(
            "The eigenvalues have not been computed."
            "You have to perform the fit method."
        )
    if n_evals is None:
        n_evals = active_subspaces_object.evals.shape[0]
    if n_evals > active_subspaces_object.evals.shape[0]:
        raise TypeError("Invalid number of eigenvalues to plot.")

    #ax = ax or plt.gca()
    #eigen_values_fig = plt.figure(figsize=figsize)
    #eigen_values_fig.suptitle(title)
    #ax = eigen_values_fig.add_subplot(111)
    if np.amin(active_subspaces_object.evals[:n_evals]) == 0:
        active_subspace_eigenvalues_axes.semilogy(
            range(1, n_evals + 1),
            active_subspaces_object.evals[:n_evals] + np.finfo(float).eps,
            "ko-",
            markersize=8,
            linewidth=2,
        )
    else:
        #active_subspace_eigenvalues_axes.semilogy(range(1, n_evals + 1),
        active_subspace_eigenvalues_axes.plot(
            range(1, n_evals + 1),
            active_subspaces_object.evals[:n_evals],
            "ko-",
            markersize=8,
            linewidth=2,
        )
        active_subspace_eigenvalues_axes.set_yscale("log")

    active_subspace_eigenvalues_axes.set_xticks(range(1, n_evals + 1))
    active_subspace_eigenvalues_axes.set_xlabel("Index")
    active_subspace_eigenvalues_axes.set_ylabel("Eigenvalues")

    if active_subspaces_object.evals_br is None:
        active_subspace_eigenvalues_axes.axis(
            [
                0,
                n_evals + 1,
                0.1 * np.amin(active_subspaces_object.evals[:n_evals]),
                10 * np.amax(active_subspaces_object.evals[:n_evals]),
            ]
        )
    else:
        if np.amin(active_subspaces_object.evals[:n_evals]) == 0:
            active_subspace_eigenvalues_axes.fill_between(
                range(1, n_evals + 1),
                active_subspaces_object.evals_br[:n_evals, 0]
                * (1 + np.finfo(float).eps),
                active_subspaces_object.evals_br[:n_evals, 1]
                * (1 + np.finfo(float).eps),
                facecolor="0.7",
                interpolate=True,
            )
        else:
            active_subspace_eigenvalues_axes.fill_between(
                range(1, n_evals + 1),
                active_subspaces_object.evals_br[:n_evals, 0],
                active_subspaces_object.evals_br[:n_evals, 1],
                facecolor="0.7",
                interpolate=True,
            )
        active_subspace_eigenvalues_axes.axis(
            [
                0,
                n_evals + 1,
                0.1 * np.amin(active_subspaces_object.evals_br[:n_evals, 0]),
                10 * np.amax(active_subspaces_object.evals_br[:n_evals, 1]),
            ]
        )

    active_subspace_eigenvalues_axes.grid(linestyle="dotted")
    active_subspace_eigenvalues_axes.set_facecolor("xkcd:dark grey")
    #eigen_values_fig.tight_layout

    #if filename:
    #    eigen_values_fig.savefig(filename)
    #else:
    #    return eigen_values_fig
    return active_subspace_eigenvalues_axes


def plot_activesubspace_eigenvectors(
    #self,
    active_subspaces_object,
    active_subspace_figure=None,
    active_subspace_eigenvectors_axes=None,
    n_evects=None,
    filename=None,
    figsize=None,
    labels=None,
    title="",
):
    """
    Plot the eigenvectors.

    :param int n_evects: number of eigenvectors to plot.
        Default is active_subspaces_object.dim.
    :param str filename: if specified, the plot is saved at `filename`.
    :param tuple(int,int) figsize: tuple in inches defining the figure size.
        Default is (8, 2 * n_evects).
    :param str labels: labels for the components of the eigenvectors.
    :param str title: title of the plot.
    :raises: ValueError, TypeError

    .. warning:: `active_subspaces_object.fit` has to be called in advance.
    """
    #ax = self or plt.gca()
    if active_subspaces_object.evects is None:
        raise TypeError(
            "The eigenvectors have not been computed."
            "You have to perform the fit method."
        )
    # print(f"active_subspaces_object.evects={active_subspaces_object.evects}")
    if n_evects is None:
        n_evects = active_subspaces_object.dim
    if n_evects > active_subspaces_object.evects.shape[0]:
        raise ValueError("Invalid number of eigenvectors to plot.")

    if figsize is None:
        figsize = (8, 2 * n_evects)

    n_pars = active_subspaces_object.evects.shape[0]
    #fig, axes = plt.subplots(n_evects, 1, figsize=figsize)
    #fig.suptitle(title)
    # to ensure generality for subplots (1, 1)
    active_subspace_eigenvectors_axes = np.array(active_subspace_eigenvectors_axes)
    for i, ax in enumerate(active_subspace_eigenvectors_axes.flat):
        ax.scatter(
            range(1, n_pars + 1),
            active_subspaces_object.evects[: n_pars + 1, i],
            c="blue",
            s=60,
            alpha=0.9,
            edgecolors="k",
        )
        ax.axhline(linewidth=0.7, color="black")

        ax.set_xticks(range(1, n_pars + 1))
        if labels:
            ax.set_xticklabels(labels)

        ax.set_ylabel(f"Active eigenvector {i + 1}")
        ax.grid(linestyle="dotted")
        ax.axis([0, n_pars + 1, -1 - 0.1, 1 + 0.1])
        ax.set_facecolor("xkcd:dark grey")

    active_subspace_eigenvectors_axes.flat[-1].set_xlabel("Eigenvector components")
    #fig.tight_layout()
    # tight_layout does not consider suptitle so we adjust it manually
    #plt.subplots_adjust(top=0.94)
    #ax.figure=fig
    #self.add_child_axes(fig)
    #breakpoint()
    #if filename:
    #    plt.savefig(filename)
    #else:
    #    return fig
    return active_subspace_eigenvectors_axes


def plot_activesubspace_sufficient_summary(
    active_subspaces_object,
    inputs,
    outputs,
    result_object,
    next_point=None,
    active_subspace_figure=None,
    active_subspace_sufficient_summary_axes=None,
    filename=None,
    figsize=(10, 8),
    title="",
):
    """
    Plot the sufficient summary.

    :param numpy.ndarray inputs: array n_samples-by-n_params containing the
        points in the full input space.
    :param numpy.ndarray outputs: array n_samples-by-1 containing the
        corresponding function evaluations.
    :param str filename: if specified, the plot is saved at `filename`.
    :param tuple(int,int) figsize: tuple in inches defining the figure
        size. Defaults to (10, 8).
    :param str title: title of the plot.
    :raises: ValueError, TypeError

    .. warning:: `active_subspaces_object.fit` has to be called in advance.

        Plot only available for partitions up to dimension 2.
    """
    #ax = self or plt.gca()
    if active_subspaces_object.evects is None:
        raise TypeError(
            "The eigenvectors have not been computed."
            "You have to perform the fit method."
        )

    #plt.figure(figsize=figsize)
    #plt.title(title)
    #sufficient_summary_fig = plt.figure(figsize=figsize)
    #sufficient_summary_fig.suptitle(title)
    #ax = sufficient_summary_fig.add_subplot(111)

    best_point, best_value = expected_ucb(result_object, alpha=0.0)
    tuner_sample_points = np.asarray(result_object.x_iters)
    best_point_normalized_zero_to_one = result_object.space.transform([best_point])
    next_point_normalized_zero_to_one = result_object.space.transform([next_point])
    tuner_sample_points_normalized_zero_to_one = result_object.space.transform(
        tuner_sample_points
    )
    #best_point_normalized_minus_one_to_one = Normalizer(0, 1).fit_transform(
        #best_point_normalized_zero_to_one
    #)
    best_point_normalized_minus_one_to_one = best_point_normalized_zero_to_one * 2 - 1
    next_point_normalized_minus_one_to_one = next_point_normalized_zero_to_one * 2 - 1
    #tuner_sample_points_normalized_minus_one_to_one = Normalizer(0, 1).fit_transform(
        #tuner_sample_points_normalized_zero_to_one
    #)
    tuner_sample_points_normalized_minus_one_to_one = (
        tuner_sample_points_normalized_zero_to_one * 2 - 1
    )

    if active_subspaces_object.dim == 1:
        active_subspace_sufficient_summary_axes.scatter(
            active_subspaces_object.transform(inputs)[0],
            outputs,
            c="blue",
            s=40,
            alpha=0.9,
            edgecolors="k",
        )
        active_subspace_sufficient_summary_axes.set_xlabel(
            "Active variable " + r"$W_1^T \mathbf{\mu}}$", fontsize=18
        )
        active_subspace_sufficient_summary_axes.set_ylabel(
            r"$f \, (\mathbf{\mu})$", fontsize=18
        )
    elif active_subspaces_object.dim == 2:
        active_subspace_x = active_subspaces_object.transform(inputs)[0]
        active_subspace_best_point = active_subspaces_object.transform(
            best_point_normalized_minus_one_to_one
        )[0]
        active_subspace_next_point = active_subspaces_object.transform(
            next_point_normalized_minus_one_to_one
        )[0]
        active_subspace_tuner_sample_points = active_subspaces_object.transform(
            tuner_sample_points_normalized_minus_one_to_one
        )[0]
        #scatter_plot= active_subspace_sufficient_summary_axes.scatter(
            #active_subspace_x[:, 0],
            #active_subspace_x[:, 1],
            #c=outputs.reshape(-1),
            #s=60,
            #alpha=0.9,
            #edgecolors='k',
            #vmin=np.min(outputs),
            #vmax=np.max(outputs)
            #)
        contour_plot = active_subspace_sufficient_summary_axes.tricontourf(
            active_subspace_x[:, 0],
            active_subspace_x[:, 1],
            outputs.reshape(-1),
            levels=20,
            alpha=0.9,
            cmap="viridis_r",
            edgecolors="k",
            vmin=np.min(outputs),
            vmax=np.max(outputs),
        )
        active_subspace_sufficient_summary_axes.scatter(
            active_subspace_tuner_sample_points[:, 0],
            active_subspace_tuner_sample_points[:, 1],
            c="k",
            s=20,
            lw=0.0,
            alpha=0.25,
        )
        active_subspace_sufficient_summary_axes.scatter(
            active_subspace_next_point[0, 0],
            active_subspace_next_point[0, 1],
            c=["xkcd:pink"],
            s=20,
            lw=0.0,
        )
        active_subspace_sufficient_summary_axes.scatter(
            active_subspace_best_point[0, 0],
            active_subspace_best_point[0, 1],
            c=["r"],
            s=20,
            lw=0.0,
        )
        active_subspace_sufficient_summary_axes.set_xlabel(
            "First active variable", fontsize=18
        )
        active_subspace_sufficient_summary_axes.set_ylabel(
            "Second active variable", fontsize=18
        )
        ymin = 1.1 * np.amin(
            [np.amin(active_subspace_x[:, 0]), np.amin(active_subspace_x[:, 1])]
        )
        ymax = 1.1 * np.amax(
            [np.amax(active_subspace_x[:, 0]), np.amax(active_subspace_x[:, 1])]
        )
        active_subspace_sufficient_summary_axes.axis("equal")
        active_subspace_sufficient_summary_axes.axis([ymin, ymax, ymin, ymax])

        active_subspace_figure.colorbar(
            contour_plot, ax=active_subspace_sufficient_summary_axes
        )
    else:
        raise ValueError(
            "Sufficient summary plots cannot be made in more than 2 dimensions."
        )

    active_subspace_sufficient_summary_axes.set_facecolor("xkcd:dark grey")
    active_subspace_sufficient_summary_axes.grid(linestyle="dotted")
    #sufficient_summary_fig.tight_layout()

    #if filename:
    #    sufficient_summary_fig.savefig(filename)
    #else:
    #    return sufficient_summary_fig
    return active_subspace_sufficient_summary_axes
