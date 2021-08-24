import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from skopt.plots import _format_scatter_plot_axes

from tune.utils import expected_ucb

__all__ = ["partial_dependence", "plot_objective", "plot_activesubspace_eigenvalues", "plot_activesubspace_eigenvectors", "plot_activesubspace_sufficient_summary"]


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
    space, model, i, j=None, sample_points=None, n_samples=250, n_points=40, x_eval=None
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
        yi = []
        for x_ in xi_transformed:
            rvs_ = np.array(sample_points)  # copy
            # We replace the values in the dimension that we want to keep
            # fixed
            rvs_[:, dim_locs[i] : dim_locs[i + 1]] = x_
            # In case of `x_eval=None` rvs conists of random samples.
            # Calculating the mean of these samples is how partial dependence
            # is implemented.
            yi.append(np.mean(model.predict(rvs_)))

        return xi, yi

    else:
        xi, xi_transformed = _evenly_sample(space.dimensions[j], n_points)
        yi, yi_transformed = _evenly_sample(space.dimensions[i], n_points)

        zi = []
        for x_ in xi_transformed:
            row = []
            for y_ in yi_transformed:
                rvs_ = np.array(sample_points)  # copy
                rvs_[:, dim_locs[j] : dim_locs[j + 1]] = x_
                rvs_[:, dim_locs[i] : dim_locs[i + 1]] = y_
                row.append(np.mean(model.predict(rvs_)))
            zi.append(row)

        return xi, yi, np.array(zi).T


def plot_objective(
    result,
    levels=20,
    n_points=200,
    n_samples=30,
    size=3,
    zscale="linear",
    dimensions=None,
    n_random_restarts=100,
    alpha=0.25,
    margin=0.65,
    colors=None,
    fig=None,
    ax=None,
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
    * `fig` [Matplotlib figure, default=None]
        Figure to use for plotting. If None, it will create one.
    * `ax` [k x k axes, default=None]
        Axes on which to plot the marginals. If None, it will create appropriate
        axes.
    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    if colors is None:
        colors = plt.cm.get_cmap("Set3").colors
    space = result.space
    samples = np.asarray(result.x_iters)
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))
    z_ranges = np.zeros((space.n_dims,space.n_dims))

    if zscale == "log":
        locator = LogLocator()
    elif zscale == "linear":
        locator = None
    else:
        raise ValueError(
            "Valid values for zscale are 'linear' and 'log'," " not '%s'." % zscale
        )
    if fig is None:
        fig, ax = plt.subplots(
            space.n_dims,
            space.n_dims,
            figsize=(size * space.n_dims, size * space.n_dims),
        )
    width, height = fig.get_size_inches()

    fig.subplots_adjust(
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
                xi, yi = partial_dependence(
                    space,
                    result.models[-1],
                    i,
                    j=None,
                    sample_points=rvs_transformed,
                    n_points=n_points,
                )
                yi_min, yi_max = np.min(yi), np.max(yi)
                ax[i, i].plot(xi, yi, color=colors[1])
                if failures != 10:
                    ax[i, i].axvline(min_x[i], linestyle="--", color=colors[3], lw=1)
                    ax[i, i].axvline(min_ucb[i], linestyle="--", color=colors[5], lw=1)
                    ax[i, i].text(
                        min_x[i],
                        yi_min + 0.9 * (yi_max - yi_min),
                        f"{np.around(min_x[i], 4)}",
                        color=colors[3],
                    )
                    ax[i, i].text(
                        min_ucb[i],
                        yi_min + 0.7 * (yi_max - yi_min),
                        f"{np.around(min_ucb[i], 4)}",
                        color=colors[5],
                    )

            # lower triangle
            elif i > j:
                xi, yi, zi = partial_dependence(
                    space, result.models[-1], i, j, rvs_transformed, n_points
                )
                contour_plot = ax[i, j].contourf(xi, yi, zi, levels, locator=locator, cmap="viridis_r")
                fig.colorbar(contour_plot, ax=ax[i, j])
                ax[i, j].scatter(
                    samples[:, j], samples[:, i], c="k", s=10, lw=0.0, alpha=alpha
                )
                if failures != 10:
                    ax[i, j].scatter(min_x[j], min_x[i], c=["r"], s=20, lw=0.0)
                    ax[i, j].scatter(
                        min_ucb[j], min_ucb[i], c=["xkcd:orange"], s=20, lw=0.0
                    )
                z_ranges[i, j] = np.max(zi) - np.min(zi)
                ax[i, j].text(0.5, 0.5, np.format_float_positional(z_ranges[i, j], precision=2, unique=False, fractional=False, trim='k'), horizontalalignment='center', verticalalignment='center', transform=ax[i,j].transAxes)
    # Get all dimensions.
    plot_dims = []
    for row in range(space.n_dims):
        if space.dimensions[row].is_constant:
            continue
        plot_dims.append((row, space.dimensions[row]))
    return _format_scatter_plot_axes(
        ax,
        space,
        ylabel="Partial dependence",
        plot_dims=plot_dims,
        dim_labels=dimensions,
    )

def plot_activesubspace_eigenvalues(asub,
                                    active_subsp_fig=None,
                                    as_eigenvalues_ax=None,
                                    n_evals=None,
                                    filename=None,
                                    figsize=(8, 8),
                                    title='', **kwargs):
        """
        Plot the eigenvalues.

        :param int n_evals: number of eigenvalues to plot. If not provided all
            the eigenvalues will be plotted.
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure size.
            Default is (8, 8).
        :param str title: title of the plot.
        :raises: TypeError

        .. warning:: `asub.fit` has to be called in advance.
        """
        #ax = self #or plt.gca()
        if asub.evals is None:
            raise TypeError('The eigenvalues have not been computed.'
                            'You have to perform the fit method.')
        if n_evals is None:
            n_evals = asub.evals.shape[0]
        if n_evals > asub.evals.shape[0]:
            raise TypeError('Invalid number of eigenvalues to plot.')

        #ax = ax or plt.gca()
        #eigen_values_fig = plt.figure(figsize=figsize)
        #eigen_values_fig.suptitle(title)
        #ax = eigen_values_fig.add_subplot(111)
        if np.amin(asub.evals[:n_evals]) == 0:
            as_eigenvalues_ax.semilogy(range(1, n_evals + 1),
                        asub.evals[:n_evals] + np.finfo(float).eps,
                        'ko-',
                        markersize=8,
                        linewidth=2)
        else:
            #as_eigenvalues_ax.semilogy(range(1, n_evals + 1),
            as_eigenvalues_ax.plot(range(1, n_evals + 1),
                        asub.evals[:n_evals],
                        'ko-',
                        markersize=8,
                        linewidth=2)
            as_eigenvalues_ax.set_yscale("log")

        as_eigenvalues_ax.set_xticks(range(1, n_evals + 1))
        as_eigenvalues_ax.set_xlabel('Index')
        as_eigenvalues_ax.set_ylabel('Eigenvalues')

        if asub.evals_br is None:
            as_eigenvalues_ax.axis([
                0, n_evals + 1, 0.1 * np.amin(asub.evals[:n_evals]),
                10 * np.amax(asub.evals[:n_evals])
            ])
        else:
            if np.amin(asub.evals[:n_evals]) == 0:
                as_eigenvalues_ax.fill_between(
                    range(1, n_evals + 1),
                    asub.evals_br[:n_evals, 0] * (1 + np.finfo(float).eps),
                    asub.evals_br[:n_evals, 1] * (1 + np.finfo(float).eps),
                    facecolor='0.7',
                    interpolate=True)
            else:
                as_eigenvalues_ax.fill_between(range(1, n_evals + 1),
                                asub.evals_br[:n_evals, 0],
                                asub.evals_br[:n_evals, 1],
                                facecolor='0.7',
                                interpolate=True)
            as_eigenvalues_ax.axis([
                0, n_evals + 1, 0.1 * np.amin(asub.evals_br[:n_evals, 0]),
                10 * np.amax(asub.evals_br[:n_evals, 1])
            ])

        as_eigenvalues_ax.grid(linestyle='dotted')
        #eigen_values_fig.tight_layout

        #if filename:
        #    eigen_values_fig.savefig(filename)
        #else:
        #    return eigen_values_fig
        return as_eigenvalues_ax

def plot_activesubspace_eigenvectors(#self,
                                    asub,
                                    active_subsp_fig=None,
                                    as_eigenvectors_axs=None,
                                    n_evects=None,
                                    filename=None,
                                    figsize=None,
                                    labels=None,
                                    title=''):
        """
        Plot the eigenvectors.

        :param int n_evects: number of eigenvectors to plot. Default is asub.dim.
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure size.
            Default is (8, 2 * n_evects).
        :param str labels: labels for the components of the eigenvectors.
        :param str title: title of the plot.
        :raises: ValueError, TypeError

        .. warning:: `asub.fit` has to be called in advance.
        """
        #ax = self or plt.gca()
        if asub.evects is None:
            raise TypeError('The eigenvectors have not been computed.'
                            'You have to perform the fit method.')
        if n_evects is None:
            n_evects = asub.dim
        if n_evects > asub.evects.shape[0]:
            raise ValueError('Invalid number of eigenvectors to plot.')

        if figsize is None:
            figsize = (8, 2 * n_evects)

        n_pars = asub.evects.shape[0]
        #fig, axes = plt.subplots(n_evects, 1, figsize=figsize)
        #fig.suptitle(title)
        # to ensure generality for subplots (1, 1)
        as_eigenvectors_axs = np.array(as_eigenvectors_axs)
        for i, ax in enumerate(as_eigenvectors_axs.flat):
            ax.scatter(range(1, n_pars + 1),
                    asub.evects[:n_pars + 1, i],
                    c='blue',
                    s=60,
                    alpha=0.9,
                    edgecolors='k')
            ax.axhline(linewidth=0.7, color='black')

            ax.set_xticks(range(1, n_pars + 1))
            if labels:
                ax.set_xticklabels(labels)

            ax.set_ylabel('Active eigenvector {}'.format(i + 1))
            ax.grid(linestyle='dotted')
            ax.axis([0, n_pars + 1, -1 - 0.1, 1 + 0.1])

        as_eigenvectors_axs.flat[-1].set_xlabel('Eigenvector components')
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
        return as_eigenvectors_axs

def plot_activesubspace_sufficient_summary(asub,
                                            inputs,
                                            outputs,
                                            active_subsp_fig=None,
                                            as_sufficient_summary_ax=None,
                                            filename=None,
                                            figsize=(10, 8),
                                            title=''):
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

        .. warning:: `asub.fit` has to be called in advance.

            Plot only available for partitions up to dimension 2.
        """
        #ax = self or plt.gca()
        if asub.evects is None:
            raise TypeError('The eigenvectors have not been computed.'
                            'You have to perform the fit method.')

        #plt.figure(figsize=figsize)
        #plt.title(title)
        #sufficient_summary_fig = plt.figure(figsize=figsize)
        #sufficient_summary_fig.suptitle(title)
        #ax = sufficient_summary_fig.add_subplot(111)

        if asub.dim == 1:
            as_sufficient_summary_ax.scatter(asub.transform(inputs)[0],
                        outputs,
                        c='blue',
                        s=40,
                        alpha=0.9,
                        edgecolors='k')
            as_sufficient_summary_ax.set_xlabel('Active variable ' + r'$W_1^T \mathbf{\mu}}$',
                    fontsize=18)
            as_sufficient_summary_ax.set_ylabel(r'$f \, (\mathbf{\mu})$', fontsize=18)
        elif asub.dim == 2:
            x = asub.transform(inputs)[0]
            scatter_plot= as_sufficient_summary_ax.scatter(x[:, 0],
                        x[:, 1],
                        c=outputs.reshape(-1),
                        s=60,
                        alpha=0.9,
                        edgecolors='k',
                        vmin=np.min(outputs),
                        vmax=np.max(outputs))
            as_sufficient_summary_ax.set_xlabel('First active variable', fontsize=18)
            as_sufficient_summary_ax.set_ylabel('Second active variable', fontsize=18)
            ymin = 1.1 * np.amin([np.amin(x[:, 0]), np.amin(x[:, 1])])
            ymax = 1.1 * np.amax([np.amax(x[:, 0]), np.amax(x[:, 1])])
            as_sufficient_summary_ax.axis('equal')
            as_sufficient_summary_ax.axis([ymin, ymax, ymin, ymax])

            active_subsp_fig.colorbar(scatter_plot, ax=as_sufficient_summary_ax)
        else:
            raise ValueError(
                'Sufficient summary plots cannot be made in more than 2 '
                'dimensions.')

        as_sufficient_summary_ax.grid(linestyle='dotted')
        #sufficient_summary_fig.tight_layout()

        #if filename:
        #    sufficient_summary_fig.savefig(filename)
        #else:
        #    return sufficient_summary_fig
        return as_sufficient_summary_ax

