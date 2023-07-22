"""Tools to extract and plot the data from the results."""
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple, Callable, Any, Iterable, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.ticker import Formatter
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import entropy

default_keys = ['calibration', 'brier score', 'negative log likelihood', 'accuracy']

additional_keys = {
    'BayesianProgressiveNeuralNetwork': ['McAllester bound', 'Catoni bound']
}

entropy_keys = ['ID_h', 'OOD_h']


def get_value_default(x, *args):
    """Identity function.

    Args:
        x: The argument

    Returns:
        x: The argument
    """
    return x


def get_value_calibration(x, *args):
    """Returns the first element of the argument.

    Args:
        x: The argument

    Returns:
        The first element of the argument
    """
    return x[0]


get_value_dict = {'calibration': get_value_calibration}


def full_dict(root_path: str) -> Dict[str, Dict]:
    """Combines all json files in ``root_path`` to one dict.

    Args:
        root_path: The path to the folder with the json files

    Returns:
        The combined dict with the file names as keys
    """
    d = {}
    for f in os.listdir(root_path):
        if f.endswith('.json'):
            d[f[:-5]] = json.load(open(os.path.join(root_path, f)))
    return d


def average_values(metrics: List[Dict],
                   key: str,
                   get_value: Callable = get_value_default):
    """Averages the test values.

    Args:
        metrics: The metrics dict
        key: The key of the metrics
        get_value: The function to extract the value from the value
            corresponding to the key

    Returns:
        The averaged values for each dataset
    """
    return [get_value(metrics[i]['test'][0][key], 0) for i in range(len(metrics))]
    # return [np.mean([get_value(
    # [j]['test'][i][key], i)
    #                  for j in range(i, len(metrics))
    #                  if key in metrics[j]['test'][i].keys()])
    #         for i in range(len(metrics))]


def get_entropy_df(metrics: List[Dict],
                   datasets: List[str]) -> pd.DataFrame:
    """Returns the pairwise entropies used in the entropy heatmap.

    Args:
        metrics: The metrics dict
        datasets: The dataset names

    Returns:
        A DataFrame containing the pairwise entropies
    """
    l = []
    for i, column in enumerate(metrics):
        test_metrics = column['test']
        l.append(
            pd.DataFrame(
                {dataset: dataset_metrics['entropy'] for dataset, dataset_metrics in zip(datasets, test_metrics)},
                index=[f'column {i + 1}' for i in range(len(test_metrics[0]['entropy']))]))
    df_concat = pd.concat(l)
    entropy_df = df_concat.groupby(df_concat.index).mean().T
    return entropy_df


def add_parameters(d: Dict[str, Any],
                   name: str,
                   model_type):
    """Extracts the parameters from the name and adds them to the dict.

    Args:
        d: The dict the parameters should be added to
        name: The name of the model
        model_type: The type of the model. Possible choices:
            'BayesianProgressiveNeuralNetwork',
            'DropoutProgressiveNeuralNetwork',
            'ProgressiveNeuralNetwork'
    """
    split = name.split('_')
    if model_type == 'BayesianProgressiveNeuralNetwork':
        d['temperature scaling'] = float(split[-1])
        d['curvature scaling'] = split[-2]
        d['weight decay'] = float(split[-3])
        d['criterion'] = split[-4]
        d['curvature type'] = split[-5]
        d['empirically learned prior'] = 'isotropic' not in name
    elif model_type == 'DropoutProgressiveNeuralNetwork':
        d['weight decay'] = float(split[-1])
        d['dropout probability'] = float(split[-2])
    elif model_type == 'ProgressiveNeuralNetwork':
        d['weight decay'] = float(split[-1])
    else:
        warnings.warn(f'model type {model_type} not known during evaluation. '
                      f'Supported model types are '
                      f'BayesianProgressiveNeuralNetwork, '
                      f'DropoutProgressiveNeuralNetwork and '
                      f'ProgressiveNeuralNetwork')


def metrics_table(d: Dict[str, Dict],
                  datasets: List[str]) \
        -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Extracts the main metrics and parameters from the full dict.

    Args:
        d: The full dict (as returned by ``full_dict``)
        datasets: The names of the datasets

    Returns:
        A dataframe of all metrics and parameters for each column and dataset
        The entropy dataframes as returned by ``get_entropy_df``
    """
    out = {}
    entropy_dfs = {}
    for name, model_dict in d.items():
        out[name] = {}
        model_type = model_dict['kwargs']['model'].split('(')[0] if 'kwargs' in model_dict.keys() \
            else 'BayesianProgressiveNeuralNetwork'

        try:
            # parameters: curvature_type, criterion_type, weight_decay,
            #   curvature_scaling_name, temperature_scaling, dropout_probability
            add_parameters(out[name], name, model_type)
        except ValueError:
            continue

        # standard metrics
        for key in default_keys + additional_keys.get(model_type, []):
            get_value = get_value_dict.get(key, get_value_default)
            out[name][key] = average_values(model_dict['metrics'], key, get_value=get_value)

        # # entropy
        # entropy_df = get_entropy_df(model_dict['metrics'], datasets)
        # entropy_dfs[name] = entropy_df.copy(True)
        # out[name]['ID_h'] = entropy_df.values[np.diag_indices_from(entropy_df)]
        # entropy_df.values[np.diag_indices_from(entropy_df)] = np.nan
        # out[name]['OOD_h'] = entropy_df.mean().values

        if 'ood' in model_dict.keys():
            out[name]['OOD_h'] = model_dict['ood'][0]['entropy'][0]

    out_df = pd.concat({k: pd.DataFrame(v).T for k, v in out.items()}, axis=0)
    all_additional_keys = list({value for values in additional_keys.values() for value in values})
    metric_keys = default_keys + all_additional_keys + entropy_keys
    out_df['Average'] = out_df[0]
    out_df.loc[(slice(None), metric_keys), 'Average'] = out_df.loc[(slice(None), metric_keys), :].mean(axis=1)
    return out_df


def entropy_histogram(d: Dict[str, Dict],
                      datasets: List[str]) \
        -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Extracts the data for the entropy histogram from the full dict.

    Args:
        d: The full dict (as returned by ``full_dict``)
        datasets: The names of the datasets

    Returns:
        A dict that maps name, dataset, column -> list of per-example entropies
    """
    entropy_histograms = {}
    for name, model_dict in d.items():
        data = {dataset: {f'column {i + 1}': entropy_histogram
                          for i, entropy_histogram in enumerate(dataset_metrics['entropy histogram'])}
                for dataset, dataset_metrics in zip(datasets, model_dict['metrics'][-1]['test'])}
        entropy_histograms[name] = data
    return entropy_histograms


def continual_learning_metrics(d: Dict[str, Dict]) \
        -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
    """Extracts the continual learning metrics from the full dict.

    Args:
        d: The full dict (as returned by ``full_dict``)

    Returns:
        A dict mapping a model name on a list of lists for each dataset
            containing the accuracies
        A dict mapping a model name on a list of lists for each dataset
            containing the x-axis values
    """
    accuracies = {}
    x_axis = {}
    for name, model_dict in d.items():
        accuracies[name] = []
        x_axis[name] = []
        for i, metric in enumerate(model_dict['metrics']):
            train_accuracies = [v['accuracy'] for v in metric['train']['test']]
            accuracies[name].append(train_accuracies)
            len_train_accuracies = len(train_accuracies)
            x_axis[name].append([j / len_train_accuracies + i for j in range(len_train_accuracies)])
            for x, acc, test_metrics in zip(x_axis[name], accuracies[name], metric['test']):
                x.append(float(i + 1))
                acc.append(test_metrics['accuracy'])
    return accuracies, x_axis


def unstack_and_type(full_df: pd.DataFrame, column='Average') -> pd.DataFrame:
    """Unstacks a DataFrame column returned by ``metrics_table`` and types the
    resulting columns.

    Args:
        full_df: The DataFrame returned by ``metrics_table``
        column: A column of the unstacked DataFrame

    Returns:
        The unstacked and typed DataFrame
    """
    df = full_df[column].unstack(level=1)
    df = df.astype({'empirically learned prior': 'boolean',
                    'accuracy': 'float64',
                    'Catoni bound': 'float64'})
    return df


def best_models_accuracies_transfer(full_df: pd.DataFrame) -> pd.DataFrame:
    """Filters the best models w.r.t. accuracy.

    Args:
        full_df: The DataFrame returned by ``metrics_table``

    Returns:
        A DataFrame with the best models w.r.t. accuracy
    """
    out = pd.DataFrame()
    argmax = lambda df: df['accuracy'].idxmax()

    df = unstack_and_type(full_df, 0)

    mle = df[(df['weight decay'] == 0.) &
             df['empirically learned prior'].isna() &
             df['dropout probability'].isna()]
    out = out.append(mle.loc[argmax(mle)])

    map = df[(df['weight decay'] > 0.) &
             df['empirically learned prior'].isna() &
             df['dropout probability'].isna()]
    out = out.append(map.loc[argmax(map)])

    mc_dropout = df[df['empirically learned prior'].isna() &
                    (df['dropout probability'] > 0)]
    out = out.append(mc_dropout.loc[mc_dropout['accuracy'].idxmax()])

    isotropic = df[~df['empirically learned prior']]
    idx = isotropic.groupby('weight decay').apply(argmax)
    out = out.append(isotropic.loc[idx])

    empirical_prior = df[df['empirically learned prior']]
    idx = empirical_prior.groupby('curvature type').apply(argmax)
    out = out.append(empirical_prior.loc[idx])

    return out


def best_models_catoni(full_df: pd.DataFrame, column: int = 0) -> pd.DataFrame:
    """Filters the best models w.r.t. the Catoni bound.

    The DataFrame is grouped by the approach.

    Args:
        full_df: The DataFrame returned by ``metrics_table``
        column: A column of the unstacked DataFrame

    Returns:
        A DataFrame with the best models w.r.t. the Catoni bound
    """
    df = unstack_and_type(full_df, column=column)

    idx = df.groupby(['criterion', 'curvature scaling']).apply(lambda df: df['Catoni bound'].idxmin())
    return df.loc[idx].groupby(['criterion', 'curvature scaling']).min()


def best_models_catoni2(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the best models w.r.t. the Catoni bound.

    The DataFrame is grouped by the approach.

    Args:
        df: The DataFrame returned by ``metrics_table`` and unstacked and typed

    Returns:
        A DataFrame with the best models w.r.t. the Catoni bound
    """
    group = df.groupby(['criterion', 'curvature scaling'])
    curvature_scalings = ['Catoni bound', 'McAllester bound']
    columns = {}
    for scaling in curvature_scalings:
        columns[(scaling, 'mean')] = group[scaling].mean()
        columns[(scaling, 'min')] = group[scaling].min()
        columns[(scaling, 'max')] = group[scaling].max()
    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=['PAC Bayes bound', 'metric'])
    return out


def best_models_accuracies_continual(full_df: pd.DataFrame) \
        -> Tuple[pd.DataFrame, List[str]]:
    """Filters the best models w.r.t. accuracy.

    Args:
        full_df: The DataFrame returned by ``metrics_table``

    Returns:
        A DataFrame with the best models w.r.t. accuracy
        A List of the optimal model names
    """
    df = unstack_and_type(full_df, 'Average')

    method_fn = lambda name: 'BPNN' if 'BPNN' in name else \
        'PNN + MC dropout' if 'DPNN' in name else 'PNN'
    idx = df.groupby(method_fn).apply(lambda df: df['accuracy'].idxmax())
    return full_df.loc[(idx.values, slice(None)), :], idx.values.tolist()


def cold_posterior_plot(full_df: pd.DataFrame) -> Figure:
    """Shows the cold posterior effect of different models.

    Args:
        full_df: The DataFrame returned by ``metrics_table``

    Returns:
        A figure of the temperature scaling against the accuracy
    """
    df = full_df[0].unstack(level=1)
    with_prior = ~df['empirically learned prior'].isna()

    def _filter_and_average(weight_decay, new_name):
        mean = df[with_prior & (df['weight decay'] == weight_decay)] \
            .groupby('temperature scaling').mean(False)
        return mean[['accuracy']].rename({'accuracy': new_name}, axis='columns')

    means = [_filter_and_average(weight_decay, new_name)
             for weight_decay, new_name in [(1e-3, 'isotropic 1e-3'),
                                            (1e-5, 'isotropic 1e-5'),
                                            (1e-8, 'empirically learned prior')]]
    temperature_accuracy = pd.concat(means, axis='columns')
    f, ax = plt.subplots()
    temperature_accuracy.plot(ax=ax)
    ax.semilogx()
    return f


def cold_posterior_plot2(df: pd.DataFrame) -> Figure:
    f, ax = plt.subplots()
    sns.lineplot(data=df, x='temperature scaling', y='accuracy', hue='empirically learned prior',
                 style='weight decay', ax=ax)
    ax.semilogx()
    return f


def heatmap(data: np.array,
            row_labels: Iterable,
            col_labels: Iterable,
            ax: Axes = None,
            add_cbar: bool = False,
            cbar_kw: Dict = {},
            cbarlabel: str = "",
            **kwargs) -> Tuple[AxesImage, Optional[Colorbar]]:
    """Create a heatmap from a numpy array and two lists of labels.

    Adapted from
    <https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html>_

    Args:
        data: A 2D numpy array of shape (N, M).
        row_labels: A list or array of length N with the labels for the rows.
        col_labels: A list or array of length M with the labels for the columns.
        ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
            If not provided, use current axes or create a new one.  Optional.
        add_cbar: Whether a colorbar should be added
        cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`
        cbarlabel: The label for the colorbar.  Optional.
        **kwargs: All other arguments are forwarded to `imshow`.

    Returns:
        A AxesImage and if ``add_cbar`` a colorbar
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    for im in ax.get_images():
        im.set_clim(0, 2.)

    if add_cbar:
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im: AxesImage,
                     data: Optional[Iterable] = None,
                     valfmt: Union[str, Formatter] = "{x:.2f}",
                     textcolors: Tuple[str, str] = ("black", "white"),
                     threshold: Optional[float] = None,
                     **textkw) -> List[Text]:
    """A function to annotate a heatmap.

    Adapted from
    <https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html>_

    Args:
        im: The AxesImage to be labeled.
        data: Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt: The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors: A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold:  Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **textkw: All other arguments are forwarded to each call to `text` used to create
            the text labels.

    Returns:
        The annotations of the heatmap
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_entropy(entropy_df: pd.DataFrame,
                 ax: Axes,
                 add_cbar: bool = True):
    """Adds the entropy heatmap to the given axes.

    Args:
        entropy_df: A entropy DataFrame from ``metrics_table``
        ax: A matplotlib axes
        add_cbar: Whether a colorbar should be added
    """
    im, cbar = heatmap(entropy_df.values, entropy_df.columns, entropy_df.index,
                       ax=ax, add_cbar=add_cbar, cbarlabel='entropy')
    texts = annotate_heatmap(im, valfmt="{x:.2f}", threshold=1.)


def plot_entropies(entropy_dfs: Dict[str, pd.DataFrame],
                   names: List[str]) -> Figure:
    """Produces a common plot for all entropy heatmaps for given names.

    Args:
        entropy_dfs: The entropy DataFrames from ``metrics_table``
        names: List of keys of ``entropy_dfs`` that should be plotted

    Returns:
        A figure of the entropy heatmaps
    """
    f, axes = plt.subplots(ncols=len(names), figsize=(20, 10))
    for i, (ax, name) in enumerate(zip(axes, names)):
        plot_entropy(entropy_dfs[name], ax, add_cbar=False)
        ax.set_title(name)
    ax = f.add_axes([0.25, 0.1, .5, 0.05])
    cbar = f.colorbar(f.axes[0].get_images()[0], cax=ax, orientation='horizontal')
    cbar.ax.set_xlabel('entropy')
    return f


def symmetric_kl_divergence(pk: np.array,
                            qk: np.array,
                            eps: float = 1e-7) -> float:
    """Computes the symmetric variant of the discrete symmetric KL-divergence.

    The arrays are normalized to sum to one.

    Args:
        pk: The first array
        qk: The second array
        eps: A small scalar that is added to both arrays

    Returns:
        The symmetric KL-divergence
    """
    pk_eps = pk + eps
    qk_eps = qk + eps
    return entropy(pk_eps, qk_eps) + entropy(qk_eps, pk_eps)


def entropy_histogram_plot(data: Dict[str, Dict[str, List[float]]],
                           datasets: List[str]) -> Figure:
    """Plots the histograms of the entropy values for all columns and datasets.

    Additionally, the symmetric KL-divergence between the in-distribution and
    out-of-distribution entropy histograms is shown.

    Args:
        data: A dict mapping a dataset on a dict mapping a column on the list of
            per-example entropies (output of ``entropy_histogram``)
        datasets: A list of dataset names

    Returns:
        A figure of the entropy histograms
    """
    x_min = min(e for val in data.values() for v in val.values() for e in v)
    x_max = max(e for val in data.values() for v in val.values() for e in v)
    f, axes = plt.subplots(len(datasets), len(datasets), sharex=True, sharey=True, figsize=(20, 20))
    plt.setp(axes.flat, xlabel='h', ylabel='Frequency')
    pad = 5  # in points
    for ax, col in zip(axes[0], list(data.values())[0].keys()):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axes[:, 0], data.keys()):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    entropy_distribution = [[np.histogram(column_metrics, bins=50, range=[x_min, x_max])[0]
                             for column_metrics in dataset_metrics.values()]
                            for dataset_metrics in data.values()]
    for i, (dataset, dataset_metrics) in enumerate(data.items()):
        for j, (column, column_metrics) in enumerate(dataset_metrics.items()):
            if i == j:
                color = 'blue'
            else:
                color = 'orange'
            ax = axes[i, j]
            sns.histplot(column_metrics, ax=ax, stat='probability', bins=50, binrange=[x_min, x_max], color=color)
            if i != j:
                kl_sym = symmetric_kl_divergence(entropy_distribution[j][j], entropy_distribution[i][j])
                ax.text(.5, .8, f'KL_sym: {kl_sym:.1f}', ha='center', va='center',
                        bbox=dict(facecolor='none', edgecolor='black', pad=10.0), transform=ax.transAxes)
    return f


def plot_continual_learning(accuracies: Dict[str, List[List[float]]],
                            x_axis: Dict[str, List[List[float]]],
                            names: List[str]) -> Figure:
    """Plots the accuracies during the training of the different tasks.

    Args:
        accuracies: The first output of ``continual_learning_metrics``
        x_axis: The second output of ``continual_learning_metrics``
        names: The names of the models that should be shown

    Returns:
        A figure of the continual learning plot
    """
    n_tasks = len(next(iter(accuracies.values())))
    f, axes = plt.subplots(nrows=n_tasks, ncols=1, sharex=True, sharey=True, figsize=(10, 10))
    for i, ax in enumerate(axes):
        ax.set_xticks([], minor=False)
        spines = ['right', 'top']
        if i < len(axes) - 1:
            spines.append('bottom')
        for spine in spines:
            ax.spines[spine].set_visible(False)
    for name, ys in accuracies.items():
        if name not in names:
            continue
        xs = x_axis[name]
        for x, y, ax in zip(xs, ys, axes):
            ax.plot(x, y, label=name)
    for i in range(1, n_tasks + 1):
        axes[0].axvline(i, linestyle='dotted', color='gray', clip_on=False)
        for ax in axes[1:]:
            ax.axvline(i, ymax=1.2, linestyle='dotted', color='gray', clip_on=False)
    handles, labels = axes[-1].get_legend_handles_labels()
    f.legend(handles, labels, loc='lower center')
    return f
