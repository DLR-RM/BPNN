"""Tools to extract and plot the data from the results."""
import json
import os
from typing import Dict, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm


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


def extract_df_continual_learning(root_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combines all json files in ``root_path`` and extracts the main metrics
    and parameters for each continual learning dataset and method.

    Args:
        root_path: The path to the folder with the json files

    Returns:
        A dataframe containing the mean of the metrics for each dataset and method
        A dataframe containing the standard deviation of the metrics for each dataset and method
    """
    d = {}
    for f in tqdm(os.listdir(root_path)):
        if f.endswith('.json'):
            d_json = json.load(open(os.path.join(root_path, f)))
            d[f[:-5]] = [a['test'][0]['accuracy'] for a in d_json['metrics']]

    df = pd.DataFrame(d)
    df_out_mean = df.T.groupby(lambda index: index[2:], axis=0).head(5).groupby(lambda index: index[2:], axis=0).mean()
    df_out_mean['Average'] = df.T.groupby(lambda index: index[2:], axis=0).head(5).groupby(
        lambda index: index[2:],
        axis=0).mean().T.mean()
    df_out_std = df.T.groupby(lambda index: index[2:], axis=0).head(5).groupby(lambda index: index[2:], axis=0).std()
    df_out_std['Average'] = df.T.groupby(lambda index: index[2:], axis=0).head(5).groupby(
        lambda index: index[2:],
        axis=0).std().T.mean()
    return df_out_mean, df_out_std


def extract_df_transfer_learning(d: Dict[str, Dict]):
    """Extracts the main metrics and parameters from the full dict.

    Args:
        d: The full dict (as returned by ``full_dict``)

    Returns:
        A dataframe of all metrics and parameters for each column and dataset
    """
    d_new = {}
    for key, value in d.items():
        d_out = {
            'accuracy': value['metrics'][0]['test'][0]['accuracy'],
            'id entropy': value['metrics'][0]['test'][0]['entropy'][0],
            'ood entropy': value['ood'][0]['entropy'][0],
            'calibration': value['metrics'][0]['test'][0]['calibration'][0],
            'negative log likelihood': value['metrics'][0]['test'][0]['negative log likelihood'],
        }
        if 'McAllester bound' in value['metrics'][0]['test'][0].keys():
            d_out.update(
                {
                    'McAllester bound': value['metrics'][0]['test'][0]['McAllester bound'],
                    'Catoni bound': value['metrics'][0]['test'][0]['Catoni bound'],
                }
            )

        d_new[key] = d_out
    df = pd.DataFrame(d_new).T

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    drop_labels = [i for i in df.index if not is_number(i.split('_')[-1])]
    df = df.drop(drop_labels)
    df['model'] = df.apply(
        lambda row:
        'PNN' if row.name.split('_')[-6] == 'PNN' else
        'DPNN' if row.name.split('_')[-6] == 'DPNN' else
        'PBNN', axis=1)
    df['weight decay'] = df.apply(
        lambda row:
        float(row.name.split('_')[-1]) if row['model'] == 'PNN' else
        float(row.name.split('_')[-1]) if row['model'] == 'DPNN' else
        float(row.name.split('_')[-3]), axis=1)
    df['dropout probability'] = df.apply(
        lambda row:
        float(row.name.split('_')[-2]) if row['model'] == 'DPNN' else
        float('NaN'), axis=1)
    df['prior type'] = df.apply(
        lambda row:
        'learned prior' if row.name.split('_')[1] == 'learned' else
        'isotropic' if row.name.split('_')[1] == 'isotropic' else
        'zero mean isotropic' if row.name.split('_')[1] == 'zero' else
        float('NaN'), axis=1)
    df['curvature type'] = df.apply(
        lambda row:
        row.name.split('_')[-5] if row['model'] == 'PBNN' else
        float('NaN'), axis=1)
    df['criterion'] = df.apply(
        lambda row:
        row.name.split('_')[-4] if row['model'] == 'PBNN' else
        float('NaN'), axis=1)
    df['curvature scaling'] = df.apply(
        lambda row:
        row.name.split('_')[-2] if row['model'] == 'PBNN' else
        float('NaN'), axis=1)
    df['temperature scaling'] = df.apply(
        lambda row:
        float(row.name.split('_')[-1]) if row['model'] == 'PBNN' else
        float('NaN'), axis=1)
    return df


def cold_posterior_plot(df: pd.DataFrame) -> Figure:
    """Shows the cold posterior effect of different models.

    Args:
        df: The DataFrame returned by ``extract_df_transfer_learning``

    Returns:
        A figure of the temperature scaling against the accuracy
    """
    f, ax = plt.subplots()
    sns.lineplot(
        data=df, x='temperature scaling', y='accuracy', hue='empirically learned prior',
        style='weight decay', ax=ax)
    ax.semilogx()
    return f
