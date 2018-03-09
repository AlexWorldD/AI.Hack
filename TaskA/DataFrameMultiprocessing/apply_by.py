import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_pandas


def _apply_df(args):
    df, func, num, kwargs = args
    return num, df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    """
    Parallel execution function for the DataFrame
    :param df: Input DataFrame
    :param func:
    :param kwargs: additional arguments for the df.apply() such as axis and et al.
    :return: Output DataFrame
    """
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, i, kwargs) for i, d in enumerate(np.array_split(df, workers))])
    pool.close()
    result = sorted(result, key=lambda x: x[0])
    return pd.concat([i[1] for i in result])


def apply_by_multiprocessing_progress(df, func, **kwargs):
    """
    Parallel execution function for the DataFrame
    :param df: Input DataFrame
    :param func:
    :param kwargs: additional arguments for the df.apply() such as axis and et al.
    :return: Output DataFrame
    """
    # TODO still ine progress...
    workers = kwargs.pop('workers')
    result = []
    with multiprocessing.Pool(processes=workers) as pool:
        with tqdm(total=len(df)) as pbar:
            for i, res in tqdm(enumerate(pool.map(_apply_df, [(d, func, i, kwargs) for i, d in enumerate(np.array_split(df, workers))]))):
                result.append(res)
                pbar.update()
    pbar.close()
    pool.close()
    pool.join()
    result = sorted(result, key=lambda x: x[0])
    return pd.concat([i[1] for i in result])
