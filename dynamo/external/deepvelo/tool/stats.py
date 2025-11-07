from typing import List, Tuple, Union

from scipy.stats import mannwhitneyu, kruskal


def stats_test(
    *data: List,
    threshold: float = 0.05,
    verbose: bool = True,
    **kwargs,
) -> Tuple[float, float]:
    """
    Run statistical test on data.

    Args:
        data (list): list of data to compare. If len(data) == 2, use Mann-Whitney U test.
            Otherwise, use Kruskal-Wallis test.
        threshold (float): threshold for p-value.
        verbose (bool): whether to print the result.
        **kwargs: keyword arguments for `scipy.stats.mannwhitneyu` or `scipy.stats.kruskal`.

    Returns:
        tuple: (statistic, p-value)
    """
    if len(data) == 2:
        stat, pval = mannwhitneyu(*data, **kwargs)
    else:
        stat, pval = kruskal(*data, **kwargs)
    if verbose:
        print(f"statistic: {stat}, p-value: {pval}")
        if pval < threshold:
            print("Significant difference. Reject null hypothesis.")
        else:
            print("Insignificant difference. Accept null hypothesis.")

    return stat, pval
