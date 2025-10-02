from __future__ import annotations

import itertools


def agg_vector(
    agg: str,
    col: str,
    length: int,
    window: str = "()",
) -> str:
    """
    Element-wise Aggregation over List or Array

    Parameters
    ----------
    agg : str
        Aggregation Function
    col : str
        Column of List or Array
    length : int
        Length of ``col``
    window : str, optional
        WINDOW clause

    Returns
    -------
    str
        Element-wise Aggregation Expression
    """
    return (
        "["
        + ",".join((f"""{agg}({col}[{i + 1}]) OVER {window}""" for i in range(length)))
        + "]"
    )


def agg_stats(
    mu: str,
    sigma: str,
    count: str,
    window: str = "()",
) -> tuple[str, str]:
    """
    Aggregate Stats
    """
    mu_agg = f"""(sum({mu} * {count}) OVER {window}) / (sum({count}) OVER {window})"""

    sigma_agg = f"""sqrt(((sum({count} * (pow(({mu}), 2) + pow(({sigma}), 2))) OVER {window}) / (sum({count}) OVER {window}) - pow(({mu_agg}), 2)))"""

    return mu_agg, sigma_agg


def agg_data_stats(
    key: str,
    length: int,
    window: str = "()",
) -> str:
    """
    Aggregate Stats for Data

    Parameters
    ----------
    key : str
        Feature Key for Data
    length : int
        Length of List or Array
    window : str
        WINDOW Clause

    Returns
    -------
    str
        Aggregation Query Expression
    """
    m, s = itertools.tee(
        (
            agg_stats(
                f'"stats"."{key}"."mean"[{i + 1}]',
                f'"stats"."{key}"."std"[{i + 1}]',
                f'"stats"."{key}"."count"[1]',
                window,
            )
            for i in range(length)
        )
    )
    mu = f"[{','.join((mi for (mi, _) in m))}]"
    sigma = f"[{','.join((si for (_, si) in s))}]"

    max_ = agg_vector("max", f'"stats"."{key}"."max"', length)
    min_ = agg_vector("min", f'"stats"."{key}"."min"', length)

    return f"""struct_pack(
      "max":={max_},
      "min":={min_},
      "mean":={mu},
      "std":={sigma}
    )"""


def agg_image_stats(
    key: str,
    window: str = "()",
) -> str:
    """
    Aggregate Stats for Image

    Parameters
    ----------
    key : str
        Feature Key for Image
    window : str, optional
        WINDOW Clause

    Returns
    -------
    mu_agg : str
    sigma_agg : str

    Notes
    -----
    LeRobot image stats is pixel invariant.
    Its shape is (H=1, W=1, C=3).
    """
    m, s = itertools.tee(
        (
            agg_stats(
                f'"stats"."{key}"."mean"[{i + 1}][1][1]',
                f'"stats"."{key}"."std"[{i + 1}][1][1]',
                f'"stats"."{key}"."count"[1]',
                window,
            )
            for i in range(3)
        )
    )

    mu = f"[{','.join((f'[[{mi}]]' for (mi, _) in m))}]"
    sigma = f"[{','.join((f'[[{si}]]' for (_, si) in s))}]"

    max_ = agg_vector("max", f'"stats"."{key}"."max"[1][1]', 1)
    min_ = agg_vector("min", f'"stats"."{key}"."min"[1][1]', 1)

    return f"""struct_pack(
      "max":={max_},
      "min":={min_},
      "mean":={mu},
      "std":={sigma}
    )"""
