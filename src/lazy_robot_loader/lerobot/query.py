from __future__ import annotations

import itertools

from lazy_robot_loader.lerobot.core import LeRobotDatasetFeature


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
        "list_value("
        + ",".join((f"""{agg}[{i + 1}]({col}) OVER {window}""" for i in range(length)))
        + ")"
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


def agg_vector_stats(
    mu: str,
    sigma: str,
    count: str,
    length: int,
    window: str = "()",
) -> tuple[str, str]:
    """
    Aggregate Stats for List or Array

    Parameters
    ----------
    mu : str
        Mean Column
    sigma : str
        Standard Deviation Column
    length : int
        Length of List or Array
    window : str
        WINDOW Clause

    Returns
    -------
    mu_agg : str
    sigma_agg : str
    """
    m, s = itertools.tee(
        (
            agg_stats(
                mu + f"[{i + 1}]",
                sigma + f"[{i + 1}]",
                count,
                window,
            )
            for i in range(length)
        )
    )

    return (
        f"list_value({','.join(m)})",
        f"list_value({','.join(s)})",
    )


def agg_image_stats(
    mu: str,
    sigma: str,
    count: str,
    window: str = "()",
) -> tuple[str, str]:
    """
    Aggregate Stats for Image

    Parameters
    ----------
    mu : str
        Mean Column. Shape is (H=1, W=1, C=3)
    sigma : str
        Standard Deviation Column. Shape is (H=1, W=1, C=3)
    count : str
        Counts of Each Group Column
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
                mu + f"[{i + 1}][1][1]",
                sigma + f"[{i + 1}][1][1]",
                count,
                window,
            )
            for i in range(3)
        )
    )

    return (
        f"list_value({','.join((f'[[{mi}]]' for mi in m))})",
        f"list_value({','.join((f'[[{si}]]' for si in s))})",
    )


def agg_feature_stats(
    key: str,
    feature: LeRobotDatasetFeature,
    window: str = "()",
) -> tuple[str, str]:
    """
    Aggregate Stats for Feature

    Parameters
    ----------
    key : str
        Feature Key
    feature : LeRobotDatasetFeature
        Feature
    window : str, optional
        WINDOW Clause

    Returns
    -------
    mu_agg : str
    sigma_agg : str
    """
    mu = f'"stats"."{key}"."mean"'
    sigma = f'"stats"."{key}"."std"'
    count = f'"stats"."{key}"."count"[1]'

    if feature["dtype"] in ["image", "video"]:
        return agg_image_stats(
            mu,
            sigma,
            count,
            window,
        )

    return agg_vector_stats(
        mu,
        sigma,
        count,
        feature["shape"][0],
        window,
    )
