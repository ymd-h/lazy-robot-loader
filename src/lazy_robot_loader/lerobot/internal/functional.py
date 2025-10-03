import io

import av
import duckdb
import numpy as np
from numpy.typing import DTypeLike
from jaxtyping import Integer, Float, Shaped
import pyarrow as pa

from lazy_robot_loader.core import Feature
from lazy_robot_loader.lerobot.core import (
    LeRobotDatasetDataStat,
    LeRobotDatasetImageStat,
)


def to_array(
    ca: pa.ChunkedArray,
    dtype: DTypeLike | None = None,
) -> Shaped[np.ndarray, "..."]:
    """
    Convert to NumPy

    Parameters
    ----------
    ca : pyarrow.ChunkedArray
    dtype: numpy.typing.DTypeLike, optional

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Since `to_numpy()` method converts pa.(FixedSize)ListArray to
    np.ndarray of dtype='object', we first convert to python
    (nested) list by `to_pylist()` method.
    """
    return np.asarray(
        ca.to_pylist(),
        dtype=dtype,
    )


def get_stat(
    con: duckdb.DuckDBPyConnection,
    key: str,
    feature: Feature,
) -> LeRobotDatasetDataStat | LeRobotDatasetImageStat:
    stat = con.query(f'SELECT "{key}".* FROM stats;').fetch_arrow_table()

    s = {c: to_array(stat[c]).squeeze(0) for c in ("max", "min", "mean", "std")}

    if len(feature.shape) > 2:
        return LeRobotDatasetImageStat(
            max=s["max"],
            min=s["min"],
            mean=s["mean"],
            std=s["std"],
        )

    return LeRobotDatasetDataStat(
        max=s["max"],
        min=s["min"],
        mean=s["mean"],
        std=s["std"],
    )


def query_video(
    con: duckdb.DuckDBPyConnection,
    video_path: str,
    timestamp: Float[np.ndarray, " N"],
) -> Integer[np.ndarray, "N H W C=3"]:
    """
    Query Video Frames

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection
    video_path : str
        Video File Path or HF URL
    timestamp : np.ndarray
        N steps of timestamp

    Returns
    -------
    N steps of RGB frames. The shape is (N, H, W, C=3)
    """
    v = (
        io.BytesIO(
            con.query(f"""
            SELECT "content" FROM read_blob('{video_path}');
            """)
            .fetch_arrow_table()["content"][0]
            .as_py()
        )
        if video_path.startswith("hf://")
        else video_path
    )

    with av.open(v, mode="r") as container:
        s = container.streams.video[0]

        # Seek to the last key frame at or just before the specified timestamp.
        # Movie must start decoding from key frame,
        # since non key frame might contain only partial information.
        container.seek(
            offset=int(timestamp[0] // s.time_base),
            stream=s,
        )

        it = iter(container.decode(video=0))
        imgs: list[Integer[np.ndarray, "H W 3"]] = []
        img: Integer[np.ndarray, "H W 3"] | None = None
        for t in timestamp:
            for f in it:
                if t <= f.time:
                    img = f.to_ndarray(format="rgb24")
                    imgs.append(img)
                    break
            else:
                # Video has ended.
                # We re-add the last frame.
                assert img is not None
                imgs.append(img)

        return np.stack(imgs)


def query_data(
    con: duckdb.DuckDBPyConnection,
    features: dict[str, Feature],
    columns: str,
    data_path: str,
    pad_column: str | None = None,
    frame_index: int = 0,
    n_steps: int = 1,
    nested: bool = True,
) -> dict[str, Shaped[np.ndarray, "{n_steps} ..."]]:
    """
    Query Data from Parquet

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    features : dict[str, Feature]
    columns: str
        String representation of selected columns (e.g. '"timestamp", "episode_index"')
    data_path : str
        Parquet file path
    pad_column : str, optional
        Column name of padding
    frame_index : int, optional
        Frame index to be queried
    n_steps : int, optional
        N steps of queried frames
    nested : bool, optional
        Queried data contain nested data

    Returns
    -------
    dict[str, np.ndarray]
    """
    pad: str
    data: duckdb.DuckDBPyRelation
    if n_steps > 1:
        pad = (
            f'e."frame_index" != d."frame_index" AS "{pad_column}"'
            if pad_column is not None
            else ""
        )

        # We use AsOf Join since frame_index might exceed episode length.
        # When the frame_index is equal to or larger than episode length,
        # the last frame is reused and pad becomes true.
        data = con.query(f"""
        SELECT
          {columns}, {pad}
        FROM (
          SELECT {frame_index} + unnest(range({n_steps})) AS "frame_index",
        ) e ASOF JOIN '{data_path}' d
        ON e."frame_index" >= d."frame_index"
        ORDER BY e."frame_index";
        """)
    else:
        pad = f'false AS "{pad_column}"' if pad_column is not None else ""
        data = con.query(f"""
        SELECT
          {columns}, {pad}
        FROM '{data_path}'
        WHERE "frame_index" = {frame_index};
        """)

    if nested:
        table = data.fetch_arrow_table()
        return {
            c: to_array(
                table[c],
                dtype=features[c].dtype,
            )
            for c in table.column_names
        }
    else:
        return {
            c: np.asarray(v, dtype=features[c].dtype)
            for c, v in data.fetchnumpy().items()
        }
