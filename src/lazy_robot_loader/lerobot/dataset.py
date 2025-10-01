from collections.abc import Iterable
import io
import os
import pathlib
import re
from typing import cast, Any

import av
import duckdb
import numpy as np
from numpy.typing import DTypeLike
import pyarrow as pa

from jaxtyping import Integer, Float, Shaped

from lazy_robot_loader.core import Feature
from lazy_robot_loader.lerobot.core import (
    LeRobotDatasetInfo,
    LeRobotDatasetDataStat,
    LeRobotDatasetImageStat,
)
from lazy_robot_loader.lerobot.query import agg_data_stats, agg_image_stats


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


class LeRobotDataset:
    def __init__(
        self,
        *,
        repo_id: str | None = None,
        local_path: str | pathlib.Path | None = None,
        episodes: tuple[int, ...] | None = None,
        n_observation: int = 1,
        n_action: int = 1,
        observation_regexp: str | re.Pattern = "observation.*",
        action_regexp: str | re.Pattern = "action.*",
        extra_keys: Iterable[str] = tuple(),
        cache_dir: str | pathlib.Path | None = None,
        proxy_url: str | None = None,
        proxy_username: str | None = None,
        proxy_password: str | None = None,
    ):
        """
        LeRobot Dataset

        Parameters
        ----------
        repo_id: str, optional
            Huggingface Repo ID (aka. 'user_name/repo_name')
        local_path: str | pathlib.Path, optional
            Local Directory Path
        episodes: tuple[int, ...], optional
            Episodes to be used, instead of all of them.
        n_observation: int, optional
            Number of steps for observation (model input).
            Default is 1.
        n_action: int, optional
            Number of steps for action (model output).
            Default is 1.
        observation_regexp: str | re.Pattern, optional
            Regular Expression for observation (model input).
            Default is "observation.*".
        action_regexp: str | re.Pattern, optional
            Regular Expression for action (model output).
            Default is "action.*".
        extra_keys: Iterable[str], optional
            Extra keys to be queried. No keys are queried by default.
        cache_dir: str, optional
            Cache Directory.
        proxy_url: str, optional
            Proxy URL
        proxy_username: str, optional
            Proxy User Name
        proxy_password: str, optional
            Proxy Password

        Raises
        ------
        ValueError
            Unless only one of ``repo_id`` and ``local_path`` is specified.
        NotImplementedError
            When dataset is not supported version.
        ValueError
            When there are no observations or actions found.
        """
        self.n_observation = n_observation
        self.n_action = n_action

        self._con = duckdb.connect()

        if proxy_url is not None:
            self._con.query(f"SET http_proxy = '{proxy_url}';")

            if proxy_username is not None:
                self._con.query(f"SET http_proxy_username = '{proxy_username}';")

            if proxy_password is not None:
                self._con.query(f"SET http_proxy_password = '{proxy_password}';")

        cache_dir = (
            cache_dir
            or os.environ.get("LAZY_ROBOT_LOADER_CACHE_DIR")
            or (pathlib.Path.home() / ".cache/lrl")
        )

        if isinstance(cache_dir, pathlib.Path):
            cache_dir = str(cache_dir)

        os.makedirs(os.path.join(cache_dir, "hf"), exist_ok=True)

        self._con.query(f"""
        INSTALL httpfs;
        LOAD httpfs;

        INSTALL cache_httpfs FROM community;
        LOAD cache_httpfs;

        SET cache_httpfs_cache_directory = '{cache_dir}/hf';
        SELECT cache_httpfs_wrap_cache_filesystem('HuggingFaceFileSystem');
        SET enable_external_file_cache = true;
        """)

        self._base: str
        match (repo_id, local_path):
            case (None, None):
                raise ValueError("One of `repo_id` and `local_path` must be specified.")
            case (_, None):
                self._base = f"hf://datasets/{repo_id}"
            case (None, _):
                self._base = str(local_path)
            case (_, _):
                raise ValueError(
                    f"Only one of `repo_id` and `local_path` must be specified, however, both of them are spefified. {repo_id=}, {local_path=}"
                )

        self._info = cast(
            LeRobotDatasetInfo,
            {
                k: v[0]
                for k, v in self._con.query(f"FROM '{self._base}/meta/info.json';")
                .fetch_arrow_table()
                .to_pydict()
                .items()
            },
        )

        version = self.version
        if version not in ["v2.0", "v2.1"]:
            raise NotImplementedError(
                f"Only v2.0 and v2.1 are supported, got {version}"
            )

        if isinstance(observation_regexp, str):
            observation_regexp = re.compile(observation_regexp)

        if isinstance(action_regexp, str):
            action_regexp = re.compile(action_regexp)

        self.features: dict[str, Feature] = {}
        self.observation_data_keys: list[str] = []
        self.observation_video_keys: list[str] = []
        self.action_keys: list[str] = []

        self.extra_keys: list[str] = [key for key in extra_keys]
        for k in ["episode_index", "frame_index", "timestamp"]:
            if k not in self.extra_keys:
                self.extra_keys.append(k)

        for key, feature in self._info["features"].items():
            shape: list[int] = feature["shape"]
            dtype: str = feature["dtype"]

            if observation_regexp.fullmatch(key):
                self.features[key] = Feature(
                    shape=(self.n_observation, *shape),
                    dtype=dtype if dtype not in ["video", "image"] else "uint8",
                )

                if dtype != "video":
                    self.observation_data_keys.append(key)
                else:
                    self.observation_video_keys.append(key)

                continue

            if action_regexp.fullmatch(key):
                self.features[key] = Feature(
                    shape=(self.n_action, *shape),
                    dtype=dtype,
                )

                self.action_keys.append(key)
                continue

            if key in self.extra_keys:
                self.features[key] = Feature(
                    shape=tuple(shape),
                    dtype=dtype,
                )

        self.features.update(
            observation_is_pad=Feature(shape=(self.n_observation,), dtype="bool"),
            action_is_pad=Feature(shape=(self.n_action,), dtype="bool"),
        )

        if len(self.observation_data_keys) + len(self.observation_video_keys) == 0:
            raise ValueError("Observation must not be empty.")

        if len(self.action_keys) == 0:
            raise ValueError("Action must not be empty.")

        self._observation_columns: str = (
            '"' + '","'.join(self.observation_data_keys) + '"'
        )
        self._action_columns: str = '"' + '","'.join(self.action_keys) + '"'
        self._extra_columns: str = (
            ('"' + '","'.join(self.extra_keys) + '"')
            if len(self.extra_keys) > 0
            else ""
        )

        try:
            self._con.query(f"""
            CREATE TEMP TABLE "tasks" AS (FROM '{self._base}/meta/tasks.jsonl');
            """)
        except duckdb.HTTPException:
            # e.g. lerobot/pusht has strange names like [task_index, __index_level_0__]
            # https://huggingface.co/datasets/lerobot/pusht/blob/main/meta/tasks.parquet
            self._con.query(f"""
            CREATE TEMP TABLE "tasks" AS (
              SELECT
                "task_index",
                COLUMNS('[^(task_index)]') AS "tasks",
              FROM '{self._base}/meta/tasks.parquet'
            );
            """)

        where: str = (
            f"""WHERE "episode_index" IN {episodes}""" if episodes is not None else ""
        )
        self._con.query(f"""
        CREATE TEMP TABLE "episodes" AS (
          SELECT
            *,
            sum("length"::BIGINT) OVER (
              ORDER BY "episode_index" ASC
              RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS "cumsum_length",
          FROM '{self._base}/meta/episodes.jsonl'
          {where}
        );
        """)

        self._length: int = (
            self._con.query("""
          SELECT max("cumsum_length"::BIGINT) AS "length"
          FROM "episodes";
        """)
            .fetch_arrow_table()["length"][0]
            .as_py()
        )

        self._load_stats()

    @property
    def version(self) -> str:
        """
        LeRobot Dataset format version
        """
        return self._info["codebase_version"]

    @property
    def stats(self) -> dict[str, LeRobotDatasetDataStat | LeRobotDatasetImageStat]:
        keys = (
            self.observation_data_keys + self.observation_video_keys + self.action_keys
        )

        s: dict[str, LeRobotDatasetDataStat | LeRobotDatasetImageStat] = {}
        for k in keys:
            si = self._con.query(f'SELECT "{k}".* FROM stats;').fetch_arrow_table()
            s[k] = {
                c: to_array(si[c], self.features[k].dtype).squeeze(0)
                for c in si.column_names
            }

        return s

    def _load_stats(self) -> None:
        version: str = self.version

        keys = (
            self.observation_data_keys + self.observation_video_keys + self.action_keys
        )
        match version:
            case "v2.0":
                columns = '","'.join(keys)

                self._con.query(f"""
                CREATE OR REPLACE TEMP TABLE stats AS (
                  SELECT "{columns}"
                  FROM (
                    SELECT "stats".*
                    FROM '{self._base}/meta/stats.json'
                  )
                );
                """)
            case "v2.1":
                columns = ",".join(
                    (
                        agg_image_stats(k) + f' AS "{k}"'
                        if len(self.features[k].shape) > 2
                        else agg_data_stats(k, self.features[k].shape[1]) + f' AS "{k}"'
                        for k in keys
                    )
                )

                self._con.query(f"""
                CREATE OR REPLACE TEMP TABLE stats AS (
                  SELECT {columns}
                  FROM '{self._base}/meta/episodes_stats.jsonl'
                );
                """)
            case _:
                raise NotImplementedError(f"Version {version} is not supported.")

    def _data_path(self, episode_index: int) -> str:
        return (
            self._base
            + "/"
            + self._info["data_path"].format(
                episode_chunk=episode_index // self._info["chunks_size"],
                episode_index=episode_index,
            )
        )

    def _video_path(self, episode_index: int, video_key: str) -> str:
        return (
            self._base
            + "/"
            + self._info["video_path"].format(
                episode_chunk=episode_index // self._info["chunks_size"],
                video_key=video_key,
                episode_index=episode_index,
            )
        )

    def __getitem__(
        self,
        idx: Integer[Any, ""] | int,
    ) -> dict[str, Shaped[np.ndarray, "..."]]:
        if not isinstance(idx, int):
            idx = np.asarray(idx).item(0)

        ep: dict[str, np.ndarray] = self._con.query(f"""
        SELECT
          e."episode_index" AS "episode_index",
          (b."idx" - (e."cumsum_length" - e."length"))::BIGINT AS "frame_index",
        FROM (SELECT {idx}::BIGINT AS "idx") b
        ASOF JOIN "episodes" e
        ON b."idx" < e."cumsum_length";
        """).fetchnumpy()

        episode_index: int = ep["episode_index"].item(0)
        frame_index: int = ep["frame_index"].item(0)
        assert isinstance(episode_index, int)
        assert isinstance(frame_index, int)

        data_path = self._data_path(episode_index)

        observation = query_data(
            self._con,
            self.features,
            columns=self._observation_columns,
            data_path=data_path,
            pad_column="observation_is_pad",
            frame_index=frame_index,
            n_steps=self.n_observation,
            nested=True,
        )

        action = query_data(
            self._con,
            self.features,
            columns=self._action_columns,
            data_path=data_path,
            pad_column="action_is_pad",
            frame_index=frame_index,
            n_steps=self.n_action,
            nested=True,
        )

        extra = query_data(
            self._con,
            self.features,
            columns=self._extra_columns,
            data_path=data_path,
            pad_column=None,
            frame_index=frame_index,
            n_steps=1,
            nested=False,
        )

        timestamp = query_data(
            self._con,
            self.features,
            columns='"timestamp"',
            data_path=data_path,
            pad_column=None,
            frame_index=frame_index,
            n_steps=self.n_observation,
            nested=False,
        )["timestamp"]
        assert timestamp.ndim == 1, f"timestamp.ndim must be 1, but {timestamp.ndim}"
        assert len(timestamp) > 0, f"timestamp is empty: {frame_index}"

        videos = {
            video_key: query_video(
                self._con,
                video_path=self._video_path(
                    episode_index=episode_index,
                    video_key=video_key,
                ),
                timestamp=timestamp,
            )
            for video_key in self.observation_video_keys
        }

        item = {
            **observation,
            **action,
            **extra,
            **videos,
        }
        return item

    def __len__(self) -> int:
        return self._length
