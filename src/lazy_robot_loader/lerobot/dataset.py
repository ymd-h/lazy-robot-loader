from collections.abc import Iterable
from logging import getLogger
import os
import pathlib
import re
from typing import cast, Any

import duckdb
import numpy as np

from jaxtyping import Integer, Shaped

from lazy_robot_loader.core import Feature
from lazy_robot_loader.lerobot.core import (
    LeRobotDatasetInfo,
    LeRobotDatasetDataStat,
    LeRobotDatasetImageStat,
)
from lazy_robot_loader.lerobot.internal.query import agg_data_stats, agg_image_stats
from lazy_robot_loader.lerobot.internal.functional import (
    get_stat,
    query_video,
    query_data,
)

__all__ = [
    "LeRobotDataset",
]

logger = getLogger(__file__)


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

        logger.debug("Cache Dir: %s", cache_dir)
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

        logger.info("LeRobot Dataset: %s", self._base)

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
        logger.info("LeRobot Dataset Version: %s", version)
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

        logger.debug("Observation Data Keys: %s", self.observation_data_keys)
        logger.debug("Observation Video Keys: %s", self.observation_video_keys)
        logger.debug("Action Keys: %s", self.action_keys)
        logger.debug("Extra Keys: %s", self.extra_keys)

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

        return {k: get_stat(self._con, k, self.features[k]) for k in keys}

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
