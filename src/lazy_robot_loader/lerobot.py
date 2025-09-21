from __future__ import annotations
import io
import os
import pathlib
from typing import cast, Any, TYPE_CHECKING, TypedDict

import av
import duckdb
import numpy as np

if TYPE_CHECKING:
    from jaxtyping import Integer, Shaped


class LeRobotDatasetFeature(TypedDict):
    dtype: str
    shape: list[int]
    names: list[str] | None


LeRobotDatasetVideoFeatureInfo = TypedDict(
    "LeRobotDatasetVideoFeatureInfo",
    {
        "video.fps": int,
        "video.codec": str,
        "video.pix_fmt": str,
        "video.is_depth_map": bool,
        "has_audio": bool,
    },
)


class LeRobotDatasetVideoFeature(LeRobotDatasetFeature):
    info: LeRobotDatasetVideoFeatureInfo


class LeRobotDatasetInfo(TypedDict):
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    splits: dict[str, str]
    data_path: str
    video_path: str
    features: dict[str, LeRobotDatasetFeature]


class LeRobotDataset:
    def __init__(
        self,
        *,
        repo_id: str | None = None,
        local_path: str | pathlib.Path | None = None,
        episodes: tuple[int, ...] | None = None,
        n_observation: int = 1,
        n_action: int = 1,
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

        if cache_dir is None:
            cache_dir = pathlib.Path.home() / ".cache/lrl"

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
                .arrow()
                .to_pydict()
                .items()
            },
        )
        assert isinstance(self._info, dict), f"{self._info}"

        version = self._info["codebase_version"]
        if version not in ["v2.0", "v2.1"]:
            raise NotImplementedError(
                f"Only v2.0 and v2.1 are supported, got {version}"
            )

        self.features = {
            k: {
                "shape": (self.n_observation, *v["shape"])
                if k.startswith("observation")
                else (self.n_action, *v["shape"])
                if k.startswith("action")
                else v["shape"],
                "dtype": v["dtype"]
                if v["dtype"] not in ["video", "image"]
                else "uint8",
            }
            for k, v in self._info["features"].items()
        } | {
            "observation_is_pad": {"shape": (self.n_observation,), "dtype": "bool"},
            "action_is_pad": {"shape": (self.n_action,), "dtype": "bool"},
        }

        self._video_keys = [
            k for k, v in self._info["features"].items() if v["dtype"] == "video"
        ]
        self._extra_keys = [
            k
            for k in self.features.keys()
            if (not k.startswith("observation")) and (not k.startswith("action"))
        ]

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
            sum("length") OVER (
              ORDER BY "episode_index" ASC
              RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS "cumsum_length",
          FROM '{self._base}/meta/episodes.jsonl'
          {where}
        );
        """)

        self._length: int = (
            self._con.query("""
          SELECT max("cumsum_length")::BIGINT AS "length"
          FROM "episodes";
        """)
            .arrow()["length"][0]
            .as_py()
        )

    def __getitem__(
        self, idx: Integer[Any, ""]
    ) -> dict[str, Shaped[np.ndarray, "..."]]:
        ep: dict[str, np.ndarray] = self._con.query(f"""
        SELECT
          e."episode_index" AS "episode_index",
          e."length" AS "length",
          b."idx" - (e."cumsum_length" - e."length") AS "frame_index",
        FROM (SELECT unnest([{idx.item()}]) AS "idx") b
        ASOF JOIN "episodes" e
        ON b."idx" < e."cumsum_length";
        """).fetchnumpy()

        episode_index: int = ep["episode_index"][0]
        episode_chunk: int = episode_index // self._info["chunks_size"]

        data_path = (
            self._base
            + "/"
            + self._info["data_path"].format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
        )

        f_idx: int = ep["frame_index"][0]

        observation = self._con.query(f"""
        SELECT
          COLUMNS('observation.*'),
          e."frame_index" != d."frame_index" AS "observation_is_pad",
        FROM (
          SELECT {f_idx} + unnest(range({self.n_observation})) AS "frame_index",
        ) e ASOF JOIN '{data_path}' d
        ON e."frame_index" >= d."frame_index";
        """).arrow()

        action = self._con.query(f"""
        SELECT
          COLUMNS('action.*'),
          e."frame_index" != d."frame_index" AS "action_is_pad",
        FROM (
          SELECT {f_idx} + unnest(range({self.n_action})) AS "frame_index"
        ) e ASOF JOIN '{data_path}' d
        ON e."frame_index" >= d."frame_index";
        """).arrow()

        item = (
            {
                c: np.asarray(
                    observation[c].to_pylist(),
                    dtype=self.features[c].dtype,
                )
                for c in observation.column_names
            }
            | {
                c: np.asarray(
                    action[c].to_pylist(),
                    dtype=self.features[c].dtype,
                )
                for c in action.column_names
            }
            | {
                k: np.asarray(v, dtype=self.features[k].dtype)
                for k, v in self._con.query(f"""
                SELECT "{'","'.join(self._extra_keys)}",
                FROM '{data_path}'
                WHERE "frame_index" = {f_idx};
                """)
                .fetchnumpy()
                .items()
            }
        )

        videos: dict[str, Integer[np.ndarray, "{self.n_observation} H W C"]] = {}
        timestamp = self._con.query(f"""
        SELECT "timestamp"
        FROM (
          SELECT {f_idx} + unnest(range({self.n_observation})) AS "frame_index"
        ) e ASOF JOIN '{data_path}' d
        ON e."frame_index" >= d."frame_index";
        """).fetchnumpy()["timestamp"]
        assert len(timestamp) > 0, f"timestamp is empty: {f_idx}"

        for k in self._video_keys:
            video_path = (
                self._base
                + "/"
                + self._info["video_path"].format(
                    episode_chunk=episode_chunk,
                    video_key=k,
                    episode_index=episode_index,
                )
            )

            v = (
                io.BytesIO(
                    self._con.query(f"""
                    SELECT "content" FROM read_blob('{video_path}');
                    """)
                    .arrow()["content"][0]
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
                imgs: list[Integer[np.ndarray, "H W C"]] = []
                img: Integer[np.ndarray, "H W C"] | None = None
                for t in timestamp:
                    for f in it:
                        if t <= f.time:
                            img = f.to_ndarray(format="rgb24")
                            imgs.append(img)
                            break
                    else:
                        assert img is not None
                        imgs.append(img)

                videos[k] = np.stack(imgs)

        item |= videos
        return item

    def __len__(self) -> int:
        return self._length
