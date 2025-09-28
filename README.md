# Lazy DataLoader for LeRobot (lazy-robot-loader)

> [!WARNING]<br/>
> This package is still under development.

`lezy-robot-loader` provides functionality
to load (and download if necessary) [LeRobot](https://huggingface.co/lerobot) Dataset lazily.


## Features
- Version
  - [X] v2.0
  - [X] v2.1
  - [ ] v3.0
- Data
  - [X] Signal (Observation, Action, etc.) in Parquet
  - [X] Video in MP4
- Meta
  - [X] Info
  - [X] Episodes
  - [ ] Stats
- Source
  - [X] Hugging Face
  - [X] Local Path

## Prerequisite
- [FFmpeg](https://ffmpeg.org/)
  - To decode MP4 videos through [PyAV](https://pyav.basswood-io.com/)


## Internal
`lazy-robot-loader` utilizes [DuckDB](https://duckdb.org/) with
its extensions [httpfs](https://duckdb.org/docs/stable/core_extensions/httpfs/overview.html)
and [cache_httpfs](https://duckdb.org/community_extensions/extensions/cache_httpfs.html).

It enables us to query data hosted at Hugging Face directly.
(Ref. [1](https://duckdb.org/docs/stable/core_extensions/httpfs/hugging_face.html),
[2](https://huggingface.co/docs/hub/datasets-duckdb))
