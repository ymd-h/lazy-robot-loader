import numpy as np

from lazy_robot_loader.lerobot import LeRobotDataset


def test_pushT(benchmark):
    data = LeRobotDataset(repo_id="lerobot/pusht@v2.1")

    def run(i):
        return data[i]

    benchmark(run, np.asarray(0))


def test_pushT_5obs(benchmark):
    data = LeRobotDataset(
        repo_id="lerobot/pusht@v2.1",
        n_observation=5,
    )

    def run(i):
        return data[i]

    benchmark(run, np.asarray(0))
